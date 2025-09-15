import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    HAS_TORCH = False

LSTM_APPEAR_HIDDEN = int(os.getenv("LSTM_APPEAR_HIDDEN", "128"))

if HAS_TORCH:
    class SeqTrackLSTM(nn.Module):
        """
        SeqTrack-LSTM: BiLSTM motion model with optional appearance fusion.
        TorchScriptable; inference-focused. Two variants:
        - variant 'A': motion-only features
        - variant 'B': motion + appearance (ReID/HSV via FiLM-like gating)
        """

        def __init__(self,
                     input_size: int = 8,
                     hidden_size: int = 128,
                     num_layers: int = 2,
                     variant: str = 'A',
                     device: str = 'cuda',
                     fp16: bool = True):
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.num_layers = int(num_layers)
            self.variant = (variant or 'A').upper()
            self.input_size = int(input_size)
            self.device = torch.device(
                device if device in ('cpu', 'cuda') else ('cuda' if torch.cuda.is_available() else 'cpu'))
            self.fp16 = bool(fp16) and (self.device.type == 'cuda')
            # Core BiLSTM backbone
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                                batch_first=True, bidirectional=True, dropout=0.1)
            out_dim = self.hidden_size * 2
            # Optional appearance projection for variant B
            if self.variant == 'B':
                self.reid_proj = nn.Linear(2048, 128)  # will adapt at runtime if dim differs
                self.hist_proj = nn.Linear(64, 32)  # simple placeholder, accepts small HSV vector
                self.film_gamma = nn.Linear(out_dim + 128 + 32, out_dim)
                self.film_beta = nn.Linear(out_dim + 128 + 32, out_dim)
            # Heads
            self.head_delta = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 4)  # dx, dy, dlogw, dlogh
            )
            self.head_cont = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 1), nn.Sigmoid()
            )
            self.head_score = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 1)
            )
            self.head_logvar = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 4)  # log-variance for dx,dy,dlogw,dlogh
            )
            self.to(self.device)
            if self.fp16 and self.device.type == 'cuda':
                self.half()

        def _maybe_adapt_proj(self, reid_dim: int, hist_dim: int):
            # Recreate projection layers if dimensions differ
            if self.variant != 'B':
                return
            if hasattr(self, 'reid_proj') and isinstance(self.reid_proj, nn.Linear):
                if self.reid_proj.in_features != reid_dim:
                    self.reid_proj = nn.Linear(reid_dim, 128).to(self.device)
                    if self.fp16 and self.device.type == 'cuda':
                        self.reid_proj = self.reid_proj.half()
            if hasattr(self, 'hist_proj') and isinstance(self.hist_proj, nn.Linear):
                if self.hist_proj.in_features != hist_dim:
                    self.hist_proj = nn.Linear(hist_dim, 32).to(self.device)
                    if self.fp16 and self.device.type == 'cuda':
                        self.hist_proj = self.hist_proj.half()

        def forward(self, x: torch.Tensor,
                    reid_vec: Optional[torch.Tensor] = None,
                    hist_vec: Optional[torch.Tensor] = None) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # x: (B, T, F)
            B = x.size(0)
            h, _ = self.lstm(x)
            h_last = h[:, -1, :]  # (B, 2*hidden)
            if self.variant == 'B' and (reid_vec is not None or hist_vec is not None):
                # Normalize reid to unit length when present
                phi = [h_last]
                if reid_vec is not None and reid_vec.numel() > 0:
                    rv = torch.nn.functional.normalize(reid_vec, p=2, dim=1)
                    self._maybe_adapt_proj(rv.size(1), int(hist_vec.size(1) if hist_vec is not None else 0))
                    rvp = self.reid_proj(rv)
                    phi.append(rvp)
                if hist_vec is not None and hist_vec.numel() > 0:
                    hsp = self.hist_proj(hist_vec)
                    phi.append(hsp)
                cat = torch.cat(phi, dim=1)
                gamma = self.film_gamma(cat)
                beta = self.film_beta(cat)
                h_last = (1.0 + gamma) * h_last + beta
            delta = self.head_delta(h_last)
            cont = self.head_cont(h_last)
            score = self.head_score(h_last)
            logv = self.head_logvar(h_last)
            return delta, cont, score, logv

        def _app_mem_from_seq(self, reid_seq: list, hist_seq: list):
            """
            Build a compact appearance memory vector from short sequences of reid/hist.
            Lazy-creates a 1-layer GRU and a linear projection to keep footprint tiny.
            Returns a torch.Tensor of shape (1, H) on self.device, or None if not available.
            """
            if not LSTM_APPEAR_ENABLE:
                return None
            try:
                T = max(len(reid_seq or []), len(hist_seq or []))
                if T <= 0:
                    return None

                # Build per-timestep concatenated vectors (np.float32)
                Xs = []
                # Determine dims (dynamic; we’ll adapt on the fly)
                r_dim = int((reid_seq[0].size if (reid_seq and isinstance(reid_seq[0], np.ndarray)) else 0))
                h_dim = int((hist_seq[0].size if (hist_seq and isinstance(hist_seq[0], np.ndarray)) else 0))
                if r_dim == 0 and h_dim == 0:
                    return None

                for i in range(T):
                    r = reid_seq[i] if (reid_seq and i < len(reid_seq)) else None
                    h = hist_seq[i] if (hist_seq and i < len(hist_seq)) else None
                    if r is None and h is None:
                        continue
                    if r is None:
                        r = np.zeros((r_dim,), np.float32)
                    if h is None:
                        h = np.zeros((h_dim,), np.float32)
                    Xs.append(np.concatenate([r.astype(np.float32), h.astype(np.float32)], axis=0))

                if not Xs:
                    return None

                xs = torch.from_numpy(np.stack(Xs, axis=0)).unsqueeze(0).to(self.device)  # (1, T, Din)
                Din = xs.size(-1)

                # Lazy-create or re-create if Din changes
                app_hidden = int(LSTM_APPEAR_HIDDEN)
                if not hasattr(self, "_app_proj") or not hasattr(self, "_app_rnn") \
                   or (hasattr(self._app_proj, "in_features") and int(self._app_proj.in_features) != int(Din)):
                    self._app_proj = nn.Linear(Din, app_hidden).to(self.device)
                    self._app_rnn  = nn.GRU(input_size=app_hidden, hidden_size=app_hidden,
                                            num_layers=1, batch_first=True).to(self.device)
                    if self.fp16 and self.device.type == 'cuda':
                        self._app_proj = self._app_proj.half()
                        self._app_rnn  = self._app_rnn.half()

                xs = self._app_proj(xs)  # (1, T, H)
                _, h_app = self._app_rnn(xs)  # h_app: (1, 1, H)
                return h_app[-1]  # (1, H)
            except Exception:
                return None

        def predict(self, track_window: Dict[str, Any]) -> Dict[str, Any]:
            # Build motion features from minimal info; tolerate missing keys.
            try:
                centers = track_window.get('centers', []) or []
                boxes = track_window.get('boxes', []) or []
                sizes = track_window.get('sizes', []) or []
                confs = track_window.get('conf', []) or []
                reid = track_window.get('reid', None)
                hist = track_window.get('hist', None)
                reid_seq = track_window.get('reid_seq', []) or []
                hist_seq = track_window.get('hist_seq', []) or []

                # Require at least 2 timesteps
                if centers is None or len(centers) < 2:
                    return {"delta": (0.0, 0.0, 0.0, 0.0), "cont": 0.5, "score": 0.0,
                            "cov": np.diag([25.0, 25.0, 0.04, 0.04]).astype(np.float32)}
                # Compute per-step features (T-1, F)
                feats = []
                prev_iou = 0.0
                for i in range(1, len(centers)):
                    (cx0, cy0) = centers[i - 1]
                    (cx1, cy1) = centers[i]
                    dx = float(cx1 - cx0);
                    dy = float(cy1 - cy0)
                    # size/log-size
                    if sizes and i < len(sizes):
                        w1, h1 = sizes[i]
                        w0, h0 = sizes[i - 1] if i - 1 < len(sizes) else (max(1.0, w1), max(1.0, h1))
                    elif boxes and i < len(boxes):
                        x1, y1, x2, y2 = boxes[i]
                        w1 = max(1.0, x2 - x1);
                        h1 = max(1.0, y2 - y1)
                        if i - 1 < len(boxes):
                            xp, yp, xq, yq = boxes[i - 1]
                            w0 = max(1.0, xq - xp);
                            h0 = max(1.0, yq - yp)
                        else:
                            w0, h0 = w1, h1
                    else:
                        w1 = w0 = 1.0;
                        h1 = h0 = 1.0
                    logw1 = float(np.log(max(1.0, w1)));
                    logh1 = float(np.log(max(1.0, h1)))
                    logw0 = float(np.log(max(1.0, w0)));
                    logh0 = float(np.log(max(1.0, h0)))
                    dlogw = logw1 - logw0;
                    dlogh = logh1 - logh0
                    spd = float(np.hypot(dx, dy) + 1e-6)
                    heading = float(np.arctan2(dy, dx))
                    conf = float(confs[i]) if i < len(confs) else 1.0
                    if boxes and (i - 1 < len(boxes)) and (i < len(boxes)):
                        # approximate IoU to previous
                        prev_iou = float(_iou_xyxy(tuple(boxes[i - 1]), tuple(boxes[i])))
                    feats.append([dx, dy, logw1, logh1, spd, heading, conf, prev_iou])
                X = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
                if self.fp16 and self.device.type == 'cuda':
                    X = X.half()
                rv = None;
                hv = None
                if self.variant == 'B':
                    if isinstance(reid, np.ndarray) and reid.size > 0:
                        rv = torch.from_numpy(reid).to(self.device)
                        if rv.dim() == 1:
                            rv = rv.unsqueeze(0)
                        rv = rv.half() if (self.fp16 and self.device.type == 'cuda') else rv.float()
                    if isinstance(hist, np.ndarray) and hist.size > 0:
                        hv = torch.from_numpy(hist).to(self.device)
                        if hv.dim() == 1:
                            hv = hv.unsqueeze(0)
                        hv = hv.half() if (self.fp16 and self.device.type == 'cuda') else hv.float()
                with torch.no_grad():
                    d, c, s, logv = self.forward(X, rv, hv)

                # Build appearance memory from short sequences and fuse into reid channel
                app_mem = self._app_mem_from_seq(reid_seq, hist_seq)
                if app_mem is not None:
                    if rv is not None:
                        rv = torch.cat([rv, app_mem], dim=1)
                    else:
                        rv = app_mem

                d = d.float().squeeze(0).cpu().numpy().tolist()
                c = float(c.float().squeeze(0).cpu().numpy().reshape(-1)[0])
                s = float(s.float().squeeze(0).cpu().numpy().reshape(-1)[0])
                logv = logv.float().squeeze(0).cpu().numpy()
                var = np.exp(logv)
                # Clamp extreme variances
                var = np.clip(var, a_min=np.array([1.0, 1.0, 1e-4, 1e-4], dtype=np.float32),
                              a_max=np.array([400.0, 400.0, 0.25, 0.25], dtype=np.float32))
                cov = np.diag(var.astype(np.float32))
                out = {"delta": (float(d[0]), float(d[1]), float(d[2]), float(d[3])),
                       "cont": c, "score": s, "cov": cov}
                try:
                    if app_mem is not None:
                        out["app_mem"] = app_mem.float().detach().cpu().numpy().squeeze().astype(np.float32)
                except Exception:
                    pass
                return out

            except Exception:
                return {"delta": (0.0, 0.0, 0.0, 0.0), "cont": 0.5, "score": 0.0,
                        "cov": np.diag([25.0, 25.0, 0.04, 0.04]).astype(np.float32)}
else:
    class SeqTrackLSTM(object):  # CPU/dummy fallback when torch is not available
        def __init__(self, *args, **kwargs):
            self.device = 'cpu'

        def predict(self, track_window: Dict[str, Any]) -> Dict[str, Any]:
            return {"delta": (0.0, 0.0, 0.0, 0.0), "cont": 0.5, "score": 0.0,
                    "cov": np.diag([25.0, 25.0, 0.04, 0.04]).astype(np.float32)}

