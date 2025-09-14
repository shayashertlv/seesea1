from typing import List, Optional
import os

import numpy as np
import cv2

# Optional dependencies
try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

# Heavy (optional) deps
try:
    import open_clip  # type: ignore
    HAS_OPENCLIP = True
except Exception:
    HAS_OPENCLIP = False
try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except Exception:
    HAS_PIL = False

from reid_backbones import BACKBONE_LOADERS

# Normalization tensors (CPU) hoisted to avoid per-call allocation
if HAS_TORCH:
    _NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    _NORM_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
else:
    _NORM_MEAN = None
    _NORM_STD = None


class ReIDExtractor:
    """
    Pluggable ReID embedding extractor.
    forward(list_of_crops_bgr, batch_size) -> np.ndarray of shape (N, D), L2-normalized.
    Backends:
    - osnet (default): via torchreid or torchvision osnet_x1_0; fallback to resnet50 features.
    - fastreid_r50: requires fastreid installed; if unavailable, raises.
    - dinov2_vits14: via timm ViT small dinov2 global pooled embedding.
    - dinov2_vitl14: via timm ViT large patch14 DINOv2.
    - clip_vitl14 / clip_vith14: via open_clip pretrained="openai".
    - fusion: concat of CLIP-L/14 and DINOv2-L/14 (then optional PCA).
    """
    def __init__(self, backend: str = "osnet", device: str = "auto", fp16: bool = False):
        self.backend = (backend or "osnet").lower()
        self.fp16 = bool(fp16)
        self.model = None
        self.model_clip = None
        self.model_dino = None
        self.device = None
        self.input_size = (256, 128)  # default for osnet/person
        self.is_vit_square = False
        # TTA / PCA from env (fail-open defaults)
        try:
            self.tta_mode = int(os.getenv("REID_TTA", "0"))
        except Exception:
            self.tta_mode = 0
        try:
            self.pca_dim = int(os.getenv("REID_PCA_DIM", "0"))
        except Exception:
            self.pca_dim = 0
        # PCA cache (class-level shared)
        if not hasattr(ReIDExtractor, "_pca_ready"):
            ReIDExtractor._pca_ready = False  # type: ignore[attr-defined]
            ReIDExtractor._pca_mean = None    # type: ignore[attr-defined]
            ReIDExtractor._pca_comp = None    # type: ignore[attr-defined]
            ReIDExtractor._pca_buf = []       # type: ignore[attr-defined]
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available for ReIDExtractor")
        # Device selection
        if device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "mps":
            # explicit MPS request
            use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            self.device = torch.device("mps") if use_mps else torch.device("cpu")
        else:
            # auto: prefer CUDA, then MPS, then CPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        # Load backend with graceful fallbacks (no internet required)
        load_ok = False
        load_fn = BACKBONE_LOADERS.get(self.backend, BACKBONE_LOADERS.get("osnet"))
        try:
            if load_fn is not None:
                load_fn(self)
                load_ok = True
        except Exception:
            load_ok = False
        # 2) If still not ok, final fallback to torchvision resnet50 with no weights (offline-safe)
        if not load_ok or (self.model is None and self.backend != "fusion"):
            try:
                import torchvision
                model = torchvision.models.resnet50(weights=None)
                model.fc = torch.nn.Identity()
                self.input_size = (224, 224)
                model.eval().to(self.device)
                self.model = model
                load_ok = True
            except Exception:
                load_ok = False
        # Ensure fp16 on CUDA only
        if load_ok and self.fp16 and self.device.type == "cuda" and self.model is not None:
            try:
                self.model = self.model.half()
            except Exception:
                pass
        if not load_ok or (self.model is None and self.backend != "fusion"):
            raise RuntimeError("Failed to initialize any ReID backbone (torchvision/timm/osnet/open_clip)")

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        # Convert BGR->RGB and resize appropriately (ImageNet normalization)
        img = bgr[:, :, ::-1]  # RGB
        h, w = img.shape[:2]
        if self.is_vit_square:
            # pad to square then resize to 224x224
            size = max(h, w)
            pad = np.zeros((size, size, 3), dtype=img.dtype)
            pad[:h, :w] = img
            img = pad
            target = (self.input_size[0], self.input_size[1])
        else:
            target = (self.input_size[1], self.input_size[0])  # (W,H)
        img = cv2.resize(img, target, interpolation=cv2.INTER_LINEAR)  # type: ignore[name-defined]
        ten = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # Use hoisted normalization tensors (CPU)
        mean = _NORM_MEAN if _NORM_MEAN is not None else torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = _NORM_STD if _NORM_STD is not None else torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        ten = (ten - mean) / std
        return ten

    def _preprocess_clip(self, bgr: np.ndarray) -> Optional[torch.Tensor]:
        # Use open_clip-provided preprocessing if available
        try:
            if not HAS_OPENCLIP or getattr(self, "clip_tf", None) is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if HAS_PIL and Image is not None:
                pil = Image.fromarray(rgb)
                ten = self.clip_tf(pil)
            else:
                # fallback: resize to 224, no CLIP normalization (still works reasonably)
                img = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                ten = torch.from_numpy(img).permute(2,0,1).float() / 255.0
            return ten
        except Exception:
            return None

    def _pca_reduce_cached(self, vecs: np.ndarray, out_dim: int) -> np.ndarray:
        # Fit once per process using simple SVD PCA on reservoir
        if out_dim <= 0:
            return vecs
        try:
            V = vecs
            if V.ndim == 1:
                V = V[None, :]
            if not getattr(ReIDExtractor, "_pca_ready", False):
                buf = getattr(ReIDExtractor, "_pca_buf", [])
                # reservoir limit ~5000
                max_buf = 5000
                for i in range(min(V.shape[0], 64)):
                    buf.append(V[i].astype(np.float32))
                if len(buf) > max_buf:
                    buf = buf[-max_buf:]
                ReIDExtractor._pca_buf = buf  # type: ignore[attr-defined]
                # Fit when enough samples
                if len(buf) >= max(256, out_dim + 8):
                    X = np.stack(buf, axis=0).astype(np.float32)
                    mu = X.mean(axis=0, keepdims=True)
                    Xc = X - mu
                    # economy SVD
                    try:
                        U, S, VT = np.linalg.svd(Xc, full_matrices=False)
                        comp = VT[:out_dim].T  # D x out_dim
                    except Exception:
                        # fallback to eigen on covariance (may be slower)
                        C = (Xc.T @ Xc) / max(1, Xc.shape[0]-1)
                        eigvals, eigvecs = np.linalg.eigh(C)
                        order = np.argsort(eigvals)[::-1][:out_dim]
                        comp = eigvecs[:, order]
                    ReIDExtractor._pca_mean = mu.astype(np.float32)  # type: ignore[attr-defined]
                    ReIDExtractor._pca_comp = comp.astype(np.float32)  # type: ignore[attr-defined]
                    ReIDExtractor._pca_ready = True  # type: ignore[attr-defined]
            # Transform using cached params if available
            if getattr(ReIDExtractor, "_pca_ready", False):
                mu = getattr(ReIDExtractor, "_pca_mean", None)
                comp = getattr(ReIDExtractor, "_pca_comp", None)
                if mu is not None and comp is not None:
                    X = V.astype(np.float32)
                    Xc = X - mu
                    Y = Xc @ comp  # (N,out_dim)
                    return Y if vecs.ndim > 1 else Y[0]
            return vecs
        except Exception:
            return vecs

    def _embed_one(self, crop: np.ndarray) -> Optional[np.ndarray]:
        try:
            dev = self.device
            with torch.no_grad():
                if self.backend in ("clip_vitl14", "clip_vith14"):
                    outs = []
                    # Base
                    t = self._preprocess_clip(crop)
                    if t is not None:
                        outs.append(t)
                    # Flip TTA
                    if self.tta_mode in (1,3):
                        t2 = self._preprocess_clip(cv2.flip(crop, 1))
                        if t2 is not None:
                            outs.append(t2)
                    # Padded TTA
                    if self.tta_mode == 3:
                        pad = cv2.copyMakeBorders(crop, 8,8,8,8, cv2.BORDER_CONSTANT, value=(0,0,0))
                        t3 = self._preprocess_clip(pad)
                        if t3 is not None:
                            outs.append(t3)
                    if not outs:
                        return None
                    batch = torch.stack(outs, dim=0)
                    if self.fp16 and dev.type == "cuda":
                        batch = batch.half()
                    batch = batch.to(dev)
                    if hasattr(self.model, 'encode_image'):
                        emb = self.model.encode_image(batch)
                    else:
                        emb = self.model(batch)
                    emb = emb.float()
                    emb = emb.mean(dim=0, keepdim=True) if len(outs) > 1 else emb
                    v = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    # Optional PCA then L2
                    if self.pca_dim > 0:
                        v = self._pca_reduce_cached(v, self.pca_dim)
                    n = float(np.linalg.norm(v))
                    if n > 1e-12:
                        v = v / n
                    return v.astype(np.float32)
                elif self.backend in ("dinov2_vits14", "dinov2_vitl14"):
                    outs = []
                    # Base
                    outs.append(self._preprocess(crop))
                    # Flip TTA
                    if self.tta_mode in (1,3):
                        outs.append(self._preprocess(cv2.flip(crop, 1)))
                    # Padded TTA
                    if self.tta_mode == 3:
                        pad = cv2.copyMakeBorders(crop, 8,8,8,8, cv2.BORDER_CONSTANT, value=(0,0,0))
                        outs.append(self._preprocess(pad))
                    batch = torch.stack(outs, dim=0).to(dev)
                    if self.fp16 and dev.type == "cuda":
                        batch = batch.half()
                    emb = self.model(batch)  # type: ignore
                    emb = emb.float()
                    if isinstance(emb, (list,tuple)):
                        emb = emb[0]
                    emb = emb.mean(dim=0, keepdim=True) if len(outs) > 1 else emb
                    v = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    if self.pca_dim > 0:
                        v = self._pca_reduce_cached(v, self.pca_dim)
                    n = float(np.linalg.norm(v))
                    if n > 1e-12:
                        v = v / n
                    return v.astype(np.float32)
                elif self.backend == "fusion":
                    # CLIP part
                    # Compute explicitly for CLIP and DINO branches (avoid recursion)
                    outs_c = []
                    t = self._preprocess_clip(crop)
                    if t is not None:
                        outs_c.append(t)
                    if self.tta_mode in (1,3):
                        t2 = self._preprocess_clip(cv2.flip(crop,1))
                        if t2 is not None:
                            outs_c.append(t2)
                    if self.tta_mode == 3:
                        pad = cv2.copyMakeBorders(crop, 8,8,8,8, cv2.BORDER_CONSTANT, value=(0,0,0))
                        t3 = self._preprocess_clip(pad)
                        if t3 is not None:
                            outs_c.append(t3)
                    if not outs_c:
                        return None
                    bc = torch.stack(outs_c, dim=0).to(dev)
                    if self.fp16 and dev.type == "cuda":
                        bc = bc.half()
                    mc = self.model_clip
                    if hasattr(mc, 'encode_image'):
                        ec = mc.encode_image(bc)
                    else:
                        ec = mc(bc)
                    ec = ec.float()
                    ec = ec.mean(dim=0, keepdim=True) if len(outs_c)>1 else ec
                    vclip = ec.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    # DINO part
                    outs_d = []
                    outs_d.append(self._preprocess(crop))
                    if self.tta_mode in (1,3):
                        outs_d.append(self._preprocess(cv2.flip(crop,1)))
                    if self.tta_mode == 3:
                        pad = cv2.copyMakeBorders(crop, 8,8,8,8, cv2.BORDER_CONSTANT, value=(0,0,0))
                        outs_d.append(self._preprocess(pad))
                    bd = torch.stack(outs_d, dim=0).to(dev)
                    if self.fp16 and dev.type == "cuda":
                        bd = bd.half()
                    md = self.model_dino
                    ed = md(bd)
                    ed = ed.float()
                    if isinstance(ed, (list,tuple)):
                        ed = ed[0]
                    ed = ed.mean(dim=0, keepdim=True) if len(outs_d)>1 else ed
                    vdino = ed.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    v = np.concatenate([vclip, vdino], axis=0).astype(np.float32)
                    if self.pca_dim > 0:
                        v = self._pca_reduce_cached(v, self.pca_dim)
                    n = float(np.linalg.norm(v))
                    if n > 1e-12:
                        v = v / n
                    return v.astype(np.float32)
                else:
                    # default path (osnet/resnet/fastreid): single pass
                    ten = self._preprocess(crop).unsqueeze(0).to(dev)
                    if self.fp16 and dev.type == "cuda":
                        ten = ten.half()
                    emb = self.model(ten)  # type: ignore
                    if isinstance(emb, (list,tuple)):
                        emb = emb[0]
                    v = emb.squeeze(0).float().detach().cpu().numpy().astype(np.float32)
                    n = float(np.linalg.norm(v))
                    if n > 1e-12:
                        v = v / n
                    return v.astype(np.float32)
        except Exception:
            return None

    def forward(self, list_of_crops_bgr: List[Optional[np.ndarray]], batch_size: int = 32) -> np.ndarray:
        # For heavy backends, run per-crop with TTA and PCA. Keep batching for legacy paths.
        if self.model is None and self.backend != "fusion" and self.model_clip is None:
            return np.zeros((len(list_of_crops_bgr), 0), dtype=np.float32)
        if len(list_of_crops_bgr) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        heavy = self.backend in ("clip_vitl14","clip_vith14","fusion","dinov2_vitl14")
        outs: List[Optional[np.ndarray]] = [None] * len(list_of_crops_bgr)
        if heavy:
            for i, crop in enumerate(list_of_crops_bgr):
                if crop is None:
                    continue
                v = self._embed_one(crop)
                outs[i] = v if (isinstance(v, np.ndarray) and v.size>0) else None
        else:
            # Collect tensors
            tens = []
            idxs = []
            for i, crop in enumerate(list_of_crops_bgr):
                if crop is None:
                    continue
                try:
                    ten = self._preprocess(crop)
                    tens.append(ten)
                    idxs.append(i)
                except Exception:
                    pass
            if not tens:
                return np.zeros((len(list_of_crops_bgr), 0), dtype=np.float32)
            with torch.no_grad():
                cur = []
                cidx = []
                for i, ten in enumerate(tens):
                    cur.append(ten.unsqueeze(0))
                    cidx.append(idxs[i])
                    if len(cur) == max(1, int(batch_size)):
                        # Pin memory and transfer non-blocking on CUDA
                        batch = torch.cat(cur, dim=0)
                        if self.device.type == "cuda":
                            try:
                                batch = batch.pin_memory()
                            except Exception:
                                pass
                            batch = batch.to(self.device, non_blocking=True)
                        else:
                            batch = batch.to(self.device)
                        if self.fp16 and self.device.type == "cuda":
                            batch = batch.half()
                        emb = self.model(batch)  # type: ignore[operator]
                        if isinstance(emb, (list, tuple)):
                            emb = emb[0]
                        emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
                        vecs = emb.detach().cpu().numpy()
                        for j, vi in enumerate(cidx):
                            outs[vi] = vecs[j]
                        cur = []
                        cidx = []
                if cur:
                    # Pin memory and transfer non-blocking on CUDA
                    batch = torch.cat(cur, dim=0)
                    if self.device.type == "cuda":
                        try:
                            batch = batch.pin_memory()
                        except Exception:
                            pass
                        batch = batch.to(self.device, non_blocking=True)
                    else:
                        batch = batch.to(self.device)
                    if self.fp16 and self.device.type == "cuda":
                        batch = batch.half()
                    emb = self.model(batch)  # type: ignore[operator]
                    if isinstance(emb, (list, tuple)):
                        emb = emb[0]
                    emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
                    vecs = emb.detach().cpu().numpy()
                    for j, vi in enumerate(cidx):
                        outs[vi] = vecs[j]
        # Fill missing with zeros
        maxd = 0
        for v in outs:
            if isinstance(v, np.ndarray) and v.size > maxd:
                maxd = v.size
        arr = np.zeros((len(list_of_crops_bgr), maxd), dtype=np.float32)
        for i, v in enumerate(outs):
            if isinstance(v, np.ndarray) and v.size > 0:
                # ensure L2 normalization
                n = float(np.linalg.norm(v))
                if n > 1e-12:
                    v = v / n
                arr[i, :v.size] = v.astype(np.float32)
        return arr
