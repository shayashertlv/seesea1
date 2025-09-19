from typing import List, Optional
import os
from collections import deque
import threading

import numpy as np
import cv2

PCA_BUF_MAX = 5000

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
    import timm  # type: ignore
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False
try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except Exception:
    HAS_PIL = False

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
    - transreid_vitb16: ViT-Base TransReID style encoder via timm.
    - fusion: concat of CLIP-L/14 and DINOv2-L/14 (then optional PCA).

    The optional PCA cache is shared across instances and guarded by a
    threading lock so that this extractor can be called from multiple
    threads concurrently.
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
            ReIDExtractor._pca_buf = deque(maxlen=PCA_BUF_MAX)  # type: ignore[attr-defined]
            ReIDExtractor._pca_lock = threading.Lock()  # type: ignore[attr-defined]
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
        # 1) Try requested backend
        try:
            if self.backend == "osnet":
                self._load_osnet()
                load_ok = True
            elif self.backend == "fastreid_r50":
                self._load_fastreid_r50()
                load_ok = True
            elif self.backend == "dinov2_vits14":
                try:
                    self._load_dinov2_small()
                    load_ok = True
                except Exception:
                    self._load_osnet(); load_ok = True
            elif self.backend == "dinov2_vitl14":
                self._load_dinov2_large(); load_ok = True
            elif self.backend in ("clip_vitl14", "clip_vith14"):
                self._load_openclip(self.backend); load_ok = True
            elif self.backend == "transreid_vitb16":
                self._load_transreid_vit_base(); load_ok = True
            elif self.backend == "fusion":
                # requires both CLIP-L/14 and DINOv2-L/14
                self._load_fusion()
                load_ok = True
            else:
                self._load_osnet(); load_ok = True
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

    def _load_osnet(self):
        self.is_vit_square = False
        self.input_size = (256, 128)
        model = None
        try:
            import torchreid  # type: ignore
            model = torchreid.models.build_model('osnet_x1_0', num_classes=1, pretrained=True)  # type: ignore
            if hasattr(model, 'classifier'):
                model.classifier = torch.nn.Identity()
        except Exception:
            # Try torchvision osnet
            try:
                import torchvision
                osnet = getattr(torchvision.models, 'osnet_x1_0', None)
                if osnet is not None:
                    model = osnet(pretrained=True)
                    if hasattr(model, 'classifier'):
                        model.classifier = torch.nn.Identity()
                    elif hasattr(model, 'fc'):
                        model.fc = torch.nn.Identity()
            except Exception:
                model = None
        # Fallback to resnet50 features if osnet not available
        if model is None:
            try:
                import torchvision
                weights = getattr(torchvision.models, 'ResNet50_Weights', None)
                if weights is not None:
                    model = torchvision.models.resnet50(weights=weights.DEFAULT)
                else:
                    model = torchvision.models.resnet50(pretrained=True)
                model.fc = torch.nn.Identity()
                self.input_size = (224, 224)
            except Exception:
                model = None
        if model is None:
            raise RuntimeError("Failed to load OSNet/ResNet for ReID")
        model.eval().to(self.device)
        self.model = model
        # Ensure fp16 is only used on CUDA and set once
        if self.fp16 and self.device.type == "cuda":
            self.model = self.model.half()

    def _load_fastreid_r50(self):
        self.is_vit_square = False
        self.input_size = (256, 128)
        try:
            from fastreid.config import get_cfg  # type: ignore
            from fastreid.modeling import build_model  # type: ignore
            from fastreid.utils.checkpoint import Checkpointer  # type: ignore
        except Exception as e:
            raise RuntimeError("FastReID not installed") from e
        cfg = get_cfg()
        # BagTricks R50 default config; using built-in weights if available
        cfg.merge_from_list([
            'MODEL.BACKBONE.NAME', 'build_resnet_backbone',
            'MODEL.HEADS.IN_FEAT', '2048',
            'MODEL.META_ARCHITECTURE', 'Baseline',
        ])
        model = build_model(cfg)
        model.eval().to(self.device)
        # Attempt to load default weights if present via env
        weights_path = os.getenv('FASTREID_R50_WEIGHTS', '')
        try:
            if weights_path and os.path.exists(weights_path):
                Checkpointer(model).load(weights_path)  # type: ignore
        except Exception:
            pass
        self.model = model
        # Ensure fp16 is only used on CUDA and set once
        if self.fp16 and self.device.type == "cuda":
            self.model = self.model.half()

    def _load_dinov2_small(self):
        self.is_vit_square = True
        self.input_size = (224, 224)
        try:
            import timm
            model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
            model.eval().to(self.device)
            self.model = model
            if self.fp16 and self.device.type == "cuda":
                self.model = self.model.half()
        except Exception as e:
            raise RuntimeError("Failed to load DINOv2 ViT-S/14 via timm") from e

    def _load_dinov2_large(self):
        self.is_vit_square = True
        self.input_size = (224, 224)
        last_err = None
        try:
            import timm
            try:
                model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
            except Exception as e1:
                last_err = e1
                model = timm.create_model('vit_large_patch14_dinov2', pretrained=True, num_classes=0)
            model.eval().to(self.device)
            self.model = model
            if self.fp16 and self.device.type == "cuda":
                self.model = self.model.half()
        except Exception as e:
            if last_err is None:
                last_err = e
            raise RuntimeError("Failed to load DINOv2 ViT-L/14 via timm") from last_err

    def _load_transreid_vit_base(self):
        self.is_vit_square = True
        self.input_size = (256, 256)
        try:
            import timm
            try:
                model = timm.create_model('transreid_vit_base_patch16_224', pretrained=True, num_classes=0)
            except Exception:
                model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            model.eval().to(self.device)
            self.model = model
            if self.fp16 and self.device.type == "cuda":
                self.model = self.model.half()
        except Exception as e:
            raise RuntimeError('Failed to load TransReID ViT-B/16 via timm') from e

    def _load_openclip(self, kind: str = "clip_vitl14"):
        self.is_vit_square = True
        self.input_size = (224, 224)
        if not HAS_OPENCLIP:
            raise RuntimeError("open_clip not available")
        model_name = 'ViT-L-14' if 'vitl' in kind or 'clip_vitl14' in kind else 'ViT-H-14'
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")  # type: ignore
            model.to(self.device)
            model.eval()
            if self.fp16 and self.device.type == "cuda":
                model = model.half()
            self.model = model
            self.clip_tf = preprocess
        except Exception as e:
            raise RuntimeError("Failed to load OpenCLIP model") from e

    def _load_fusion(self):
        # Load both CLIP-L/14 and DINOv2-L/14
        self.is_vit_square = True
        self.input_size = (224, 224)
        self.model = None  # fusion uses self.model_clip and self.model_dino
        # CLIP
        if not HAS_OPENCLIP:
            raise RuntimeError("open_clip not available for fusion")
        try:
            m_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained="openai")  # type: ignore
            m_clip.to(self.device); m_clip.eval()
            if self.fp16 and self.device.type == "cuda":
                m_clip = m_clip.half()
            self.model_clip = m_clip
            self.clip_tf = preprocess
        except Exception as e:
            raise RuntimeError("Failed to load CLIP-L/14 for fusion") from e
        # DINOv2-L
        try:
            import timm
            m_dino = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
            m_dino.to(self.device); m_dino.eval()
            if self.fp16 and self.device.type == "cuda":
                m_dino = m_dino.half()
            self.model_dino = m_dino
        except Exception as e:
            raise RuntimeError("Failed to load DINOv2-L/14 for fusion") from e

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
            with ReIDExtractor._pca_lock:
                buf = ReIDExtractor._pca_buf
                for i in range(min(V.shape[0], 64)):
                    buf.append(V[i].astype(np.float32))
                if (not ReIDExtractor._pca_ready and
                        len(buf) >= max(256, out_dim + 8)):
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
                    ReIDExtractor._pca_mean = mu.astype(np.float32)
                    ReIDExtractor._pca_comp = comp.astype(np.float32)
                    ReIDExtractor._pca_ready = True
                ready = ReIDExtractor._pca_ready
                mu = ReIDExtractor._pca_mean if ready else None
                comp = ReIDExtractor._pca_comp if ready else None
            if ready and mu is not None and comp is not None:
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

    def _forward_heavy_batch(self, crops: List[Optional[np.ndarray]], batch_size: int) -> List[Optional[np.ndarray]]:
        dev = self.device
        bs = max(1, int(batch_size))
        outs: List[Optional[np.ndarray]] = [None] * len(crops)
        if self.backend in ("clip_vitl14", "clip_vith14"):
            groups: List[Optional[tuple[int, int]]] = []
            tens: List[torch.Tensor] = []
            for crop in crops:
                if crop is None:
                    groups.append(None)
                    continue
                tta: List[torch.Tensor] = []
                t = self._preprocess_clip(crop)
                if t is not None:
                    tta.append(t)
                if self.tta_mode in (1, 3):
                    t2 = self._preprocess_clip(cv2.flip(crop, 1))
                    if t2 is not None:
                        tta.append(t2)
                if self.tta_mode == 3:
                    pad = cv2.copyMakeBorders(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    t3 = self._preprocess_clip(pad)
                    if t3 is not None:
                        tta.append(t3)
                if not tta:
                    groups.append(None)
                    continue
                groups.append((len(tens), len(tta)))
                tens.extend(tta)
            vecs = None
            if tens:
                with torch.no_grad():
                    cur: List[torch.Tensor] = []
                    vec_list: List[torch.Tensor] = []
                    for ten in tens:
                        cur.append(ten.unsqueeze(0))
                        if len(cur) == bs:
                            batch = torch.cat(cur, dim=0)
                            if dev.type == "cuda":
                                try:
                                    batch = batch.pin_memory()
                                except Exception:
                                    pass
                                batch = batch.to(dev, non_blocking=True)
                            else:
                                batch = batch.to(dev)
                            if self.fp16 and dev.type == "cuda":
                                batch = batch.half()
                            if hasattr(self.model, "encode_image"):
                                emb = self.model.encode_image(batch)
                            else:
                                emb = self.model(batch)
                            emb = emb.float()
                            vec_list.append(emb.detach().cpu())
                            cur = []
                    if cur:
                        batch = torch.cat(cur, dim=0)
                        if dev.type == "cuda":
                            try:
                                batch = batch.pin_memory()
                            except Exception:
                                pass
                            batch = batch.to(dev, non_blocking=True)
                        else:
                            batch = batch.to(dev)
                        if self.fp16 and dev.type == "cuda":
                            batch = batch.half()
                        if hasattr(self.model, "encode_image"):
                            emb = self.model.encode_image(batch)
                        else:
                            emb = self.model(batch)
                        emb = emb.float()
                        vec_list.append(emb.detach().cpu())
                if vec_list:
                    vecs = torch.cat(vec_list, dim=0).numpy()
            if vecs is not None:
                for i, g in enumerate(groups):
                    if g is None:
                        continue
                    s, c = g
                    v = vecs[s:s + c].mean(axis=0)
                    if self.pca_dim > 0:
                        v = self._pca_reduce_cached(v, self.pca_dim)
                    n = float(np.linalg.norm(v))
                    if n > 1e-12:
                        v = v / n
                    outs[i] = v.astype(np.float32)
        elif self.backend == "dinov2_vitl14":
            groups = []
            tens = []
            for crop in crops:
                if crop is None:
                    groups.append(None)
                    continue
                tta = [self._preprocess(crop)]
                if self.tta_mode in (1, 3):
                    tta.append(self._preprocess(cv2.flip(crop, 1)))
                if self.tta_mode == 3:
                    pad = cv2.copyMakeBorders(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    tta.append(self._preprocess(pad))
                groups.append((len(tens), len(tta)))
                tens.extend(tta)
            vecs = None
            if tens:
                with torch.no_grad():
                    cur = []
                    vec_list = []
                    for ten in tens:
                        cur.append(ten.unsqueeze(0))
                        if len(cur) == bs:
                            batch = torch.cat(cur, dim=0)
                            if dev.type == "cuda":
                                try:
                                    batch = batch.pin_memory()
                                except Exception:
                                    pass
                                batch = batch.to(dev, non_blocking=True)
                            else:
                                batch = batch.to(dev)
                            if self.fp16 and dev.type == "cuda":
                                batch = batch.half()
                            emb = self.model(batch)  # type: ignore[operator]
                            if isinstance(emb, (list, tuple)):
                                emb = emb[0]
                            emb = emb.float()
                            vec_list.append(emb.detach().cpu())
                            cur = []
                    if cur:
                        batch = torch.cat(cur, dim=0)
                        if dev.type == "cuda":
                            try:
                                batch = batch.pin_memory()
                            except Exception:
                                pass
                            batch = batch.to(dev, non_blocking=True)
                        else:
                            batch = batch.to(dev)
                        if self.fp16 and dev.type == "cuda":
                            batch = batch.half()
                        emb = self.model(batch)  # type: ignore[operator]
                        if isinstance(emb, (list, tuple)):
                            emb = emb[0]
                        emb = emb.float()
                        vec_list.append(emb.detach().cpu())
                if vec_list:
                    vecs = torch.cat(vec_list, dim=0).numpy()
            if vecs is not None:
                for i, g in enumerate(groups):
                    if g is None:
                        continue
                    s, c = g
                    v = vecs[s:s + c].mean(axis=0)
                    if self.pca_dim > 0:
                        v = self._pca_reduce_cached(v, self.pca_dim)
                    n = float(np.linalg.norm(v))
                    if n > 1e-12:
                        v = v / n
                    outs[i] = v.astype(np.float32)
        else:  # fusion
            groups_c: List[Optional[tuple[int, int]]] = []
            tens_c: List[torch.Tensor] = []
            groups_d: List[Optional[tuple[int, int]]] = []
            tens_d: List[torch.Tensor] = []
            for crop in crops:
                if crop is None:
                    groups_c.append(None)
                    groups_d.append(None)
                    continue
                tta_c: List[torch.Tensor] = []
                t = self._preprocess_clip(crop)
                if t is not None:
                    tta_c.append(t)
                if self.tta_mode in (1, 3):
                    t2 = self._preprocess_clip(cv2.flip(crop, 1))
                    if t2 is not None:
                        tta_c.append(t2)
                if self.tta_mode == 3:
                    pad = cv2.copyMakeBorders(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    t3 = self._preprocess_clip(pad)
                    if t3 is not None:
                        tta_c.append(t3)
                if tta_c:
                    groups_c.append((len(tens_c), len(tta_c)))
                    tens_c.extend(tta_c)
                else:
                    groups_c.append(None)
                tta_d = [self._preprocess(crop)]
                if self.tta_mode in (1, 3):
                    tta_d.append(self._preprocess(cv2.flip(crop, 1)))
                if self.tta_mode == 3:
                    pad = cv2.copyMakeBorders(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    tta_d.append(self._preprocess(pad))
                groups_d.append((len(tens_d), len(tta_d)))
                tens_d.extend(tta_d)
            vecs_c = None
            if tens_c:
                with torch.no_grad():
                    cur = []
                    vec_list = []
                    for ten in tens_c:
                        cur.append(ten.unsqueeze(0))
                        if len(cur) == bs:
                            batch = torch.cat(cur, dim=0)
                            if dev.type == "cuda":
                                try:
                                    batch = batch.pin_memory()
                                except Exception:
                                    pass
                                batch = batch.to(dev, non_blocking=True)
                            else:
                                batch = batch.to(dev)
                            if self.fp16 and dev.type == "cuda":
                                batch = batch.half()
                            mc = self.model_clip
                            if hasattr(mc, "encode_image"):
                                emb = mc.encode_image(batch)
                            else:
                                emb = mc(batch)
                            emb = emb.float()
                            vec_list.append(emb.detach().cpu())
                            cur = []
                    if cur:
                        batch = torch.cat(cur, dim=0)
                        if dev.type == "cuda":
                            try:
                                batch = batch.pin_memory()
                            except Exception:
                                pass
                            batch = batch.to(dev, non_blocking=True)
                        else:
                            batch = batch.to(dev)
                        if self.fp16 and dev.type == "cuda":
                            batch = batch.half()
                        mc = self.model_clip
                        if hasattr(mc, "encode_image"):
                            emb = mc.encode_image(batch)
                        else:
                            emb = mc(batch)
                        emb = emb.float()
                        vec_list.append(emb.detach().cpu())
                if vec_list:
                    vecs_c = torch.cat(vec_list, dim=0).numpy()
            vecs_d = None
            if tens_d:
                with torch.no_grad():
                    cur = []
                    vec_list = []
                    for ten in tens_d:
                        cur.append(ten.unsqueeze(0))
                        if len(cur) == bs:
                            batch = torch.cat(cur, dim=0)
                            if dev.type == "cuda":
                                try:
                                    batch = batch.pin_memory()
                                except Exception:
                                    pass
                                batch = batch.to(dev, non_blocking=True)
                            else:
                                batch = batch.to(dev)
                            if self.fp16 and dev.type == "cuda":
                                batch = batch.half()
                            md = self.model_dino
                            emb = md(batch)
                            if isinstance(emb, (list, tuple)):
                                emb = emb[0]
                            emb = emb.float()
                            vec_list.append(emb.detach().cpu())
                            cur = []
                    if cur:
                        batch = torch.cat(cur, dim=0)
                        if dev.type == "cuda":
                            try:
                                batch = batch.pin_memory()
                            except Exception:
                                pass
                            batch = batch.to(dev, non_blocking=True)
                        else:
                            batch = batch.to(dev)
                        if self.fp16 and dev.type == "cuda":
                            batch = batch.half()
                        md = self.model_dino
                        emb = md(batch)
                        if isinstance(emb, (list, tuple)):
                            emb = emb[0]
                        emb = emb.float()
                        vec_list.append(emb.detach().cpu())
                if vec_list:
                    vecs_d = torch.cat(vec_list, dim=0).numpy()
            if vecs_c is not None and vecs_d is not None:
                for i in range(len(crops)):
                    g_c = groups_c[i]
                    g_d = groups_d[i]
                    if g_c is None or g_d is None:
                        continue
                    s_c, c_c = g_c
                    s_d, c_d = g_d
                    vclip = vecs_c[s_c:s_c + c_c].mean(axis=0)
                    vdino = vecs_d[s_d:s_d + c_d].mean(axis=0)
                    v = np.concatenate([vclip, vdino], axis=0).astype(np.float32)
                    if self.pca_dim > 0:
                        v = self._pca_reduce_cached(v, self.pca_dim)
                    n = float(np.linalg.norm(v))
                    if n > 1e-12:
                        v = v / n
                    outs[i] = v.astype(np.float32)
        return outs

    def forward(self, list_of_crops_bgr: List[Optional[np.ndarray]], batch_size: int = 32) -> np.ndarray:
        # For heavy backends, run per-crop with TTA and PCA. Keep batching for legacy paths.
        if self.model is None and self.backend != "fusion" and self.model_clip is None:
            return np.zeros((len(list_of_crops_bgr), 0), dtype=np.float32)
        if len(list_of_crops_bgr) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        heavy = self.backend in ("clip_vitl14","clip_vith14","fusion","dinov2_vitl14")
        outs: List[Optional[np.ndarray]]
        if heavy:
            outs = self._forward_heavy_batch(list_of_crops_bgr, batch_size)
        else:
            outs = [None] * len(list_of_crops_bgr)
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
