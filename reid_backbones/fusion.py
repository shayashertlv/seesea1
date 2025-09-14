"""Fusion of OpenCLIP ViT-L/14 and DINOv2-L/14 backbones."""
from typing import TYPE_CHECKING

from . import register

if TYPE_CHECKING:  # pragma: no cover
    from reid_extractor import ReIDExtractor

@register("fusion")
def load(extractor: "ReIDExtractor") -> None:
    """Load fusion of CLIP-L/14 and DINOv2-L/14 models."""
    extractor.is_vit_square = True
    extractor.input_size = (224, 224)
    extractor.model = None
    try:
        import open_clip  # type: ignore
    except Exception as e:  # pragma: no cover - dependency missing
        raise RuntimeError("open_clip not available for fusion") from e
    m_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained="openai")  # type: ignore
    m_clip.to(extractor.device); m_clip.eval()
    if extractor.fp16 and extractor.device.type == "cuda":
        m_clip = m_clip.half()
    extractor.model_clip = m_clip
    extractor.clip_tf = preprocess
    try:
        import timm
        m_dino = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        m_dino.to(extractor.device); m_dino.eval()
        if extractor.fp16 and extractor.device.type == "cuda":
            m_dino = m_dino.half()
        extractor.model_dino = m_dino
    except Exception as e:  # pragma: no cover - dependency missing
        raise RuntimeError("Failed to load DINOv2-L/14 for fusion") from e
