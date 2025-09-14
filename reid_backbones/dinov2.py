"""DINOv2 ViT backbones via timm."""
from typing import TYPE_CHECKING

from . import register

if TYPE_CHECKING:  # pragma: no cover
    from reid_extractor import ReIDExtractor

@register("dinov2_vits14")
def load_small(extractor: "ReIDExtractor") -> None:
    """Load DINOv2 ViT-S/14 model."""
    extractor.is_vit_square = True
    extractor.input_size = (224, 224)
    try:
        import timm
        model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        model.eval().to(extractor.device)
        extractor.model = model
        if extractor.fp16 and extractor.device.type == "cuda":
            extractor.model = extractor.model.half()
    except Exception as e:  # pragma: no cover - dependency missing
        raise RuntimeError("Failed to load DINOv2 ViT-S/14 via timm") from e

@register("dinov2_vitl14")
def load_large(extractor: "ReIDExtractor") -> None:
    """Load DINOv2 ViT-L/14 model."""
    extractor.is_vit_square = True
    extractor.input_size = (224, 224)
    last_err = None
    try:
        import timm
        try:
            model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        except Exception as e1:
            last_err = e1
            model = timm.create_model('vit_large_patch14_dinov2', pretrained=True, num_classes=0)
        model.eval().to(extractor.device)
        extractor.model = model
        if extractor.fp16 and extractor.device.type == "cuda":
            extractor.model = extractor.model.half()
    except Exception as e:
        if last_err is None:
            last_err = e
        raise RuntimeError("Failed to load DINOv2 ViT-L/14 via timm") from last_err
