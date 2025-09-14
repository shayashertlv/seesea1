"""OpenCLIP backbones."""
from typing import TYPE_CHECKING

from . import register

if TYPE_CHECKING:  # pragma: no cover
    from reid_extractor import ReIDExtractor


def _load(extractor: "ReIDExtractor", kind: str) -> None:
    extractor.is_vit_square = True
    extractor.input_size = (224, 224)
    try:
        import open_clip  # type: ignore
    except Exception as e:  # pragma: no cover - dependency missing
        raise RuntimeError("open_clip not available") from e
    model_name = 'ViT-L-14' if 'vitl' in kind or 'clip_vitl14' in kind else 'ViT-H-14'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")  # type: ignore
    model.to(extractor.device)
    model.eval()
    if extractor.fp16 and extractor.device.type == "cuda":
        model = model.half()
    extractor.model = model
    extractor.clip_tf = preprocess

@register("clip_vitl14")
def load_vitl14(extractor: "ReIDExtractor") -> None:
    """Load OpenCLIP ViT-L/14 model."""
    _load(extractor, "clip_vitl14")

@register("clip_vith14")
def load_vith14(extractor: "ReIDExtractor") -> None:
    """Load OpenCLIP ViT-H/14 model."""
    _load(extractor, "clip_vith14")
