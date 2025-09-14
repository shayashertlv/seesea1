"""FastReID ResNet-50 backbone."""
from typing import TYPE_CHECKING

from . import register

if TYPE_CHECKING:  # pragma: no cover
    from reid_extractor import ReIDExtractor

@register("fastreid_r50")
def load(extractor: "ReIDExtractor") -> None:
    """Load FastReID ResNet-50 model."""
    extractor.is_vit_square = False
    extractor.input_size = (256, 128)
    try:
        from fastreid.config import get_cfg  # type: ignore
        from fastreid.modeling import build_model  # type: ignore
        from fastreid.utils.checkpoint import Checkpointer  # type: ignore
    except Exception as e:  # pragma: no cover - dependency missing
        raise RuntimeError("FastReID not installed") from e
    cfg = get_cfg()
    cfg.merge_from_list([
        'MODEL.BACKBONE.NAME', 'build_resnet_backbone',
        'MODEL.HEADS.NUM_CLASSES', '1',
        'MODEL.WEIGHTS', 'pretrained',
    ])
    model = build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval().to(extractor.device)
    if extractor.fp16 and extractor.device.type == "cuda":
        model = model.half()
    extractor.model = model
