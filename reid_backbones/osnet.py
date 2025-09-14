"""OSNet-based person ReID backbone."""
from typing import TYPE_CHECKING

from . import register

if TYPE_CHECKING:  # pragma: no cover
    from reid_extractor import ReIDExtractor

@register("osnet")
def load(extractor: "ReIDExtractor") -> None:
    """Load OSNet or ResNet fallback for ReID."""
    extractor.is_vit_square = False
    extractor.input_size = (256, 128)
    model = None
    try:
        import torchreid  # type: ignore
        model = torchreid.models.build_model('osnet_x1_0', num_classes=1, pretrained=True)  # type: ignore
        if hasattr(model, 'classifier'):
            model.classifier = __import__('torch').nn.Identity()
    except Exception:
        try:
            import torchvision
            osnet = getattr(torchvision.models, 'osnet_x1_0', None)
            if osnet is not None:
                model = osnet(pretrained=True)
                if hasattr(model, 'classifier'):
                    model.classifier = __import__('torch').nn.Identity()
                elif hasattr(model, 'fc'):
                    model.fc = __import__('torch').nn.Identity()
        except Exception:
            model = None
    if model is None:
        try:
            import torchvision
            weights = getattr(torchvision.models, 'ResNet50_Weights', None)
            if weights is not None:
                model = torchvision.models.resnet50(weights=weights.DEFAULT)
            else:
                model = torchvision.models.resnet50(pretrained=True)
            model.fc = __import__('torch').nn.Identity()
            extractor.input_size = (224, 224)
        except Exception:
            model = None
    if model is None:
        raise RuntimeError("Failed to load OSNet/ResNet for ReID")
    model.eval().to(extractor.device)
    extractor.model = model
    if extractor.fp16 and extractor.device.type == "cuda":
        extractor.model = extractor.model.half()
