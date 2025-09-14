"""ReID backbone registry and loaders."""
from typing import Callable, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from reid_extractor import ReIDExtractor

BACKBONE_LOADERS: Dict[str, Callable[["ReIDExtractor"], None]] = {}

def register(name: str) -> Callable[[Callable[["ReIDExtractor"], None]], Callable[["ReIDExtractor"], None]]:
    """Decorator to register a backbone loader."""
    def deco(fn: Callable[["ReIDExtractor"], None]) -> Callable[["ReIDExtractor"], None]:
        BACKBONE_LOADERS[name] = fn
        return fn
    return deco

# Import side-effect modules to populate registry
from . import osnet  # noqa: F401
from . import fastreid  # noqa: F401
from . import dinov2  # noqa: F401
from . import open_clip  # noqa: F401
from . import fusion  # noqa: F401

__all__ = ["BACKBONE_LOADERS", "register"]
