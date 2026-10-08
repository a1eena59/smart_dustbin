"""Waste classification core: detection, burst voting, and decision logging."""

# Only stdlib-backed modules are re-exported so `import wastevision` never pulls in torch or cv2.
from wastevision.settings import CATEGORIES, CATEGORY_INDEX, Settings

__all__ = ["CATEGORIES", "CATEGORY_INDEX", "Settings"]
__version__ = "0.1.0"
