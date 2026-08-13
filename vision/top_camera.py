"""Independent Top Camera (Camera 2) stream module.

Camera 2 is used strictly for Top View analysis:
- Hungry fish detection
- Fish counting during feeding
- Food dispensing logic

Never use Camera 2 for stress analysis.
"""

from vision.side_camera import SideCamera
from config import TOP_CAMERA_INDEX


class TopCamera(SideCamera):
    """Manages Camera 2 top-view video capture stream independently of Camera 1."""

    def __init__(self, index: int = TOP_CAMERA_INDEX):
        super().__init__(index=index)
