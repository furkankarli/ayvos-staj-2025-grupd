# Detection module for YOLO human detection and ID tracking
from .detector import HumanDetector
from .tracker import HumanTracker

__all__ = ["HumanDetector", "HumanTracker"]
