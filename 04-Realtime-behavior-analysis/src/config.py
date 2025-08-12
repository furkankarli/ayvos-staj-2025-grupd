"""
General configuration for the realtime behavior analysis system.
Contains model paths, thresholds, and other settings.
"""

import os
from pathlib import Path

class Config:
    """Configuration class for the behavior analysis system."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    DATA_DIR = PROJECT_ROOT / "data"
    CONFIGS_DIR = PROJECT_ROOT / "configs"
    
    # Model paths
    YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
    POSE_MODEL_PATH = MODELS_DIR / "pose_model.pt"
    
    # Detection settings
    DETECTION_CONFIDENCE = 0.5
    TRACKING_MAX_DISAPPEARED = 30
    
    # Pose estimation settings
    POSE_CONFIDENCE = 0.3
    
    # Analysis settings
    DANGER_THRESHOLD = 5.0  # seconds
    LOG_INTERVAL = 1.0  # seconds
    
    # Video settings
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 480
    FPS = 30
    
    def __init__(self):
        """Initialize configuration and create necessary directories."""
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.MODELS_DIR,
            self.DATA_DIR / "inputs",
            self.DATA_DIR / "outputs", 
            self.DATA_DIR / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)