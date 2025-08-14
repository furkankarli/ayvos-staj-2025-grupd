"""
General configuration for the realtime behavior analysis system.
Contains model paths, thresholds, and other settings.
"""

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
    POSE_MODEL_PATH = MODELS_DIR / "yolo11n-pose.pt"  # Updated for pose estimation

    # Detection settings
    DETECTION_CONFIDENCE = 0.5
    TRACKING_MAX_DISAPPEARED = 60

    # Pose estimation settings
    POSE_CONFIDENCE = 0.3
    POSE_IOU_THRESHOLD = 0.7
    KEYPOINT_CONFIDENCE = 0.5
    POSE_INPUT_SIZE = 640  # Input size for pose model
    POSE_PADDING = 20  # Padding around bbox for pose estimation

    # Classification thresholds - Improved values
    STANDING_ANGLE_THRESHOLD = 150.0  # More lenient for standing (was 160)
    SITTING_ANGLE_THRESHOLD = 130.0  # More lenient for sitting (was 120)
    FALL_HEIGHT_RATIO = 0.2  # More strict for falling (was 0.3)
    RUNNING_MOVEMENT_THRESHOLD = 15.0  # Lower threshold for running (was 20)

    # Advanced classification parameters - Improved values
    FALL_ORIENTATION_THRESHOLD = 65.0  # More strict for fall detection (was 45)
    FALL_EMERGENCY_THRESHOLD = 85.0  # More strict emergency threshold (was 80)
    RUNNING_FOOT_HEIGHT_THRESHOLD = 15.0  # Lower threshold for running gait (was 20)
    RUNNING_OPTIMAL_LEAN = 20.0  # Optimal body lean for running (was 25)
    ARM_EXTENSION_THRESHOLD = 35.0  # Lower arm extension threshold (was 40)

    # Confidence scoring weights
    POSE_BASE_CONFIDENCE_WEIGHT = 0.4  # Weight for keypoint visibility in confidence
    POSE_SPECIFIC_CONFIDENCE_WEIGHT = 0.6  # Weight for pose-specific criteria

    # Fall detection system
    FALL_HISTORY_SIZE = 30  # Number of detections to keep in history
    SUSTAINED_FALL_THRESHOLD = 3  # Minimum detections for sustained fall
    FALL_RECOVERY_THRESHOLD = 3  # Non-fall detections needed for recovery

    # Movement analysis
    MOVEMENT_HISTORY_TIMEOUT = 2.0  # Seconds to keep movement history
    MIN_TIME_DIFF = 0.01  # Minimum time difference for speed calculation

    # Performance optimization
    ENABLE_POSE_ESTIMATION = True  # Enable/disable pose estimation
    POSE_FRAME_SKIP = 1  # Process every N frames (1 = no skip)
    MAX_POSE_DETECTIONS = 10  # Maximum poses to process per frame
    POSE_ROI_ONLY = True  # Process only ROI around detected person

    # Visualization settings
    KEYPOINT_RADIUS = 3
    SKELETON_THICKNESS = 2
    POSE_COLORS = {
        "standing": (0, 255, 0),  # Green
        "sitting": (255, 255, 0),  # Yellow
        "falling": (0, 0, 255),  # Red
        "running": (255, 0, 255),  # Magenta
        "unknown": (128, 128, 128),  # Gray
    }
    DANGER_COLOR = (0, 0, 255)  # Red for dangerous poses
    ALERT_COLOR = (0, 0, 255)  # Red for alerts

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
        self._performance_stats = {
            "frame_count": 0,
            "total_processing_time": 0.0,
            "avg_fps": 0.0,
        }

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.MODELS_DIR,
            self.DATA_DIR / "inputs",
            self.DATA_DIR / "outputs",
            self.DATA_DIR / "logs",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def update_performance_stats(self, processing_time: float):
        """
        Update performance statistics for adaptive optimization.

        Args:
            processing_time: Time taken to process current frame (seconds)
        """
        self._performance_stats["frame_count"] += 1
        self._performance_stats["total_processing_time"] += processing_time

        if self._performance_stats["frame_count"] > 0:
            avg_time = (
                self._performance_stats["total_processing_time"]
                / self._performance_stats["frame_count"]
            )
            self._performance_stats["avg_fps"] = 1.0 / avg_time if avg_time > 0 else 0.0

    def get_performance_stats(self) -> dict:
        """
        Get current performance statistics.

        Returns:
            Dictionary containing performance metrics
        """
        return self._performance_stats.copy()

    def adapt_settings_for_performance(self):
        """
        Automatically adapt settings based on performance.
        """
        avg_fps = self._performance_stats["avg_fps"]

        # If FPS is too low, reduce quality for better performance
        if avg_fps < 15 and avg_fps > 0:
            print(
                f"Low FPS detected ({avg_fps:.1f}), "
                f"adapting settings for performance..."
            )

            # Increase frame skip
            if self.POSE_FRAME_SKIP < 3:
                self.POSE_FRAME_SKIP += 1
                print(f"Increased frame skip to {self.POSE_FRAME_SKIP}")

            # Reduce input size
            if self.POSE_INPUT_SIZE > 320:
                self.POSE_INPUT_SIZE = max(320, self.POSE_INPUT_SIZE - 160)
                print(f"Reduced pose input size to {self.POSE_INPUT_SIZE}")

            # Reduce max detections
            if self.MAX_POSE_DETECTIONS > 5:
                self.MAX_POSE_DETECTIONS = max(5, self.MAX_POSE_DETECTIONS - 2)
                print(f"Reduced max pose detections to {self.MAX_POSE_DETECTIONS}")

        # If FPS is good, we can increase quality
        elif avg_fps > 25:
            # Decrease frame skip
            if self.POSE_FRAME_SKIP > 1:
                self.POSE_FRAME_SKIP = max(1, self.POSE_FRAME_SKIP - 1)
                print(f"Decreased frame skip to {self.POSE_FRAME_SKIP}")

    def get_model_path_options(self) -> list:
        """
        Get list of available pose model paths in order of preference.

        Returns:
            List of model paths to try
        """
        return [
            self.POSE_MODEL_PATH,
            self.MODELS_DIR / "yolo11n-pose.pt",
            self.MODELS_DIR / "yolov8n-pose.pt",
            self.MODELS_DIR / "yolo11s-pose.pt",  # Slightly larger model
            self.MODELS_DIR / "yolov8s-pose.pt",
            "yolo11n-pose.pt",  # Download from ultralytics
            "yolov8n-pose.pt",  # Fallback
        ]

    def get_optimized_settings(self, target_fps: float = 30.0) -> dict:
        """
        Get optimized settings for target FPS.

        Args:
            target_fps: Target frames per second

        Returns:
            Dictionary of optimized settings
        """
        if target_fps >= 30:
            # High performance settings
            return {
                "pose_input_size": 640,
                "pose_frame_skip": 1,
                "max_detections": 10,
                "keypoint_confidence": 0.5,
                "pose_confidence": 0.3,
            }
        elif target_fps >= 20:
            # Balanced settings
            return {
                "pose_input_size": 480,
                "pose_frame_skip": 1,
                "max_detections": 8,
                "keypoint_confidence": 0.4,
                "pose_confidence": 0.4,
            }
        else:
            # Performance-focused settings
            return {
                "pose_input_size": 320,
                "pose_frame_skip": 2,
                "max_detections": 5,
                "keypoint_confidence": 0.3,
                "pose_confidence": 0.5,
            }

    def apply_optimized_settings(self, target_fps: float = 30.0):
        """
        Apply optimized settings for target FPS.

        Args:
            target_fps: Target frames per second
        """
        settings = self.get_optimized_settings(target_fps)

        self.POSE_INPUT_SIZE = settings["pose_input_size"]
        self.POSE_FRAME_SKIP = settings["pose_frame_skip"]
        self.MAX_POSE_DETECTIONS = settings["max_detections"]
        self.KEYPOINT_CONFIDENCE = settings["keypoint_confidence"]
        self.POSE_CONFIDENCE = settings["pose_confidence"]

        print(f"Applied optimized settings for {target_fps} FPS target")
        print(f"  - Input size: {self.POSE_INPUT_SIZE}")
        print(f"  - Frame skip: {self.POSE_FRAME_SKIP}")
        print(f"  - Max detections: {self.MAX_POSE_DETECTIONS}")

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._performance_stats = {
            "frame_count": 0,
            "total_processing_time": 0.0,
            "avg_fps": 0.0,
        }
