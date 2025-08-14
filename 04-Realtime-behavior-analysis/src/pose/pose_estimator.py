"""
Pose Estimation Module

Advanced human pose estimation system using YOLO pose models.
Integrates keypoint detection with comprehensive pose classification,
providing real-time human pose analysis with fall detection capabilities.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .pose_classifier import PoseClassifier
from .pose_utils import (
    COCO_KEYPOINTS,
    SKELETON_CONNECTIONS,
    get_body_center,
    is_keypoint_visible,
    validate_keypoints,
)


class PoseEstimator:
    """
    Advanced human pose estimation system with integrated classification.

    This class provides comprehensive pose analysis by combining:
    - YOLO-based keypoint detection for accurate human pose estimation
    - Multi-criteria pose classification (standing, sitting, running, falling)
    - Two-level fall detection system (emergency falls + lying down)
    - Real-time performance optimization and adaptive processing
    - Comprehensive statistics and monitoring capabilities

    Features:
    - Support for multiple YOLO pose models (YOLOv8, YOLO11)
    - ROI-based processing for improved performance
    - Temporal tracking integration for movement analysis
    - Detailed pose visualization with classification results
    - Performance metrics and comprehensive statistics
    """

    def __init__(self, config):
        """
        Initialize the pose estimation system with configuration.

        Args:
            config: Configuration object containing:
                   - Model paths and fallback options
                   - Confidence thresholds for detection and keypoints
                   - Performance optimization settings
                   - Classification parameters
        """
        self.config = config
        self.model = None

        # Load detection and keypoint confidence thresholds
        self.confidence_threshold = getattr(config, "POSE_CONFIDENCE", 0.3)
        self.keypoint_confidence = getattr(config, "KEYPOINT_CONFIDENCE", 0.5)

        # Initialize integrated pose classification system
        self.pose_classifier = PoseClassifier(config)

        # Load and initialize the YOLO pose model
        self._load_model()

    def _load_model(self):
        """
        Load YOLO pose model with fallback options.

        Attempts to load pose models in order of preference, falling back
        to alternative models or downloading from Ultralytics if local
        models are not available.
        """
        try:
            # Define model loading priority (local files first, then downloads)
            model_paths = [
                getattr(self.config, "POSE_MODEL_PATH", None),  # User-specified path
                self.config.MODELS_DIR / "yolo11n-pose.pt",  # Latest YOLO11 nano
                self.config.MODELS_DIR / "yolov8n-pose.pt",  # YOLOv8 nano fallback
                "yolo11n-pose.pt",  # Auto-download YOLO11 from Ultralytics
                "yolov8n-pose.pt",  # Auto-download YOLOv8 fallback
            ]

            for model_path in model_paths:
                if model_path is None:
                    continue

                try:
                    print(f"Trying to load pose model: {model_path}")
                    self.model = YOLO(str(model_path))
                    print(f"Successfully loaded pose model: {model_path}")
                    break
                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")
                    continue

            if self.model is None:
                raise RuntimeError("Could not load any pose model")

        except Exception as e:
            print(f"Error loading pose model: {e}")
            raise

    def estimate_pose(self, frame: np.ndarray, tracks: List[Dict]) -> List[Dict]:
        """
        Perform comprehensive pose estimation and classification for tracked humans.

        This method processes each tracked person through the complete pose
        analysis pipeline:
        1. ROI extraction with padding for better keypoint detection
        2. YOLO pose model inference for keypoint detection
        3. Keypoint validation and coordinate adjustment
        4. Multi-criteria pose classification (standing, sitting, running, falling)
        5. Fall detection with temporal tracking
        6. Comprehensive result compilation with statistics

        Args:
            frame: Input video frame as numpy array (H, W, C)
            tracks: List of tracked person dictionaries containing:
                   - 'id': Person tracking ID
                   - 'bbox': Bounding box as (x, y, w, h)

        Returns:
            List of comprehensive pose result dictionaries containing:
            - track_id: Person tracking ID
            - keypoints: 17 COCO keypoints as (x, y, confidence) tuples
            - bbox: Original bounding box
            - body_center: Calculated body center point
            - visible_keypoints: Count of reliable keypoints
            - pose_class: Classified pose type
            - pose_confidence: Classification confidence score
            - angles: Dictionary of calculated body angles
            - is_dangerous: Boolean indicating dangerous pose
            - fall_alert: Optional fall alert information
            - timestamp: Processing timestamp
        """
        if self.model is None:
            return []

        pose_results = []

        try:
            # Process each tracked person
            for track in tracks:
                track_id = track["id"]
                bbox = track["bbox"]  # (x, y, w, h)

                # Extract ROI for pose estimation
                x, y, w, h = bbox

                # Add padding to bbox
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)

                roi = frame[y1:y2, x1:x2]

                if roi.size == 0:
                    continue

                # Run pose estimation on ROI
                results = self.model(roi, conf=self.confidence_threshold, verbose=False)

                # Process results
                for r in results:
                    if r.keypoints is None:
                        continue

                    for keypoints in r.keypoints.data:
                        # Convert keypoints to list format
                        kp_list = []
                        for i in range(len(COCO_KEYPOINTS)):
                            if i < len(keypoints):
                                kp = keypoints[i]
                                # Adjust coordinates back to original frame
                                x_adj = float(kp[0]) + x1
                                y_adj = float(kp[1]) + y1
                                conf = float(kp[2])
                                kp_list.append((x_adj, y_adj, conf))
                            else:
                                kp_list.append((0.0, 0.0, 0.0))

                        # Validate keypoints
                        if not validate_keypoints(kp_list):
                            continue

                        # Classify pose using PoseClassifier
                        classification_result = self.pose_classifier.classify_pose(
                            kp_list, track_id
                        )

                        # Create comprehensive pose result
                        pose_result = {
                            "track_id": track_id,
                            "keypoints": kp_list,
                            "bbox": bbox,
                            "body_center": get_body_center(kp_list),
                            "visible_keypoints": sum(
                                1
                                for kp in kp_list
                                if is_keypoint_visible(kp, self.keypoint_confidence)
                            ),
                            # Add classification results
                            "pose_class": classification_result["pose_class"],
                            "pose_confidence": classification_result["pose_confidence"],
                            "angles": classification_result["angles"],
                            "is_dangerous": classification_result["is_dangerous"],
                            "timestamp": classification_result["timestamp"],
                        }

                        # Add fall alert info if available
                        if "fall_alert" in classification_result:
                            pose_result["fall_alert"] = classification_result[
                                "fall_alert"
                            ]

                        pose_results.append(pose_result)
                        break  # Only take first detection per track

        except Exception as e:
            print(f"Error in pose estimation: {e}")

        return pose_results

    def draw_keypoints(
        self,
        frame: np.ndarray,
        pose_results: List[Dict],
        color: Tuple[int, int, int] = None,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw keypoints, skeleton, and pose classification on the frame.

        Args:
            frame: Input frame
            pose_results: List of pose results
            color: BGR color tuple (if None, uses pose-specific colors)
            thickness: Line thickness

        Returns:
            Frame with drawn keypoints and pose information
        """
        frame_copy = frame.copy()

        # Pose-specific colors
        pose_colors = {
            "standing": (0, 255, 0),  # Green
            "sitting": (255, 255, 0),  # Yellow
            "falling": (0, 0, 255),  # Red
            "running": (255, 0, 255),  # Magenta
            "unknown": (128, 128, 128),  # Gray
        }

        try:
            for pose_result in pose_results:
                keypoints = pose_result["keypoints"]
                track_id = pose_result["track_id"]
                pose_class = pose_result.get("pose_class", "unknown")
                pose_confidence = pose_result.get("pose_confidence", 0.0)
                is_dangerous = pose_result.get("is_dangerous", False)

                # Choose color based on pose class or use provided color
                if color is None:
                    draw_color = pose_colors.get(pose_class, (128, 128, 128))
                    # Use red for dangerous poses
                    if is_dangerous:
                        draw_color = (0, 0, 255)
                else:
                    draw_color = color

                # Draw keypoints
                for i, (x, y, conf) in enumerate(keypoints):
                    if is_keypoint_visible((x, y, conf), self.keypoint_confidence):
                        cv2.circle(frame_copy, (int(x), int(y)), 3, draw_color, -1)

                # Draw skeleton connections
                for connection in SKELETON_CONNECTIONS:
                    idx1, idx2 = connection
                    if idx1 < len(keypoints) and idx2 < len(keypoints):
                        kp1 = keypoints[idx1]
                        kp2 = keypoints[idx2]

                        if is_keypoint_visible(
                            kp1, self.keypoint_confidence
                        ) and is_keypoint_visible(kp2, self.keypoint_confidence):

                            pt1 = (int(kp1[0]), int(kp1[1]))
                            pt2 = (int(kp2[0]), int(kp2[1]))
                            cv2.line(frame_copy, pt1, pt2, draw_color, thickness)

                # Draw pose information
                body_center = pose_result.get("body_center", (0, 0))
                if body_center != (0, 0):
                    # Main label with ID and pose
                    main_label = f"ID: {track_id} - {pose_class.upper()}"
                    cv2.putText(
                        frame_copy,
                        main_label,
                        (int(body_center[0] - 50), int(body_center[1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        draw_color,
                        thickness,
                    )

                    # Confidence score
                    conf_label = f"Conf: {pose_confidence:.2f}"
                    cv2.putText(
                        frame_copy,
                        conf_label,
                        (int(body_center[0] - 50), int(body_center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        draw_color,
                        1,
                    )

                    # Danger warning
                    if is_dangerous:
                        danger_label = "⚠️ DANGER!"
                        cv2.putText(
                            frame_copy,
                            danger_label,
                            (int(body_center[0] - 50), int(body_center[1] + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

                    # Fall alert
                    if "fall_alert" in pose_result:
                        alert_label = "🚨 FALL ALERT!"
                        cv2.putText(
                            frame_copy,
                            alert_label,
                            (int(body_center[0] - 60), int(body_center[1] + 40)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )

        except Exception as e:
            print(f"Error drawing keypoints: {e}")

        return frame_copy

    def get_pose_stats(self, pose_results: List[Dict]) -> Dict:
        """
        Get comprehensive statistics about pose estimation and classification results.

        Args:
            pose_results: List of pose results

        Returns:
            Dictionary containing detailed pose statistics
        """
        if not pose_results:
            return {
                "total_poses": 0,
                "avg_visible_keypoints": 0.0,
                "min_visible_keypoints": 0,
                "max_visible_keypoints": 0,
                "pose_distribution": {},
                "dangerous_poses": 0,
                "avg_pose_confidence": 0.0,
                "fall_alerts": 0,
            }

        try:
            visible_counts = [result["visible_keypoints"] for result in pose_results]
            pose_classes = [
                result.get("pose_class", "unknown") for result in pose_results
            ]
            pose_confidences = [
                result.get("pose_confidence", 0.0) for result in pose_results
            ]
            dangerous_count = sum(
                1 for result in pose_results if result.get("is_dangerous", False)
            )
            fall_alerts = sum(1 for result in pose_results if "fall_alert" in result)

            # Count pose distribution
            pose_distribution = {}
            for pose_class in pose_classes:
                pose_distribution[pose_class] = pose_distribution.get(pose_class, 0) + 1

            return {
                "total_poses": len(pose_results),
                "avg_visible_keypoints": float(np.mean(visible_counts)),
                "min_visible_keypoints": int(np.min(visible_counts)),
                "max_visible_keypoints": int(np.max(visible_counts)),
                "pose_distribution": pose_distribution,
                "dangerous_poses": dangerous_count,
                "avg_pose_confidence": (
                    float(np.mean(pose_confidences)) if pose_confidences else 0.0
                ),
                "fall_alerts": fall_alerts,
                "classification_stats": {
                    "standing": pose_distribution.get("standing", 0),
                    "sitting": pose_distribution.get("sitting", 0),
                    "falling": pose_distribution.get("falling", 0),
                    "running": pose_distribution.get("running", 0),
                    "unknown": pose_distribution.get("unknown", 0),
                },
            }

        except Exception as e:
            print(f"Error calculating pose stats: {e}")
            return {
                "total_poses": len(pose_results),
                "avg_visible_keypoints": 0.0,
                "min_visible_keypoints": 0,
                "max_visible_keypoints": 0,
                "pose_distribution": {},
                "dangerous_poses": 0,
                "avg_pose_confidence": 0.0,
                "fall_alerts": 0,
            }

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"model_loaded": False}

        try:
            return {
                "model_loaded": True,
                "model_path": (
                    str(self.model.model_name)
                    if hasattr(self.model, "model_name")
                    else "Unknown"
                ),
                "confidence_threshold": self.confidence_threshold,
                "keypoint_confidence": self.keypoint_confidence,
                "num_keypoints": len(COCO_KEYPOINTS),
            }
        except Exception as e:
            return {"model_loaded": True, "error": str(e)}

    def get_fall_statistics(self) -> Dict:
        """
        Get fall detection statistics from pose classifier.

        Returns:
            Dictionary containing fall statistics
        """
        return self.pose_classifier.get_fall_statistics()

    def get_movement_statistics(self) -> Dict:
        """
        Get movement statistics from pose classifier.

        Returns:
            Dictionary containing movement statistics
        """
        return self.pose_classifier.get_movement_statistics()

    def clear_fall_alert(self, track_id: int):
        """
        Clear fall alert for a specific person.

        Args:
            track_id: Person's track ID
        """
        self.pose_classifier.clear_fall_alert(track_id)

    def get_comprehensive_stats(self, pose_results: List[Dict]) -> Dict:
        """
        Get comprehensive statistics combining pose estimation and classification data.

        Args:
            pose_results: List of pose results

        Returns:
            Dictionary containing all statistics
        """
        pose_stats = self.get_pose_stats(pose_results)
        fall_stats = self.get_fall_statistics()
        movement_stats = self.get_movement_statistics()

        return {
            "pose_estimation": pose_stats,
            "fall_detection": fall_stats,
            "movement_analysis": movement_stats,
            "summary": {
                "total_people": pose_stats["total_poses"],
                "dangerous_situations": pose_stats["dangerous_poses"],
                "active_alerts": fall_stats["active_fall_alerts"],
                "avg_confidence": pose_stats["avg_pose_confidence"],
            },
        }
