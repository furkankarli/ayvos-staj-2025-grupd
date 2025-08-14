"""
Pose Classification Module

Advanced human pose classification algorithms using keypoint analysis.
Implements multi-criteria pose detection for standing, sitting, running,
and falling poses with two-level fall detection system for both
emergency falls and lying down situations.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .pose_utils import (
    calculate_angle,
    calculate_body_ratio,
    calculate_distance,
    get_body_center,
    get_body_orientation,
    get_keypoint_by_name,
    is_keypoint_visible,
    validate_keypoints,
)


class PoseClassifier:
    """
    Advanced human pose classification system using keypoint analysis.

    This classifier implements a multi-criteria scoring system to detect:
    - Standing poses (upright posture with straight legs)
    - Sitting poses (bent hip-knee angles, compact posture)
    - Running poses (gait analysis, foot height differences, body lean)
    - Falling poses (two-level detection: emergency falls and lying down)

    Features:
    - Weighted scoring algorithms for each pose type
    - Movement analysis for running detection
    - Two-level fall detection (emergency + lying down)
    - Temporal tracking for sustained fall alerts
    - Confidence scoring based on keypoint visibility and pose-specific criteria
    """

    def __init__(self, config):
        """
        Initialize the pose classifier with configuration settings.

        Args:
            config: Configuration object containing classification thresholds,
                   model parameters, and performance settings
        """
        self.config = config

        # Load classification thresholds from config
        self.standing_angle_threshold = getattr(
            config, "STANDING_ANGLE_THRESHOLD", 160.0
        )
        self.sitting_angle_threshold = getattr(config, "SITTING_ANGLE_THRESHOLD", 120.0)
        self.fall_height_ratio = getattr(config, "FALL_HEIGHT_RATIO", 0.3)
        self.running_movement_threshold = getattr(
            config, "RUNNING_MOVEMENT_THRESHOLD", 20.0
        )
        self.keypoint_confidence = getattr(config, "KEYPOINT_CONFIDENCE", 0.5)

        # Movement tracking data for temporal analysis
        self.previous_positions = (
            {}
        )  # track_id -> (x, y, timestamp) for speed calculation

        # Fall detection system with temporal tracking
        self.fall_history = {}  # track_id -> [fall_detections_with_timestamps]
        self.fall_alerts = {}  # track_id -> alert_info for sustained fall alerts

    def classify_pose(
        self, keypoints: List[Tuple[float, float, float]], track_id: int = None
    ) -> Dict:
        """
        Classify human pose from keypoint data using multi-criteria analysis.

        This method performs comprehensive pose analysis including:
        - Keypoint validation and preprocessing
        - Body angle calculations (legs, hips, torso)
        - Multi-criteria pose classification with weighted scoring
        - Confidence calculation based on keypoint visibility and pose-specific criteria
        - Temporal fall detection with sustained alert management

        Args:
            keypoints: List of 17 COCO keypoints as (x, y, confidence) tuples
            track_id: Optional person tracking ID for movement analysis and fall history

        Returns:
            Dictionary containing:
            - pose_class: Detected pose ('standing', 'sitting', 'running',
              'falling', 'unknown')
            - pose_confidence: Classification confidence score (0.0-1.0)
            - angles: Dictionary of calculated body angles
            - is_dangerous: Boolean indicating if pose represents danger
            - timestamp: Classification timestamp
            - fall_alert: Optional fall alert information if applicable
        """
        try:
            # Step 1: Validate keypoint data quality
            if not validate_keypoints(keypoints):
                return self._create_result("unknown", 0.0, {}, False)

            # Step 2: Calculate essential body angles for pose analysis
            angles = self._calculate_angles(keypoints)

            # Step 3: Perform multi-criteria pose classification
            pose_class = self._analyze_body_position(keypoints, angles, track_id)

            # Step 4: Calculate classification confidence score
            confidence = self._calculate_confidence(keypoints, angles, pose_class)

            # Step 5: Assess danger level (falling/lying detection)
            is_dangerous = self._detect_fall(keypoints, angles)

            # Step 6: Update temporal fall tracking system
            if track_id is not None:
                self.update_fall_history(track_id, is_dangerous, confidence)

                # Check for fall recovery (person no longer in dangerous state)
                if not is_dangerous and track_id in self.fall_alerts:
                    # Count recent non-fall detections for recovery confirmation
                    recent_non_falls = 0
                    if track_id in self.fall_history:
                        recent_detections = self.fall_history[track_id][-5:]
                        recent_non_falls = sum(
                            1 for d in recent_detections if not d["is_falling"]
                        )

                    # Clear alert after 3 consecutive non-fall detections
                    if recent_non_falls >= 3:
                        self.clear_fall_alert(track_id)

            result = self._create_result(pose_class, confidence, angles, is_dangerous)

            # Add fall alert info if available
            if track_id is not None:
                fall_alert = self.get_fall_alert_info(track_id)
                if fall_alert:
                    result["fall_alert"] = fall_alert

            return result

        except Exception as e:
            print(f"Error in pose classification: {e}")
            return self._create_result("unknown", 0.0, {}, False)

    def _calculate_angles(
        self, keypoints: List[Tuple[float, float, float]]
    ) -> Dict[str, float]:
        """
        Calculate essential body angles for pose classification.

        Computes key anatomical angles used in pose analysis:
        - Leg angles (hip-knee-ankle) for standing/sitting detection
        - Hip-knee angles for sitting posture analysis
        - Knee-ankle angles for detailed leg position assessment
        - Body orientation angle for fall/lying detection

        Args:
            keypoints: List of 17 COCO keypoints as (x, y, confidence) tuples

        Returns:
            Dictionary containing calculated angles:
            - left_leg, right_leg: Hip-knee-ankle angles (degrees)
            - left_hip_knee, right_hip_knee: Shoulder-hip-knee angles (degrees)
            - left_knee_ankle, right_knee_ankle: Hip-knee-ankle angles (degrees)
            - body_orientation: Body tilt from vertical (degrees,
              0=upright, 90=horizontal)
        """
        angles = {}

        try:
            # Calculate left leg angle (hip-knee-ankle)
            left_hip = get_keypoint_by_name(keypoints, "left_hip")
            left_knee = get_keypoint_by_name(keypoints, "left_knee")
            left_ankle = get_keypoint_by_name(keypoints, "left_ankle")

            if (
                left_hip
                and left_knee
                and left_ankle
                and is_keypoint_visible(left_hip, self.keypoint_confidence)
                and is_keypoint_visible(left_knee, self.keypoint_confidence)
                and is_keypoint_visible(left_ankle, self.keypoint_confidence)
            ):

                angles["left_leg"] = calculate_angle(
                    (left_hip[0], left_hip[1]),
                    (left_knee[0], left_knee[1]),
                    (left_ankle[0], left_ankle[1]),
                )

            # Calculate right leg angle (hip-knee-ankle)
            right_hip = get_keypoint_by_name(keypoints, "right_hip")
            right_knee = get_keypoint_by_name(keypoints, "right_knee")
            right_ankle = get_keypoint_by_name(keypoints, "right_ankle")

            if (
                right_hip
                and right_knee
                and right_ankle
                and is_keypoint_visible(right_hip, self.keypoint_confidence)
                and is_keypoint_visible(right_knee, self.keypoint_confidence)
                and is_keypoint_visible(right_ankle, self.keypoint_confidence)
            ):

                angles["right_leg"] = calculate_angle(
                    (right_hip[0], right_hip[1]),
                    (right_knee[0], right_knee[1]),
                    (right_ankle[0], right_ankle[1]),
                )

            # Calculate hip-knee angles (critical for sitting detection)
            left_shoulder = get_keypoint_by_name(keypoints, "left_shoulder")
            if (
                left_shoulder
                and left_hip
                and left_knee
                and is_keypoint_visible(left_shoulder, self.keypoint_confidence)
                and is_keypoint_visible(left_hip, self.keypoint_confidence)
                and is_keypoint_visible(left_knee, self.keypoint_confidence)
            ):

                angles["left_hip_knee"] = calculate_angle(
                    (left_shoulder[0], left_shoulder[1]),
                    (left_hip[0], left_hip[1]),
                    (left_knee[0], left_knee[1]),
                )

            right_shoulder = get_keypoint_by_name(keypoints, "right_shoulder")
            if (
                right_shoulder
                and right_hip
                and right_knee
                and is_keypoint_visible(right_shoulder, self.keypoint_confidence)
                and is_keypoint_visible(right_hip, self.keypoint_confidence)
                and is_keypoint_visible(right_knee, self.keypoint_confidence)
            ):

                angles["right_hip_knee"] = calculate_angle(
                    (right_shoulder[0], right_shoulder[1]),
                    (right_hip[0], right_hip[1]),
                    (right_knee[0], right_knee[1]),
                )

            # Calculate overall body orientation (crucial for fall detection)
            angles["body_orientation"] = get_body_orientation(keypoints)

            # Calculate knee-ankle angles (additional sitting detection criteria)
            if (
                left_knee
                and left_ankle
                and left_hip
                and is_keypoint_visible(left_knee, self.keypoint_confidence)
                and is_keypoint_visible(left_ankle, self.keypoint_confidence)
                and is_keypoint_visible(left_hip, self.keypoint_confidence)
            ):

                angles["left_knee_ankle"] = calculate_angle(
                    (left_hip[0], left_hip[1]),
                    (left_knee[0], left_knee[1]),
                    (left_ankle[0], left_ankle[1]),
                )

            if (
                right_knee
                and right_ankle
                and right_hip
                and is_keypoint_visible(right_knee, self.keypoint_confidence)
                and is_keypoint_visible(right_ankle, self.keypoint_confidence)
                and is_keypoint_visible(right_hip, self.keypoint_confidence)
            ):

                angles["right_knee_ankle"] = calculate_angle(
                    (right_hip[0], right_hip[1]),
                    (right_knee[0], right_knee[1]),
                    (right_ankle[0], right_ankle[1]),
                )

        except Exception as e:
            print(f"Error calculating angles: {e}")

        return angles

    def _analyze_body_position(
        self,
        keypoints: List[Tuple[float, float, float]],
        angles: Dict[str, float],
        track_id: int = None,
    ) -> str:
        """
        Analyze body position from keypoints and angles with improved priority order.

        Args:
            keypoints: List of keypoints
            angles: Dictionary of calculated angles
            track_id: Optional track ID for movement analysis

        Returns:
            Pose class string
        """
        try:
            # Check for running first (higher priority than standing/sitting)
            if self._detect_running(keypoints, angles, track_id):
                return "running"

            # Check for standing
            if self._detect_standing(keypoints, angles):
                return "standing"

            # Check for sitting
            if self._detect_sitting(keypoints, angles):
                return "sitting"

            # Check for falling last (lowest priority, most strict)
            if self._detect_fall(keypoints, angles):
                return "falling"

            # Default to unknown
            return "unknown"

        except Exception as e:
            print(f"Error analyzing body position: {e}")
            return "unknown"

    def _detect_standing(
        self, keypoints: List[Tuple[float, float, float]], angles: Dict[str, float]
    ) -> bool:
        """
        Detect standing pose with improved algorithm.

        Args:
            keypoints: List of keypoints
            angles: Dictionary of angles

        Returns:
            True if standing detected
        """
        try:
            standing_score = 0.0
            max_score = 0.0

            # Criterion 1: Leg straightness (35% weight)
            leg_angles = []
            if "left_leg" in angles:
                leg_angles.append(angles["left_leg"])
            if "right_leg" in angles:
                leg_angles.append(angles["right_leg"])

            if leg_angles:
                max_score += 0.35
                # Average leg angle should be close to 180 degrees (straight)
                avg_leg_angle = sum(leg_angles) / len(leg_angles)
                # More lenient threshold for standing
                if avg_leg_angle > 150:  # Reduced from standing_angle_threshold (160)
                    standing_score += 0.35 * min((avg_leg_angle - 150) / 30, 1.0)

            # Criterion 2: Body uprightness (30% weight)
            max_score += 0.3
            body_orientation = angles.get("body_orientation", 90)
            if body_orientation < 25:  # Less than 25 degrees from vertical (was 30)
                standing_score += 0.3 * (1 - body_orientation / 25)

            # Criterion 3: Body center above feet (25% weight)
            max_score += 0.25
            body_center = get_body_center(keypoints)
            left_ankle = get_keypoint_by_name(keypoints, "left_ankle")
            right_ankle = get_keypoint_by_name(keypoints, "right_ankle")

            feet_y_coords = []
            if left_ankle and is_keypoint_visible(left_ankle, self.keypoint_confidence):
                feet_y_coords.append(left_ankle[1])
            if right_ankle and is_keypoint_visible(
                right_ankle, self.keypoint_confidence
            ):
                feet_y_coords.append(right_ankle[1])

            if feet_y_coords:
                avg_feet_y = sum(feet_y_coords) / len(feet_y_coords)
                if body_center[1] < avg_feet_y:  # Body center above feet
                    standing_score += 0.25

            # Criterion 4: Hip-knee alignment (10% weight)
            max_score += 0.1
            left_hip = get_keypoint_by_name(keypoints, "left_hip")
            right_hip = get_keypoint_by_name(keypoints, "right_hip")
            left_knee = get_keypoint_by_name(keypoints, "left_knee")
            right_knee = get_keypoint_by_name(keypoints, "right_knee")

            hip_knee_aligned = False
            if (
                left_hip
                and left_knee
                and is_keypoint_visible(left_hip, self.keypoint_confidence)
                and is_keypoint_visible(left_knee, self.keypoint_confidence)
            ):
                # Hip should be roughly above knee for standing
                hip_knee_distance = abs(left_hip[0] - left_knee[0])
                if hip_knee_distance < 40:  # More lenient alignment (was 30)
                    hip_knee_aligned = True

            if (
                right_hip
                and right_knee
                and is_keypoint_visible(right_hip, self.keypoint_confidence)
                and is_keypoint_visible(right_knee, self.keypoint_confidence)
            ):
                hip_knee_distance = abs(right_hip[0] - right_knee[0])
                if hip_knee_distance < 40:
                    hip_knee_aligned = True

            if hip_knee_aligned:
                standing_score += 0.1

            # Standing detected if score > 55% of maximum possible (reduced from 60%)
            return max_score > 0 and (standing_score / max_score) > 0.55

        except Exception as e:
            print(f"Error detecting standing: {e}")
            return False

    def _detect_sitting(
        self, keypoints: List[Tuple[float, float, float]], angles: Dict[str, float]
    ) -> bool:
        """
        Detect sitting pose with improved algorithm.

        Args:
            keypoints: List of keypoints
            angles: Dictionary of angles

        Returns:
            True if sitting detected
        """
        try:
            sitting_score = 0.0
            max_score = 0.0

            # Criterion 1: Hip-knee angles (bent for sitting) (35% weight)
            hip_knee_angles = []
            if "left_hip_knee" in angles:
                hip_knee_angles.append(angles["left_hip_knee"])
            if "right_hip_knee" in angles:
                hip_knee_angles.append(angles["right_hip_knee"])

            if hip_knee_angles:
                max_score += 0.35
                bent_hip_knee_count = sum(
                    1
                    for angle in hip_knee_angles
                    if angle < self.sitting_angle_threshold
                )
                if bent_hip_knee_count > 0:
                    # Score based on how bent the angles are
                    avg_bent_angle = (
                        sum(
                            angle
                            for angle in hip_knee_angles
                            if angle < self.sitting_angle_threshold
                        )
                        / bent_hip_knee_count
                    )
                    sitting_score += 0.35 * (
                        1 - avg_bent_angle / self.sitting_angle_threshold
                    )

            # Criterion 2: Knee-ankle angles (bent for sitting) (25% weight)
            knee_ankle_angles = []
            if "left_knee_ankle" in angles:
                knee_ankle_angles.append(angles["left_knee_ankle"])
            if "right_knee_ankle" in angles:
                knee_ankle_angles.append(angles["right_knee_ankle"])

            if knee_ankle_angles:
                max_score += 0.25
                bent_knee_ankle_count = sum(
                    1
                    for angle in knee_ankle_angles
                    if angle < self.sitting_angle_threshold
                )
                if bent_knee_ankle_count > 0:
                    avg_bent_angle = (
                        sum(
                            angle
                            for angle in knee_ankle_angles
                            if angle < self.sitting_angle_threshold
                        )
                        / bent_knee_ankle_count
                    )
                    sitting_score += 0.25 * (
                        1 - avg_bent_angle / self.sitting_angle_threshold
                    )

            # Criterion 3: Hip position relative to knees (25% weight)
            max_score += 0.25
            left_hip = get_keypoint_by_name(keypoints, "left_hip")
            right_hip = get_keypoint_by_name(keypoints, "right_hip")
            left_knee = get_keypoint_by_name(keypoints, "left_knee")
            right_knee = get_keypoint_by_name(keypoints, "right_knee")

            hip_knee_score = 0.0
            hip_knee_count = 0

            # Check left side
            if (
                left_hip
                and left_knee
                and is_keypoint_visible(left_hip, self.keypoint_confidence)
                and is_keypoint_visible(left_knee, self.keypoint_confidence)
            ):
                hip_knee_count += 1
                # For sitting, hip should be above and slightly back from knee
                if left_hip[1] < left_knee[1]:  # Hip above knee
                    hip_knee_score += 0.5
                    # Check horizontal alignment (hip slightly back is OK for sitting)
                    horizontal_diff = abs(left_hip[0] - left_knee[0])
                    if horizontal_diff < 50:  # Reasonable alignment
                        hip_knee_score += 0.5

            # Check right side
            if (
                right_hip
                and right_knee
                and is_keypoint_visible(right_hip, self.keypoint_confidence)
                and is_keypoint_visible(right_knee, self.keypoint_confidence)
            ):
                hip_knee_count += 1
                if right_hip[1] < right_knee[1]:
                    hip_knee_score += 0.5
                    horizontal_diff = abs(right_hip[0] - right_knee[0])
                    if horizontal_diff < 50:
                        hip_knee_score += 0.5

            if hip_knee_count > 0:
                sitting_score += 0.25 * (hip_knee_score / hip_knee_count)

            # Criterion 4: Body compactness (15% weight)
            max_score += 0.15
            body_ratio = calculate_body_ratio(keypoints)
            if 0.5 < body_ratio < 2.0:  # Sitting should have moderate body ratio
                sitting_score += 0.15 * (
                    1 - abs(body_ratio - 1.0)
                )  # Closer to 1.0 is better

            # Sitting detected if score > 65% of maximum possible
            return max_score > 0 and (sitting_score / max_score) > 0.65

        except Exception as e:
            print(f"Error detecting sitting: {e}")
            return False

    def _detect_fall(
        self, keypoints: List[Tuple[float, float, float]], angles: Dict[str, float]
    ) -> bool:
        """
        Detect falling pose with two-level detection: emergency falls and lying down.

        Args:
            keypoints: List of keypoints
            angles: Dictionary of angles

        Returns:
            True if falling or lying down detected
        """
        try:
            # First check for emergency fall (strict criteria)
            emergency_fall = self._detect_emergency_fall(keypoints, angles)
            if emergency_fall:
                return True

            # Then check for lying down (more lenient criteria)
            lying_down = self._detect_lying_down(keypoints, angles)
            return lying_down

        except Exception as e:
            print(f"Error detecting fall: {e}")
            return False

    def _detect_emergency_fall(
        self, keypoints: List[Tuple[float, float, float]], angles: Dict[str, float]
    ) -> bool:
        """
        Detect emergency falling situations (active falling motion).

        Args:
            keypoints: List of keypoints
            angles: Dictionary of angles

        Returns:
            True if emergency fall detected
        """
        try:
            fall_score = 0.0
            max_score = 0.0

            # Criterion 1: Very horizontal body orientation (40% weight)
            max_score += 0.4
            body_orientation = angles.get("body_orientation", 0)
            if body_orientation > 75:  # Very horizontal
                fall_score += 0.4 * min((body_orientation - 75) / 15, 1.0)

            # Criterion 2: Very low body ratio (30% weight)
            max_score += 0.3
            body_ratio = calculate_body_ratio(keypoints)
            if body_ratio < 0.5:  # Very flat
                fall_score += 0.3 * (1 - body_ratio / 0.5)

            # Criterion 3: Limb positioning (defensive/falling position) (30% weight)
            max_score += 0.3
            limb_fall_indicators = 0
            limb_checks = 0

            # Check arms in defensive position
            left_wrist = get_keypoint_by_name(keypoints, "left_wrist")
            right_wrist = get_keypoint_by_name(keypoints, "right_wrist")
            left_shoulder = get_keypoint_by_name(keypoints, "left_shoulder")
            right_shoulder = get_keypoint_by_name(keypoints, "right_shoulder")

            if (
                left_wrist
                and left_shoulder
                and is_keypoint_visible(left_wrist, self.keypoint_confidence)
                and is_keypoint_visible(left_shoulder, self.keypoint_confidence)
            ):
                limb_checks += 1
                arm_distance = calculate_distance(
                    (left_wrist[0], left_wrist[1]), (left_shoulder[0], left_shoulder[1])
                )
                if arm_distance > 60:  # Extended arm (defensive)
                    limb_fall_indicators += 1

            if (
                right_wrist
                and right_shoulder
                and is_keypoint_visible(right_wrist, self.keypoint_confidence)
                and is_keypoint_visible(right_shoulder, self.keypoint_confidence)
            ):
                limb_checks += 1
                arm_distance = calculate_distance(
                    (right_wrist[0], right_wrist[1]),
                    (right_shoulder[0], right_shoulder[1]),
                )
                if arm_distance > 60:
                    limb_fall_indicators += 1

            if limb_checks > 0:
                fall_score += 0.3 * (limb_fall_indicators / limb_checks)

            # Emergency fall detected if score > 80% of maximum possible
            return max_score > 0 and (fall_score / max_score) > 0.8

        except Exception as e:
            print(f"Error detecting emergency fall: {e}")
            return False

    def _detect_lying_down(
        self, keypoints: List[Tuple[float, float, float]], angles: Dict[str, float]
    ) -> bool:
        """
        Detect lying down situations (person on ground, already fallen).

        Args:
            keypoints: List of keypoints
            angles: Dictionary of angles

        Returns:
            True if lying down detected
        """
        try:
            lying_score = 0.0
            max_score = 0.0

            # Criterion 1: Horizontal body orientation (40% weight)
            max_score += 0.4
            body_orientation = angles.get("body_orientation", 0)
            if body_orientation > 45:  # Moderately horizontal
                lying_score += 0.4 * min((body_orientation - 45) / 45, 1.0)

            # Criterion 2: Low body ratio (30% weight)
            max_score += 0.3
            body_ratio = calculate_body_ratio(keypoints)
            if body_ratio < 1.2:  # Relatively flat
                lying_score += 0.3 * (1 - body_ratio / 1.2)

            # Criterion 3: Head and torso at similar height (20% weight)
            max_score += 0.2
            nose = get_keypoint_by_name(keypoints, "nose")
            left_shoulder = get_keypoint_by_name(keypoints, "left_shoulder")
            right_shoulder = get_keypoint_by_name(keypoints, "right_shoulder")

            if nose and is_keypoint_visible(nose, self.keypoint_confidence):
                shoulder_y_coords = []
                if left_shoulder and is_keypoint_visible(
                    left_shoulder, self.keypoint_confidence
                ):
                    shoulder_y_coords.append(left_shoulder[1])
                if right_shoulder and is_keypoint_visible(
                    right_shoulder, self.keypoint_confidence
                ):
                    shoulder_y_coords.append(right_shoulder[1])

                if shoulder_y_coords:
                    avg_shoulder_y = sum(shoulder_y_coords) / len(shoulder_y_coords)
                    # Head and shoulders at similar height (lying down)
                    height_diff = abs(nose[1] - avg_shoulder_y)
                    if height_diff < 50:  # Head and shoulders aligned horizontally
                        lying_score += 0.2

            # Criterion 4: Feet position (10% weight)
            max_score += 0.1
            left_ankle = get_keypoint_by_name(keypoints, "left_ankle")
            right_ankle = get_keypoint_by_name(keypoints, "right_ankle")

            if (
                left_ankle
                and right_ankle
                and is_keypoint_visible(left_ankle, self.keypoint_confidence)
                and is_keypoint_visible(right_ankle, self.keypoint_confidence)
            ):

                # Both feet at similar height (lying down)
                feet_height_diff = abs(left_ankle[1] - right_ankle[1])
                if feet_height_diff < 40:  # Feet aligned (lying flat)
                    lying_score += 0.1

            # Lying down detected if score > 60% of maximum possible
            return max_score > 0 and (lying_score / max_score) > 0.6

        except Exception as e:
            print(f"Error detecting lying down: {e}")
            return False

        except Exception as e:
            print(f"Error detecting fall: {e}")
            return False

    def _detect_running(
        self,
        keypoints: List[Tuple[float, float, float]],
        angles: Dict[str, float],
        track_id: int = None,
    ) -> bool:
        """
        Detect running pose with improved gait and movement analysis.

        Args:
            keypoints: List of keypoints
            angles: Dictionary of angles
            track_id: Track ID for movement analysis

        Returns:
            True if running detected
        """
        try:
            running_score = 0.0
            max_score = 0.0

            # Criterion 1: Foot height difference (running gait) (40% weight)
            max_score += 0.4
            left_ankle = get_keypoint_by_name(keypoints, "left_ankle")
            right_ankle = get_keypoint_by_name(keypoints, "right_ankle")

            if (
                left_ankle
                and right_ankle
                and is_keypoint_visible(left_ankle, self.keypoint_confidence)
                and is_keypoint_visible(right_ankle, self.keypoint_confidence)
            ):

                height_diff = abs(left_ankle[1] - right_ankle[1])
                # Lower threshold for foot height difference
                if height_diff > 15:  # Reduced from 20
                    # Score increases with height difference (up to 80 pixels)
                    running_score += 0.4 * min(height_diff / 80, 1.0)

            # Criterion 2: Leg angles (bent knees typical in running) (25% weight)
            max_score += 0.25
            leg_angles = []
            if "left_leg" in angles:
                leg_angles.append(angles["left_leg"])
            if "right_leg" in angles:
                leg_angles.append(angles["right_leg"])

            if leg_angles:
                # Running typically has bent knees (angles < 160 degrees)
                bent_leg_count = sum(1 for angle in leg_angles if angle < 160)
                if bent_leg_count > 0:
                    # Score based on how bent the legs are
                    avg_bent_angle = (
                        sum(angle for angle in leg_angles if angle < 160)
                        / bent_leg_count
                    )
                    # Optimal running knee angle is around 120-140 degrees
                    if 100 < avg_bent_angle < 160:
                        running_score += 0.25 * (1 - abs(avg_bent_angle - 130) / 30)

            # Criterion 3: Body lean (forward lean typical in running) (20% weight)
            max_score += 0.2
            body_orientation = angles.get("body_orientation", 0)
            # Adjusted range for running lean
            if 5 < body_orientation < 35:  # Moderate forward lean (was 10-50)
                # Optimal lean is around 15-25 degrees
                optimal_lean = 20
                lean_score = 1 - abs(body_orientation - optimal_lean) / 15
                running_score += 0.2 * max(lean_score, 0)

            # Criterion 4: Movement speed (15% weight)
            max_score += 0.15
            if track_id is not None:
                import time

                body_center = get_body_center(keypoints)
                current_time = time.time()

                if track_id in self.previous_positions:
                    prev_x, prev_y, prev_time = self.previous_positions[track_id]

                    # Calculate movement speed
                    time_diff = current_time - prev_time
                    if time_diff > 0.01:  # Avoid division by very small numbers
                        distance = calculate_distance((prev_x, prev_y), body_center)
                        speed = distance / time_diff

                        # Score based on speed (running typically 20-100 pixels/second)
                        if speed > 15:  # Reduced threshold
                            running_score += 0.15 * min((speed - 15) / 85, 1.0)

                # Update position history
                self.previous_positions[track_id] = (
                    body_center[0],
                    body_center[1],
                    current_time,
                )

            # Additional running indicators (bonus scoring)
            running_bonus = 0.0

            # Very high foot difference (clear running gait)
            if (
                left_ankle
                and right_ankle
                and is_keypoint_visible(left_ankle, self.keypoint_confidence)
                and is_keypoint_visible(right_ankle, self.keypoint_confidence)
            ):
                height_diff = abs(left_ankle[1] - right_ankle[1])
                if height_diff > 40:  # Clear running gait (reduced from 60)
                    running_bonus += 0.15

            # High speed movement
            if track_id is not None and track_id in self.previous_positions:
                import time

                body_center = get_body_center(keypoints)
                prev_x, prev_y, prev_time = self.previous_positions[track_id]
                current_time = time.time()
                time_diff = current_time - prev_time

                if time_diff > 0.01:
                    distance = calculate_distance((prev_x, prev_y), body_center)
                    speed = distance / time_diff
                    if speed > 40:  # Fast movement (reduced from 50)
                        running_bonus += 0.1

            # Both legs bent (typical running stance)
            if len(leg_angles) >= 2:
                bent_legs = sum(1 for angle in leg_angles if angle < 150)
                if bent_legs >= 2:
                    running_bonus += 0.1

            running_score += running_bonus

            # Running detected if score > 50% of maximum possible (reduced from 60%)
            return max_score > 0 and (running_score / max_score) > 0.5

        except Exception as e:
            print(f"Error detecting running: {e}")
            return False

    def _calculate_confidence(
        self,
        keypoints: List[Tuple[float, float, float]],
        angles: Dict[str, float],
        pose_class: str,
    ) -> float:
        """
        Calculate confidence score for pose classification with improved algorithm.

        Args:
            keypoints: List of keypoints
            angles: Dictionary of angles
            pose_class: Classified pose

        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            # Base confidence from visible keypoints (40% weight)
            visible_count = sum(
                1
                for kp in keypoints
                if is_keypoint_visible(kp, self.keypoint_confidence)
            )
            base_confidence = min(visible_count / len(keypoints), 1.0)

            # Pose-specific confidence (60% weight)
            pose_confidence = 0.5  # Default

            if pose_class == "standing":
                # Standing confidence based on multiple criteria
                standing_factors = []

                # Leg straightness
                leg_angles = [angles.get("left_leg", 0), angles.get("right_leg", 0)]
                if leg_angles:
                    avg_leg_angle = sum(leg_angles) / len(leg_angles)
                    if avg_leg_angle > self.standing_angle_threshold:
                        standing_factors.append(
                            (avg_leg_angle - self.standing_angle_threshold)
                            / (180 - self.standing_angle_threshold)
                        )

                # Body uprightness
                body_orientation = angles.get("body_orientation", 90)
                if body_orientation < 45:
                    standing_factors.append(1 - body_orientation / 45)

                pose_confidence = (
                    sum(standing_factors) / len(standing_factors)
                    if standing_factors
                    else 0.5
                )

            elif pose_class == "sitting":
                # Sitting confidence based on bent angles and positioning
                sitting_factors = []

                # Hip-knee bend quality
                hip_knee_angles = [
                    angles.get("left_hip_knee", 180),
                    angles.get("right_hip_knee", 180),
                ]
                bent_hip_knee = [
                    angle
                    for angle in hip_knee_angles
                    if angle < self.sitting_angle_threshold
                ]
                if bent_hip_knee:
                    avg_bend = sum(bent_hip_knee) / len(bent_hip_knee)
                    sitting_factors.append(1 - avg_bend / self.sitting_angle_threshold)

                # Knee-ankle bend quality
                knee_ankle_angles = [
                    angles.get("left_knee_ankle", 180),
                    angles.get("right_knee_ankle", 180),
                ]
                bent_knee_ankle = [
                    angle
                    for angle in knee_ankle_angles
                    if angle < self.sitting_angle_threshold
                ]
                if bent_knee_ankle:
                    avg_bend = sum(bent_knee_ankle) / len(bent_knee_ankle)
                    sitting_factors.append(1 - avg_bend / self.sitting_angle_threshold)

                # Body compactness
                body_ratio = calculate_body_ratio(keypoints)
                if 0.5 < body_ratio < 2.0:
                    sitting_factors.append(1 - abs(body_ratio - 1.0))

                pose_confidence = (
                    sum(sitting_factors) / len(sitting_factors)
                    if sitting_factors
                    else 0.5
                )

            elif pose_class == "falling":
                # Falling confidence based on multiple criteria
                falling_factors = []

                # Body orientation (horizontal)
                body_orientation = angles.get("body_orientation", 0)
                if body_orientation > 45:
                    falling_factors.append(min((body_orientation - 45) / 45, 1.0))

                # Body ratio (low height/width)
                body_ratio = calculate_body_ratio(keypoints)
                if body_ratio < 1.5:
                    falling_factors.append(1 - body_ratio / 1.5)

                # Emergency conditions boost confidence
                if body_orientation > 80 or body_ratio < 0.3:
                    falling_factors.append(1.0)  # Maximum confidence for emergency

                pose_confidence = (
                    sum(falling_factors) / len(falling_factors)
                    if falling_factors
                    else 0.5
                )

            elif pose_class == "running":
                # Running confidence based on multiple movement indicators
                running_factors = []

                # Foot height difference quality
                left_ankle = get_keypoint_by_name(keypoints, "left_ankle")
                right_ankle = get_keypoint_by_name(keypoints, "right_ankle")
                if (
                    left_ankle
                    and right_ankle
                    and is_keypoint_visible(left_ankle, self.keypoint_confidence)
                    and is_keypoint_visible(right_ankle, self.keypoint_confidence)
                ):
                    height_diff = abs(left_ankle[1] - right_ankle[1])
                    if height_diff > 20:
                        running_factors.append(min(height_diff / 100, 1.0))

                # Body lean quality
                body_orientation = angles.get("body_orientation", 0)
                if 10 < body_orientation < 50:
                    optimal_lean = 25
                    lean_score = 1 - abs(body_orientation - optimal_lean) / 25
                    running_factors.append(max(lean_score, 0))

                # Movement speed (if available)
                # Note: This would need track_id context which we don't have here
                # So we use a moderate base confidence

                pose_confidence = (
                    sum(running_factors) / len(running_factors)
                    if running_factors
                    else 0.6
                )

            elif pose_class == "unknown":
                pose_confidence = 0.1  # Low confidence for unknown poses

            # Weighted combination: 40% base + 60% pose-specific
            final_confidence = (0.4 * base_confidence) + (0.6 * pose_confidence)
            return min(max(final_confidence, 0.0), 1.0)

        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5

    def _create_result(
        self,
        pose_class: str,
        confidence: float,
        angles: Dict[str, float],
        is_dangerous: bool,
    ) -> Dict:
        """
        Create pose classification result dictionary.

        Args:
            pose_class: Classified pose
            confidence: Confidence score
            angles: Dictionary of angles
            is_dangerous: Whether pose is dangerous

        Returns:
            Result dictionary
        """
        return {
            "pose_class": pose_class,
            "pose_confidence": confidence,
            "angles": angles,
            "is_dangerous": is_dangerous,
            "timestamp": np.datetime64("now"),
        }

    def update_fall_history(self, track_id: int, is_falling: bool, confidence: float):
        """
        Update fall detection history for a person.

        Args:
            track_id: Person's track ID
            is_falling: Whether person is currently falling
            confidence: Confidence of fall detection
        """
        try:
            current_time = np.datetime64("now")

            if track_id not in self.fall_history:
                self.fall_history[track_id] = []

            # Add current detection to history
            self.fall_history[track_id].append(
                {
                    "is_falling": is_falling,
                    "confidence": confidence,
                    "timestamp": current_time,
                }
            )

            # Keep only last 30 detections (about 1 second at 30fps)
            if len(self.fall_history[track_id]) > 30:
                self.fall_history[track_id] = self.fall_history[track_id][-30:]

            # Check for sustained fall (multiple consecutive detections)
            self._check_sustained_fall(track_id)

        except Exception as e:
            print(f"Error updating fall history: {e}")

    def _check_sustained_fall(self, track_id: int):
        """
        Check if person has sustained fall (multiple consecutive fall detections).

        Args:
            track_id: Person's track ID
        """
        try:
            if track_id not in self.fall_history:
                return

            history = self.fall_history[track_id]
            if len(history) < 3:  # Need at least 3 detections
                return

            # Check last 5 detections for sustained fall
            recent_detections = history[-5:]
            fall_count = sum(
                1 for detection in recent_detections if detection["is_falling"]
            )

            # If 3 out of last 5 detections are falls, trigger sustained fall alert
            if fall_count >= 3:
                avg_confidence = (
                    sum(d["confidence"] for d in recent_detections if d["is_falling"])
                    / fall_count
                )

                # Create or update alert
                self.fall_alerts[track_id] = {
                    "alert_type": "sustained_fall",
                    "confidence": avg_confidence,
                    "start_time": recent_detections[0]["timestamp"],
                    "last_update": recent_detections[-1]["timestamp"],
                    "detection_count": fall_count,
                }

                print(
                    f"🚨 SUSTAINED FALL ALERT - Person ID {track_id} - "
                    f"Confidence: {avg_confidence:.2f}"
                )

        except Exception as e:
            print(f"Error checking sustained fall: {e}")

    def get_fall_alert_info(self, track_id: int) -> Optional[Dict]:
        """
        Get fall alert information for a person.

        Args:
            track_id: Person's track ID

        Returns:
            Alert information dictionary or None
        """
        return self.fall_alerts.get(track_id, None)

    def clear_fall_alert(self, track_id: int):
        """
        Clear fall alert for a person (when they recover).

        Args:
            track_id: Person's track ID
        """
        if track_id in self.fall_alerts:
            del self.fall_alerts[track_id]
            print(f"✅ Fall alert cleared for Person ID {track_id}")

    def get_fall_statistics(self) -> Dict:
        """
        Get overall fall detection statistics.

        Returns:
            Dictionary containing fall statistics
        """
        try:
            total_people = len(self.fall_history)
            active_alerts = len(self.fall_alerts)

            # Count recent fall detections (last 10 seconds)
            current_time = np.datetime64("now")
            recent_falls = 0

            for track_id, history in self.fall_history.items():
                for detection in history:
                    time_diff = (
                        current_time - detection["timestamp"]
                    ) / np.timedelta64(1, "s")
                    if time_diff <= 10 and detection["is_falling"]:
                        recent_falls += 1
                        break  # Count each person only once

            return {
                "total_tracked_people": total_people,
                "active_fall_alerts": active_alerts,
                "recent_fall_detections": recent_falls,
                "alert_details": list(self.fall_alerts.values()),
            }

        except Exception as e:
            print(f"Error getting fall statistics: {e}")
            return {
                "total_tracked_people": 0,
                "active_fall_alerts": 0,
                "recent_fall_detections": 0,
                "alert_details": [],
            }

    def get_movement_speed(
        self, track_id: int, current_position: Tuple[float, float]
    ) -> float:
        """
        Calculate movement speed for a person.

        Args:
            track_id: Person's track ID
            current_position: Current body center position (x, y)

        Returns:
            Movement speed in pixels per second
        """
        try:
            current_time = np.datetime64("now")

            if track_id not in self.previous_positions:
                self.previous_positions[track_id] = (
                    current_position[0],
                    current_position[1],
                    current_time,
                )
                return 0.0

            prev_x, prev_y, prev_time = self.previous_positions[track_id]

            # Calculate time difference
            time_diff = (current_time - prev_time) / np.timedelta64(1, "s")

            if time_diff <= 0.01:  # Too small time difference
                return 0.0

            # Calculate distance moved
            distance = calculate_distance((prev_x, prev_y), current_position)

            # Calculate speed
            speed = distance / time_diff

            # Update position history
            self.previous_positions[track_id] = (
                current_position[0],
                current_position[1],
                current_time,
            )

            return float(speed)

        except Exception as e:
            print(f"Error calculating movement speed: {e}")
            return 0.0

    def get_movement_statistics(self) -> Dict:
        """
        Get movement statistics for all tracked people.

        Returns:
            Dictionary containing movement statistics
        """
        try:
            if not self.previous_positions:
                return {
                    "total_moving_people": 0,
                    "average_speed": 0.0,
                    "max_speed": 0.0,
                    "running_people": 0,
                }

            current_time = np.datetime64("now")
            speeds = []
            running_count = 0

            for track_id, (x, y, timestamp) in self.previous_positions.items():
                # Only consider recent positions (last 2 seconds)
                time_diff = (current_time - timestamp) / np.timedelta64(1, "s")
                if time_diff <= 2.0:
                    # Estimate current speed based on recent movement
                    # This is approximate since we don't have current position here
                    speeds.append(0.0)  # Placeholder - would need current positions

                    # Count as running if recent speed was high
                    # This would be better implemented with actual speed tracking
                    if time_diff <= 0.5:  # Very recent
                        running_count += 1 if time_diff > 0 else 0

            return {
                "total_moving_people": len(speeds),
                "average_speed": np.mean(speeds) if speeds else 0.0,
                "max_speed": np.max(speeds) if speeds else 0.0,
                "running_people": running_count,
            }

        except Exception as e:
            print(f"Error getting movement statistics: {e}")
            return {
                "total_moving_people": 0,
                "average_speed": 0.0,
                "max_speed": 0.0,
                "running_people": 0,
            }
