"""
Pose Estimation Utility Functions

Essential geometric calculations and keypoint processing utilities for human
pose analysis. Provides core mathematical functions for angle calculations,
distance measurements, body orientation analysis, and keypoint validation
used throughout the pose estimation system.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

# COCO 17 keypoints format
COCO_KEYPOINTS = [
    "nose",  # 0
    "left_eye",  # 1
    "right_eye",  # 2
    "left_ear",  # 3
    "right_ear",  # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11
    "right_hip",  # 12
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle",  # 16
]

# Keypoint indices for easy access
KEYPOINT_INDICES = {name: idx for idx, name in enumerate(COCO_KEYPOINTS)}

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    # Head
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    # Body
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    # Left arm
    (5, 7),
    (7, 9),
    # Right arm
    (6, 8),
    (8, 10),
    # Left leg
    (11, 13),
    (13, 15),
    # Right leg
    (12, 14),
    (14, 16),
]


def calculate_angle(
    p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]
) -> float:
    """
    Calculate the angle between three points using vector mathematics.

    This function computes the angle at the vertex point p2 formed by the vectors
    p2->p1 and p2->p3. Essential for analyzing joint angles in human pose estimation.

    Args:
        p1: First point coordinates (x, y)
        p2: Vertex point coordinates (x, y) - the angle is measured at this point
        p3: Third point coordinates (x, y)

    Returns:
        Angle in degrees (0-180). Returns 0.0 if calculation fails due to
        invalid points or zero-length vectors.
    """
    try:
        # Create vectors from vertex point to the other two points
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        # Calculate vector magnitudes for normalization
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # Handle degenerate cases (zero-length vectors)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate angle using dot product formula: cos(θ) = (v1·v2)/(|v1||v2|)
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure numerical stability

        # Convert from radians to degrees
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    except (ValueError, ZeroDivisionError):
        return 0.0


def get_body_center(keypoints: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """
    Calculate the center point of the human body using torso keypoints.

    Computes the geometric center based on visible shoulder and hip keypoints,
    providing a stable reference point for body position analysis and tracking.

    Args:
        keypoints: List of 17 COCO keypoints as (x, y, confidence) tuples

    Returns:
        Body center coordinates (x, y). Returns (0.0, 0.0) if insufficient
        visible keypoints are available for calculation.
    """
    try:
        # Extract core torso keypoints (shoulders and hips)
        left_shoulder = keypoints[KEYPOINT_INDICES["left_shoulder"]]
        right_shoulder = keypoints[KEYPOINT_INDICES["right_shoulder"]]
        left_hip = keypoints[KEYPOINT_INDICES["left_hip"]]
        right_hip = keypoints[KEYPOINT_INDICES["right_hip"]]

        # Collect all visible torso keypoints for center calculation
        valid_points = []
        for point in [left_shoulder, right_shoulder, left_hip, right_hip]:
            if is_keypoint_visible(point):
                valid_points.append((point[0], point[1]))

        # Return default if no valid torso keypoints found
        if not valid_points:
            return (0.0, 0.0)

        # Calculate geometric center as average of valid points
        center_x = sum(p[0] for p in valid_points) / len(valid_points)
        center_y = sum(p[1] for p in valid_points) / len(valid_points)

        return (float(center_x), float(center_y))

    except (IndexError, ValueError):
        return (0.0, 0.0)


def calculate_body_ratio(keypoints: List[Tuple[float, float, float]]) -> float:
    """
    Calculate the height-to-width ratio of the human body.

    This ratio is crucial for pose classification, particularly for detecting
    horizontal positions (falling/lying) vs vertical positions (standing).
    A low ratio indicates a horizontal body position.

    Args:
        keypoints: List of 17 COCO keypoints as (x, y, confidence) tuples

    Returns:
        Height/width ratio. Higher values indicate more vertical postures,
        lower values indicate more horizontal postures. Returns 0.0 if
        calculation fails due to missing keypoints.
    """
    try:
        # Extract key reference points for height and width calculation
        nose = keypoints[KEYPOINT_INDICES["nose"]]
        left_ankle = keypoints[KEYPOINT_INDICES["left_ankle"]]
        right_ankle = keypoints[KEYPOINT_INDICES["right_ankle"]]
        left_shoulder = keypoints[KEYPOINT_INDICES["left_shoulder"]]
        right_shoulder = keypoints[KEYPOINT_INDICES["right_shoulder"]]

        # Calculate body height (head to feet distance)
        height = 0.0
        if is_keypoint_visible(nose):
            # Use maximum distance from nose to either ankle
            if is_keypoint_visible(left_ankle):
                height = max(height, abs(nose[1] - left_ankle[1]))
            if is_keypoint_visible(right_ankle):
                height = max(height, abs(nose[1] - right_ankle[1]))

        # Calculate body width (shoulder span)
        width = 0.0
        if is_keypoint_visible(left_shoulder) and is_keypoint_visible(right_shoulder):
            width = abs(left_shoulder[0] - right_shoulder[0])

        # Avoid division by zero
        if width == 0:
            return 0.0

        return height / width

    except (IndexError, ValueError, ZeroDivisionError):
        return 0.0


def is_keypoint_visible(
    keypoint: Tuple[float, float, float], min_confidence: float = 0.3
) -> bool:
    """
    Check if a keypoint is visible and reliable for pose analysis.

    Validates keypoint visibility based on confidence score and coordinate validity.
    Essential for filtering out unreliable keypoint detections that could
    negatively impact pose classification accuracy.

    Args:
        keypoint: Keypoint data as (x, y, confidence) tuple
        min_confidence: Minimum confidence threshold for visibility (default: 0.3)

    Returns:
        True if keypoint is visible and reliable, False otherwise.
        Considers both confidence score and coordinate validity.
    """
    try:
        x, y, conf = keypoint
        return conf >= min_confidence and x > 0 and y > 0
    except (ValueError, IndexError):
        return False


def normalize_keypoints(
    keypoints: List[Tuple[float, float, float]], bbox: Tuple[int, int, int, int]
) -> List[Tuple[float, float, float]]:
    """
    Keypoint'leri bbox'a göre normalize eder.

    Args:
        keypoints: Keypoint listesi
        bbox: Bounding box (x, y, w, h)

    Returns:
        Normalize edilmiş keypoint listesi
    """
    try:
        x, y, w, h = bbox

        if w == 0 or h == 0:
            return keypoints

        normalized = []
        for kp in keypoints:
            kp_x, kp_y, conf = kp

            # Bbox'a göre normalize et (0-1 arası)
            norm_x = (kp_x - x) / w if w > 0 else 0.0
            norm_y = (kp_y - y) / h if h > 0 else 0.0

            normalized.append((norm_x, norm_y, conf))

        return normalized

    except (ValueError, ZeroDivisionError):
        return keypoints


def get_keypoint_by_name(
    keypoints: List[Tuple[float, float, float]], name: str
) -> Optional[Tuple[float, float, float]]:
    """
    İsme göre keypoint döndürür.

    Args:
        keypoints: Keypoint listesi
        name: Keypoint ismi

    Returns:
        Keypoint (x, y, confidence) veya None
    """
    try:
        if name not in KEYPOINT_INDICES:
            return None

        idx = KEYPOINT_INDICES[name]
        if idx >= len(keypoints):
            return None

        return keypoints[idx]

    except (IndexError, ValueError):
        return None


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    İki nokta arasındaki Euclidean mesafeyi hesaplar.

    Args:
        p1: İlk nokta (x, y)
        p2: İkinci nokta (x, y)

    Returns:
        Mesafe
    """
    try:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx * dx + dy * dy)
    except (ValueError, TypeError):
        return 0.0


def get_body_orientation(keypoints: List[Tuple[float, float, float]]) -> float:
    """
    Calculate body orientation angle from vertical axis.

    Measures the deviation of the torso (shoulder-hip line) from vertical position.
    This is a critical measurement for fall detection and pose classification.

    Args:
        keypoints: List of 17 COCO keypoints as (x, y, confidence) tuples

    Returns:
        Orientation angle in degrees:
        - 0° = perfectly upright/vertical
        - 90° = completely horizontal
        - Values between indicate the degree of tilt from vertical
        Returns 0.0 if calculation fails due to missing keypoints.
    """
    try:
        # Extract shoulder and hip keypoints for torso line calculation
        left_shoulder = get_keypoint_by_name(keypoints, "left_shoulder")
        right_shoulder = get_keypoint_by_name(keypoints, "right_shoulder")
        left_hip = get_keypoint_by_name(keypoints, "left_hip")
        right_hip = get_keypoint_by_name(keypoints, "right_hip")

        # Calculate shoulder center point
        shoulder_center = None
        if (
            left_shoulder
            and right_shoulder
            and is_keypoint_visible(left_shoulder)
            and is_keypoint_visible(right_shoulder)
        ):
            shoulder_center = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2,
            )

        # Calculate hip center point
        hip_center = None
        if (
            left_hip
            and right_hip
            and is_keypoint_visible(left_hip)
            and is_keypoint_visible(right_hip)
        ):
            hip_center = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2,
            )

        # Return default if insufficient keypoints for calculation
        if not shoulder_center or not hip_center:
            return 0.0

        # Calculate angle between torso line and vertical reference
        body_vector = (
            hip_center[0] - shoulder_center[0],
            hip_center[1] - shoulder_center[1],
        )

        # Calculate angle using dot product with vertical vector
        dot_product = body_vector[1]  # Dot product with (0,1) is just the y-component
        magnitude = math.sqrt(body_vector[0] ** 2 + body_vector[1] ** 2)

        # Handle degenerate case
        if magnitude == 0:
            return 0.0

        # Calculate angle from vertical (0° = upright, 90° = horizontal)
        cos_angle = dot_product / magnitude
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Ensure valid range for acos

        angle_rad = math.acos(abs(cos_angle))
        angle_deg = math.degrees(angle_rad)

        return float(angle_deg)

    except (ValueError, ZeroDivisionError):
        return 0.0


def validate_keypoints(keypoints: List[Tuple[float, float, float]]) -> bool:
    """
    Validate keypoint data quality for reliable pose classification.

    Performs comprehensive validation to ensure keypoint data is suitable
    for pose analysis. Checks both structural validity (correct count) and
    content quality (sufficient visible keypoints).

    Args:
        keypoints: List of keypoints as (x, y, confidence) tuples

    Returns:
        True if keypoints are valid for pose analysis, False otherwise.
        Requires correct keypoint count and minimum visible critical keypoints.
    """
    try:
        # Verify correct number of keypoints (must match COCO 17-keypoint format)
        if len(keypoints) != len(COCO_KEYPOINTS):
            return False

        # Check visibility of critical keypoints for pose analysis
        critical_keypoints = [
            "nose",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        ]
        visible_count = 0

        for name in critical_keypoints:
            kp = get_keypoint_by_name(keypoints, name)
            if kp and is_keypoint_visible(kp):
                visible_count += 1

        # Require at least 3 critical keypoints to be visible for reliable pose analysis
        return visible_count >= 3

    except (ValueError, TypeError):
        return False
