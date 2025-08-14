"""
Advanced Visualization and Overlay System
Video üzerine pose bilgileri, uyarılar ve istatistikleri yazma sistemi.
"""

import math
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np


class PoseOverlay:
    """Advanced pose visualization and overlay system."""

    def __init__(self, config):
        """
        Initialize the pose overlay system.

        Args:
            config: Configuration object
        """
        self.config = config

        # Overlay settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.line_thickness = 2

        # Colors from config
        self.pose_colors = getattr(
            config,
            "POSE_COLORS",
            {
                "standing": (0, 255, 0),
                "sitting": (255, 255, 0),
                "falling": (0, 0, 255),
                "running": (255, 0, 255),
                "unknown": (128, 128, 128),
            },
        )

        self.danger_color = getattr(config, "DANGER_COLOR", (0, 0, 255))
        self.alert_color = getattr(config, "ALERT_COLOR", (0, 0, 255))

        # Animation state for alerts
        self.alert_animation_frame = 0
        self.blink_state = True

    def draw_comprehensive_overlay(
        self, frame: np.ndarray, pose_results: List[Dict], frame_info: Dict = None
    ) -> np.ndarray:
        """
        Draw comprehensive overlay with pose information, alerts, and statistics.

        Args:
            frame: Input frame
            pose_results: List of pose results
            frame_info: Optional frame information (frame_count, fps, etc.)

        Returns:
            Frame with comprehensive overlay
        """
        overlay_frame = frame.copy()

        # Draw individual pose overlays
        for pose_result in pose_results:
            overlay_frame = self.draw_pose_overlay(overlay_frame, pose_result)

        # Draw statistics panel
        overlay_frame = self.draw_statistics_panel(
            overlay_frame, pose_results, frame_info
        )

        # Draw alert panel
        overlay_frame = self.draw_alert_panel(overlay_frame, pose_results)

        # Draw legend
        overlay_frame = self.draw_legend(overlay_frame)

        # Update animation frame
        self.alert_animation_frame += 1
        if self.alert_animation_frame % 30 == 0:  # Blink every 30 frames
            self.blink_state = not self.blink_state

        return overlay_frame

    def draw_pose_overlay(self, frame: np.ndarray, pose_result: Dict) -> np.ndarray:
        """
        Draw overlay for a single pose result.

        Args:
            frame: Input frame
            pose_result: Single pose result

        Returns:
            Frame with pose overlay
        """
        track_id = pose_result["track_id"]
        pose_class = pose_result["pose_class"]
        confidence = pose_result["pose_confidence"]
        is_dangerous = pose_result["is_dangerous"]
        body_center = pose_result.get("body_center", (0, 0))

        if body_center == (0, 0):
            return frame

        # Choose color based on pose and danger status
        if is_dangerous:
            color = self.danger_color
        else:
            color = self.pose_colors.get(pose_class, (128, 128, 128))

        center_x, center_y = int(body_center[0]), int(body_center[1])

        # Draw main pose label with background
        main_label = f"ID:{track_id} {pose_class.upper()}"
        label_size = cv2.getTextSize(
            main_label, self.font, self.font_scale, self.font_thickness
        )[0]

        # Background rectangle for better readability
        bg_x1 = center_x - label_size[0] // 2 - 5
        bg_y1 = center_y - 40
        bg_x2 = center_x + label_size[0] // 2 + 5
        bg_y2 = center_y - 15

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)

        cv2.putText(
            frame,
            main_label,
            (center_x - label_size[0] // 2, center_y - 20),
            self.font,
            self.font_scale,
            color,
            self.font_thickness,
        )

        # Confidence bar
        conf_label = f"Conf: {confidence:.2f}"
        cv2.putText(
            frame, conf_label, (center_x - 40, center_y), self.font, 0.5, color, 1
        )

        # Confidence bar visualization
        bar_width = 80
        bar_height = 6
        bar_x = center_x - bar_width // 2
        bar_y = center_y + 5

        # Background bar
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1,
        )

        # Confidence bar
        conf_width = int(bar_width * confidence)
        cv2.rectangle(
            frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), color, -1
        )

        # Danger warning
        if is_dangerous:
            if self.blink_state:  # Blinking effect
                danger_label = "⚠️ DANGER!"
                cv2.putText(
                    frame,
                    danger_label,
                    (center_x - 50, center_y + 25),
                    self.font,
                    0.7,
                    self.danger_color,
                    2,
                )

        # Fall alert
        if "fall_alert" in pose_result:
            alert = pose_result["fall_alert"]
            if self.blink_state:  # Blinking effect
                alert_label = "🚨 FALL ALERT!"
                cv2.putText(
                    frame,
                    alert_label,
                    (center_x - 60, center_y + 45),
                    self.font,
                    0.8,
                    self.alert_color,
                    2,
                )

                # Alert details
                alert_details = f"Type: {alert['alert_type']}"
                cv2.putText(
                    frame,
                    alert_details,
                    (center_x - 60, center_y + 65),
                    self.font,
                    0.5,
                    self.alert_color,
                    1,
                )

        # Pose-specific additional info
        if pose_class == "falling":
            # Draw falling direction indicator
            angles = pose_result.get("angles", {})
            body_orientation = angles.get("body_orientation", 0)
            if body_orientation > 0:
                # Draw orientation arrow
                arrow_length = 30
                angle_rad = math.radians(body_orientation)
                end_x = center_x + int(arrow_length * math.cos(angle_rad))
                end_y = center_y + int(arrow_length * math.sin(angle_rad))
                cv2.arrowedLine(
                    frame,
                    (center_x, center_y),
                    (end_x, end_y),
                    self.danger_color,
                    3,
                    tipLength=0.3,
                )

        elif pose_class == "running":
            # Draw speed indicator if available
            # This would require speed data from pose_result
            speed_label = "🏃 RUNNING"
            cv2.putText(
                frame,
                speed_label,
                (center_x - 40, center_y + 25),
                self.font,
                0.5,
                color,
                1,
            )

        return frame

    def draw_statistics_panel(
        self, frame: np.ndarray, pose_results: List[Dict], frame_info: Dict = None
    ) -> np.ndarray:
        """
        Draw statistics panel on the frame.

        Args:
            frame: Input frame
            pose_results: List of pose results
            frame_info: Frame information

        Returns:
            Frame with statistics panel
        """
        panel_x = 10
        panel_y = 10
        panel_width = 250
        line_height = 25

        # Background panel
        panel_height = 200
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (255, 255, 255),
            2,
        )

        # Title
        cv2.putText(
            frame,
            "POSE ANALYSIS",
            (panel_x + 10, panel_y + 25),
            self.font,
            0.7,
            (255, 255, 255),
            2,
        )

        y_offset = panel_y + 50

        # Frame info
        if frame_info:
            frame_text = f"Frame: {frame_info.get('frame_count', 0)}"
            cv2.putText(
                frame,
                frame_text,
                (panel_x + 10, y_offset),
                self.font,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += line_height

            fps_text = f"FPS: {frame_info.get('fps', 0):.1f}"
            cv2.putText(
                frame,
                fps_text,
                (panel_x + 10, y_offset),
                self.font,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += line_height

        # People count
        people_text = f"People: {len(pose_results)}"
        cv2.putText(
            frame,
            people_text,
            (panel_x + 10, y_offset),
            self.font,
            0.5,
            (255, 255, 255),
            1,
        )
        y_offset += line_height

        # Pose distribution
        pose_counts = {}
        dangerous_count = 0

        for pose_result in pose_results:
            pose_class = pose_result["pose_class"]
            pose_counts[pose_class] = pose_counts.get(pose_class, 0) + 1
            if pose_result["is_dangerous"]:
                dangerous_count += 1

        # Draw pose counts with colors
        for pose_class, count in pose_counts.items():
            color = self.pose_colors.get(pose_class, (255, 255, 255))
            pose_text = f"{pose_class}: {count}"
            cv2.putText(
                frame, pose_text, (panel_x + 10, y_offset), self.font, 0.5, color, 1
            )
            y_offset += line_height

        # Danger count
        if dangerous_count > 0:
            danger_text = f"DANGER: {dangerous_count}"
            cv2.putText(
                frame,
                danger_text,
                (panel_x + 10, y_offset),
                self.font,
                0.5,
                self.danger_color,
                2,
            )

        return frame

    def draw_alert_panel(
        self, frame: np.ndarray, pose_results: List[Dict]
    ) -> np.ndarray:
        """
        Draw alert panel for active alerts.

        Args:
            frame: Input frame
            pose_results: List of pose results

        Returns:
            Frame with alert panel
        """
        # Collect active alerts
        active_alerts = []
        for pose_result in pose_results:
            if "fall_alert" in pose_result:
                alert_info = {
                    "track_id": pose_result["track_id"],
                    "alert": pose_result["fall_alert"],
                    "pose_class": pose_result["pose_class"],
                }
                active_alerts.append(alert_info)

        if not active_alerts:
            return frame

        # Alert panel position (top right)
        frame_height, frame_width = frame.shape[:2]
        panel_width = 300
        panel_height = 50 + len(active_alerts) * 30
        panel_x = frame_width - panel_width - 10
        panel_y = 10

        # Blinking background for alerts
        bg_color = (0, 0, 100) if self.blink_state else (0, 0, 50)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            bg_color,
            -1,
        )
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            self.alert_color,
            3,
        )

        # Alert title
        cv2.putText(
            frame,
            "🚨 ACTIVE ALERTS",
            (panel_x + 10, panel_y + 25),
            self.font,
            0.7,
            (255, 255, 255),
            2,
        )

        # Draw individual alerts
        y_offset = panel_y + 50
        for alert_info in active_alerts:
            track_id = alert_info["track_id"]
            alert = alert_info["alert"]

            alert_text = f"ID {track_id}: {alert['alert_type']}"
            cv2.putText(
                frame,
                alert_text,
                (panel_x + 10, y_offset),
                self.font,
                0.5,
                (255, 255, 255),
                1,
            )

            conf_text = f"Conf: {alert['confidence']:.2f}"
            cv2.putText(
                frame,
                conf_text,
                (panel_x + 150, y_offset),
                self.font,
                0.5,
                (255, 255, 255),
                1,
            )

            y_offset += 30

        return frame

    def draw_legend(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw color legend for pose classes.

        Args:
            frame: Input frame

        Returns:
            Frame with legend
        """
        frame_height, frame_width = frame.shape[:2]
        legend_width = 200
        legend_height = 150
        legend_x = frame_width - legend_width - 10
        legend_y = frame_height - legend_height - 10

        # Background
        cv2.rectangle(
            frame,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            frame,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            (255, 255, 255),
            2,
        )

        # Title
        cv2.putText(
            frame,
            "LEGEND",
            (legend_x + 10, legend_y + 25),
            self.font,
            0.6,
            (255, 255, 255),
            2,
        )

        # Draw legend items
        y_offset = legend_y + 45
        for pose_class, color in self.pose_colors.items():
            # Color square
            cv2.rectangle(
                frame,
                (legend_x + 10, y_offset - 10),
                (legend_x + 25, y_offset + 5),
                color,
                -1,
            )

            # Label
            cv2.putText(
                frame,
                pose_class.upper(),
                (legend_x + 35, y_offset),
                self.font,
                0.4,
                color,
                1,
            )

            y_offset += 20

        return frame

    def create_summary_overlay(
        self, pose_results: List[Dict], session_stats: Dict = None
    ) -> np.ndarray:
        """
        Create a summary overlay image with session statistics.

        Args:
            pose_results: Current pose results
            session_stats: Session statistics

        Returns:
            Summary overlay image
        """
        # Create blank image for summary
        summary_img = np.zeros((400, 600, 3), dtype=np.uint8)

        # Title
        cv2.putText(
            summary_img,
            "POSE ANALYSIS SUMMARY",
            (50, 40),
            self.font,
            1.0,
            (255, 255, 255),
            2,
        )

        # Current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            summary_img,
            f"Time: {timestamp}",
            (50, 70),
            self.font,
            0.5,
            (200, 200, 200),
            1,
        )

        y_offset = 110

        # Current frame statistics
        if pose_results:
            cv2.putText(
                summary_img,
                "CURRENT FRAME:",
                (50, y_offset),
                self.font,
                0.7,
                (255, 255, 255),
                2,
            )
            y_offset += 30

            pose_counts = {}
            for pose_result in pose_results:
                pose_class = pose_result["pose_class"]
                pose_counts[pose_class] = pose_counts.get(pose_class, 0) + 1

            for pose_class, count in pose_counts.items():
                color = self.pose_colors.get(pose_class, (255, 255, 255))
                cv2.putText(
                    summary_img,
                    f"  {pose_class}: {count}",
                    (70, y_offset),
                    self.font,
                    0.6,
                    color,
                    1,
                )
                y_offset += 25

        # Session statistics
        if session_stats:
            y_offset += 20
            cv2.putText(
                summary_img,
                "SESSION STATS:",
                (50, y_offset),
                self.font,
                0.7,
                (255, 255, 255),
                2,
            )
            y_offset += 30

            for key, value in session_stats.items():
                cv2.putText(
                    summary_img,
                    f"  {key}: {value}",
                    (70, y_offset),
                    self.font,
                    0.5,
                    (200, 200, 200),
                    1,
                )
                y_offset += 25

        return summary_img
