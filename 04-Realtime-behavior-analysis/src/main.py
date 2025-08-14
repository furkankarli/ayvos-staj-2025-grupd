#!/usr/bin/env python3
"""
Main execution script for realtime behavior analysis system.
Integrates all modules: detection, pose estimation, analysis, and segmentation.
"""

import time
from pathlib import Path

import cv2
from analysis import BehaviorAnalyzer
from config import Config
from detection import HumanDetector, HumanTracker
from pose import PoseEstimator
from segmentation import Segmentator
from utils import VideoInput
from visualization import PoseOverlay


def main():
    """Main function to run the realtime behavior analysis system."""
    config = Config()

    print("Realtime Behavior Analysis System initialized")

    # Initialize modules
    detector = HumanDetector(config)
    tracker = HumanTracker(config)
    pose_estimator = PoseEstimator(config)
    analyzer = BehaviorAnalyzer(config)
    segmentator = Segmentator(config)
    overlay_system = PoseOverlay(config)

    # Test video input (you can change this to your video path)
    video_path = "data/inputs/test_video.mp4"  # Change this to your video path

    # Check if video exists
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("Please place a test video in the data/inputs/ directory")
        return

    try:
        # Initialize video input
        video = VideoInput(video_path)
        video.open()

        print(f"Video opened successfully: {video_path}")
        print(f"Video info: {video.get_frame_info()}")

        # Main processing loop
        frame_count = 0
        total_processing_time = 0.0

        print("\n🚀 Starting realtime pose estimation and behavior analysis...")
        print("Press 'q' to quit, 's' to show statistics")

        while True:
            start_time = time.time()

            frame = video.read_frame()
            if frame is None:
                break

            frame_count += 1

            # Skip frames for performance if configured
            if frame_count % config.POSE_FRAME_SKIP != 0:
                continue

            print(f"\n📹 Processing frame {frame_count}")

            # 1. Human Detection
            detection_start = time.time()
            detections = detector.detect(frame)
            detection_time = time.time() - detection_start
            print(f"  🔍 Detected {len(detections)} humans ({detection_time:.3f}s)")

            # 2. Human Tracking
            tracking_start = time.time()
            tracks = tracker.update(detections, frame)
            tracking_time = time.time() - tracking_start
            print(f"  🎯 Tracking {len(tracks)} humans ({tracking_time:.3f}s)")

            # 3. Pose Estimation and Classification
            pose_results = []
            if config.ENABLE_POSE_ESTIMATION and tracks:
                pose_start = time.time()
                pose_results = pose_estimator.estimate_pose(frame, tracks)
                pose_time = time.time() - pose_start

                # Print pose results
                print(
                    f"  🤸 Pose estimation: {len(pose_results)} poses ({pose_time:.3f}s)"
                )
                for pose_result in pose_results:
                    track_id = pose_result["track_id"]
                    pose_class = pose_result["pose_class"]
                    confidence = pose_result["pose_confidence"]
                    is_dangerous = pose_result["is_dangerous"]

                    status_icon = "⚠️" if is_dangerous else "✅"
                    print(
                        f"    {status_icon} ID {track_id}: "
                        f"{pose_class.upper()} (conf: {confidence:.2f})"
                    )

                    # Check for fall alerts
                    if "fall_alert" in pose_result:
                        alert = pose_result["fall_alert"]
                        print(
                            f"    🚨 FALL ALERT - Type: {alert['alert_type']}, "
                            f"Confidence: {alert['confidence']:.2f}"
                        )

            # 4. Draw comprehensive results with advanced overlay system
            # Start with original frame
            result_frame = frame.copy()

            # Draw basic detections and tracks (lighter colors as background)
            result_frame = detector.draw_detections(
                result_frame, detections, color=(100, 0, 0)
            )
            result_frame = tracker.draw_tracks(result_frame, tracks, color=(0, 100, 0))

            # Draw pose keypoints and skeleton
            if pose_results:
                result_frame = pose_estimator.draw_keypoints(result_frame, pose_results)

            # Calculate performance metrics first
            frame_time = time.time() - start_time
            total_processing_time += frame_time
            avg_fps = (
                frame_count / total_processing_time if total_processing_time > 0 else 0
            )

            # Apply advanced overlay system
            frame_info = {
                "frame_count": frame_count,
                "fps": avg_fps,
                "processing_time": frame_time,
            }
            result_frame = overlay_system.draw_comprehensive_overlay(
                result_frame, pose_results, frame_info
            )

            # 5. Display frame
            cv2.imshow("Realtime Behavior Analysis - Pose Estimation", result_frame)

            # Update performance stats
            config.update_performance_stats(frame_time)

            # Adaptive performance adjustment (every 30 frames)
            if frame_count % 30 == 0:
                config.adapt_settings_for_performance()

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Show statistics
                print("\n📊 STATISTICS:")
                pose_stats = pose_estimator.get_comprehensive_stats(pose_results)
                print(f"  Total people: {pose_stats['summary']['total_people']}")
                print(
                    f"  Dangerous situations: "
                    f"{pose_stats['summary']['dangerous_situations']}"
                )
                print(f"  Active alerts: {pose_stats['summary']['active_alerts']}")
                print(
                    f"  Average confidence: "
                    f"{pose_stats['summary']['avg_confidence']:.2f}"
                )
                print(f"  Performance: {config.get_performance_stats()}")

            # Limit processing for demo (remove this in production)
            if frame_count > 300:  # Process more frames to see pose estimation
                print("\n⏹️ Demo limit reached (300 frames)")
                break

        video.release()
        cv2.destroyAllWindows()

        # Final statistics
        print("\n🏁 FINAL STATISTICS:")
        if frame_count > 0:
            final_stats = config.get_performance_stats()
            print(f"  📈 Total frames processed: {frame_count}")
            print(f"  ⏱️ Average FPS: {final_stats['avg_fps']:.2f}")
            print(
                f"  🕐 Total processing time: "
                f"{final_stats['total_processing_time']:.2f}s"
            )

            # Get final pose statistics
            if "pose_estimator" in locals():
                fall_stats = pose_estimator.get_fall_statistics()
                movement_stats = pose_estimator.get_movement_statistics()
                print(
                    f"  👥 Total tracked people: "
                    f"{fall_stats['total_tracked_people']}"
                )
                print(f"  🚨 Fall alerts generated: {fall_stats['active_fall_alerts']}")
                print(
                    f"  🏃 People detected running: "
                    f"{movement_stats['running_people']}"
                )

        print("\n✅ Processing completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
