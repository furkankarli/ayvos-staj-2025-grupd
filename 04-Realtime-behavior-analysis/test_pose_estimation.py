#!/usr/bin/env python3
"""
Test script for pose estimation and classification.
Tests the integrated pose estimation system with data/inputs videos.
"""


import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src directory to path
sys.path.append("src")

from config import Config  # noqa: E402
from detection import HumanDetector, HumanTracker  # noqa: E402
from pose import PoseEstimator  # noqa: E402
from visualization import PoseOverlay  # noqa: E402


def test_pose_estimation():
    """Test pose estimation with videos from data/inputs directory."""

    # Initialize configuration
    config = Config()

    # Apply optimized settings for testing
    config.apply_optimized_settings(target_fps=20.0)

    print("🧪 Pose Estimation Test Script")
    print("=" * 50)

    # Initialize modules
    print("📦 Initializing modules...")
    try:
        detector = HumanDetector(config)
        tracker = HumanTracker(config)
        pose_estimator = PoseEstimator(config)
        overlay_system = PoseOverlay(config)
        print("✅ All modules initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing modules: {e}")
        return

    # Find test videos
    input_dir = Path("data/inputs")
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        print("Please create data/inputs directory and add test videos")
        return

    video_files = (
        list(input_dir.glob("*.mp4"))
        + list(input_dir.glob("*.avi"))
        + list(input_dir.glob("*.mov"))
    )

    if not video_files:
        print(f"❌ No video files found in {input_dir}")
        print("Supported formats: .mp4, .avi, .mov")
        return

    print(f"📹 Found {len(video_files)} video files:")
    for i, video_file in enumerate(video_files):
        print(f"  {i+1}. {video_file.name}")

    # Test each video
    for video_file in video_files:
        print(f"\n🎬 Testing with: {video_file.name}")
        print("-" * 40)

        test_single_video(
            video_file, detector, tracker, pose_estimator, overlay_system, config
        )

    print("\n🏁 All tests completed!")


def test_single_video(
    video_path, detector, tracker, pose_estimator, overlay_system, config
):
    """Test pose estimation on a single video."""

    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return

        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📊 Video info: {width}x{height}, {fps:.1f} FPS, {frame_count} frames")

        # Statistics
        stats = {
            "frames_processed": 0,
            "people_detected": 0,
            "poses_classified": 0,
            "pose_distribution": {},
            "dangerous_situations": 0,
            "fall_alerts": 0,
            "processing_times": [],
        }

        frame_num = 0
        max_frames = min(100, frame_count)  # Process max 100 frames for testing

        print(f"🔄 Processing {max_frames} frames...")

        while frame_num < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            start_time = time.time()

            # Skip frames for performance
            if frame_num % config.POSE_FRAME_SKIP != 0:
                continue

            # 1. Detection
            detections = detector.detect(frame)

            # 2. Tracking
            tracks = tracker.update(detections, frame)

            # 3. Pose estimation
            pose_results = []
            if tracks:
                pose_results = pose_estimator.estimate_pose(frame, tracks)

            # Update statistics
            processing_time = time.time() - start_time
            stats["processing_times"].append(processing_time)
            stats["frames_processed"] += 1
            stats["people_detected"] += len(tracks)
            stats["poses_classified"] += len(pose_results)

            # Analyze pose results
            for pose_result in pose_results:
                pose_class = pose_result["pose_class"]
                stats["pose_distribution"][pose_class] = (
                    stats["pose_distribution"].get(pose_class, 0) + 1
                )

                if pose_result["is_dangerous"]:
                    stats["dangerous_situations"] += 1

                if "fall_alert" in pose_result:
                    stats["fall_alerts"] += 1

            # Show progress every 20 frames
            if frame_num % 20 == 0:
                avg_time = (
                    np.mean(stats["processing_times"][-20:])
                    if stats["processing_times"]
                    else 0
                )
                current_fps = 1.0 / avg_time if avg_time > 0 else 0
                print(
                    f"  📈 Frame {frame_num}/{max_frames} - FPS: {current_fps:.1f} - "
                    f"People: {len(tracks)} - Poses: {len(pose_results)}"
                )

                # Optional: Show frame with overlay (uncomment to enable)
                # if pose_results:
                #     frame_info = {'frame_count': frame_num, 'fps': current_fps}
                #     overlay_frame = overlay_system.draw_comprehensive_overlay(
                #         frame.copy(), pose_results, frame_info)
                #     cv2.imshow(f'Test: {video_path.name}', overlay_frame)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break

        cap.release()

        # Print final statistics
        print(f"\n📊 RESULTS for {video_path.name}:")
        print(f"  ✅ Frames processed: {stats['frames_processed']}")
        print(f"  👥 Total people detected: {stats['people_detected']}")
        print(f"  🤸 Total poses classified: {stats['poses_classified']}")

        if stats["processing_times"]:
            avg_processing_time = np.mean(stats["processing_times"])
            avg_fps = 1.0 / avg_processing_time
            print(f"  ⚡ Average processing time: {avg_processing_time:.3f}s")
            print(f"  🎯 Average FPS: {avg_fps:.1f}")

        if stats["pose_distribution"]:
            print("  📈 Pose distribution:")
            for pose_class, count in stats["pose_distribution"].items():
                percentage = (
                    (count / stats["poses_classified"]) * 100
                    if stats["poses_classified"] > 0
                    else 0
                )
                print(f"    - {pose_class}: {count} ({percentage:.1f}%)")

        if stats["dangerous_situations"] > 0:
            print(
                f"  ⚠️ Dangerous situations detected: "
                f"{stats['dangerous_situations']}"
            )

        if stats["fall_alerts"] > 0:
            print(f"  🚨 Fall alerts generated: {stats['fall_alerts']}")

    except Exception as e:
        print(f"❌ Error processing video {video_path}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_pose_estimation()
