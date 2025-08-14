#!/usr/bin/env python3
"""
Simple Pose Estimation Test Script

Streamlined testing tool for the pose estimation and classification system.
This script implements Task 11 requirements from the pose estimation specification:

Features:
- Automated processing of all videos in data/inputs directory
- Real-time pose estimation with classification results display
- Comprehensive pose classification statistics and distribution analysis
- Performance metrics calculation (FPS, processing times, accuracy rates)
- Clean, readable output with progress tracking and emoji indicators

Usage:
    python test_pose_simple.py

The script will automatically discover video files (.mp4, .avi) in the data/inputs
directory and process them sequentially, providing detailed statistics for each
video and overall summary results.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Add src directory to path
sys.path.append("src")

from config import Config  # noqa: E402
from detection import HumanDetector, HumanTracker  # noqa: E402
from pose import PoseEstimator  # noqa: E402


def main():
    """
    Main test execution function for pose estimation system validation.

    Orchestrates the complete testing workflow:
    1. System initialization with optimized settings
    2. Video discovery and validation
    3. Sequential processing of all test videos
    4. Statistics aggregation and analysis
    5. Comprehensive results presentation
    """
    print("🧪 POSE ESTIMATION TEST SCRIPT")
    print("=" * 50)

    # Initialize system with performance-optimized configuration
    config = Config()
    config.apply_optimized_settings(target_fps=20.0)  # Balanced performance/accuracy

    try:
        detector = HumanDetector(config)
        tracker = HumanTracker(config)
        pose_estimator = PoseEstimator(config)
        print("✅ Modules initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing modules: {e}")
        return

    # Find videos in data/inputs
    input_dir = Path("data/inputs")
    if not input_dir.exists():
        print(f"❌ Directory not found: {input_dir}")
        return

    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi"))
    if not video_files:
        print(f"❌ No video files found in {input_dir}")
        return

    print(f"📹 Found {len(video_files)} videos to test")

    # Test each video
    overall_stats = {
        "total_videos": 0,
        "total_frames": 0,
        "total_people": 0,
        "total_poses": 0,
        "total_processing_time": 0.0,
        "pose_distribution": defaultdict(int),
        "dangerous_situations": 0,
        "fall_alerts": 0,
    }

    for i, video_file in enumerate(video_files, 1):
        print(f"\n🎬 Testing Video {i}/{len(video_files)}: {video_file.name}")
        print("-" * 40)

        video_stats = test_video(video_file, detector, tracker, pose_estimator, config)

        # Update overall statistics
        overall_stats["total_videos"] += 1
        overall_stats["total_frames"] += video_stats["frames_processed"]
        overall_stats["total_people"] += video_stats["people_detected"]
        overall_stats["total_poses"] += video_stats["poses_classified"]
        overall_stats["total_processing_time"] += video_stats["total_processing_time"]

        for pose_class, count in video_stats["pose_distribution"].items():
            overall_stats["pose_distribution"][pose_class] += count

        overall_stats["dangerous_situations"] += video_stats["dangerous_situations"]
        overall_stats["fall_alerts"] += video_stats["fall_alerts"]

        # Display video results
        display_video_results(video_file.name, video_stats)

    # Display overall results
    display_overall_results(overall_stats)

    print("\n🏁 TEST COMPLETED!")


def test_video(video_path, detector, tracker, pose_estimator, config):
    """
    Process a single video through the complete pose estimation pipeline.

    Performs comprehensive analysis including:
    - Video metadata extraction and validation
    - Frame-by-frame pose detection and classification
    - Real-time performance monitoring
    - Statistical data collection and aggregation
    - Progress reporting with performance metrics

    Args:
        video_path: Path to the video file to process
        detector: Initialized human detection module
        tracker: Initialized human tracking module
        pose_estimator: Initialized pose estimation and classification module
        config: System configuration object

    Returns:
        Dictionary containing comprehensive video analysis results:
        - Processing statistics (frames, people, poses)
        - Performance metrics (FPS, processing times)
        - Pose classification distribution
        - Safety analysis (dangerous situations, fall alerts)
    """
    stats = {
        "frames_processed": 0,
        "people_detected": 0,
        "poses_classified": 0,
        "processing_times": [],
        "pose_distribution": defaultdict(int),
        "dangerous_situations": 0,
        "fall_alerts": 0,
        "total_processing_time": 0.0,
    }

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return stats

        # Extract video metadata for analysis planning
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📊 Video: {width}x{height}, {fps:.1f} FPS, {frame_count} frames")

        # Reset tracker state for clean per-video analysis
        tracker = HumanTracker(config)

        frame_num = 0
        max_frames = min(150, frame_count)  # Limit processing for efficient testing

        print(f"🔄 Processing {max_frames} frames...")

        while frame_num < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Apply frame skipping for performance optimization
            if frame_num % config.POSE_FRAME_SKIP != 0:
                continue

            start_time = time.time()

            # Stage 1: Human detection in current frame
            detections = detector.detect(frame)

            # Stage 2: Multi-object tracking across frames
            tracks = tracker.update(detections, frame)

            # Stage 3: Pose estimation and classification
            pose_results = []
            if tracks:
                pose_results = pose_estimator.estimate_pose(frame, tracks)

            processing_time = time.time() - start_time
            stats["processing_times"].append(processing_time)
            stats["total_processing_time"] += processing_time

            # Update processing statistics
            stats["frames_processed"] += 1
            stats["people_detected"] += len(tracks)
            stats["poses_classified"] += len(pose_results)

            # Analyze pose classification results for safety and distribution
            for pose_result in pose_results:
                pose_class = pose_result["pose_class"]
                stats["pose_distribution"][pose_class] += 1

                # Track dangerous situations for safety analysis
                if pose_result["is_dangerous"]:
                    stats["dangerous_situations"] += 1

                # Count active fall alerts for emergency response metrics
                if "fall_alert" in pose_result:
                    stats["fall_alerts"] += 1

            # Show progress every 30 frames
            if frame_num % 30 == 0:
                avg_time = (
                    np.mean(stats["processing_times"][-30:])
                    if stats["processing_times"]
                    else 0
                )
                current_fps = 1.0 / avg_time if avg_time > 0 else 0
                print(
                    f"  📈 Frame {frame_num}/{max_frames} - FPS: {current_fps:.1f} - "
                    f"People: {len(tracks)} - Poses: {len(pose_results)}"
                )

        cap.release()

    except Exception as e:
        print(f"❌ Error processing video: {e}")

    return stats


def display_video_results(video_name, stats):
    """
    Display comprehensive analysis results for a single video.

    Presents formatted statistics including processing metrics,
    pose classification distribution, performance analysis,
    and safety assessment with clear visual indicators.
    """
    print(f"\n📊 RESULTS for {video_name}:")
    print(f"  ✅ Frames processed: {stats['frames_processed']}")
    print(f"  👥 People detected: {stats['people_detected']}")
    print(f"  🤸 Poses classified: {stats['poses_classified']}")

    if stats["processing_times"]:
        avg_time = np.mean(stats["processing_times"])
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"  ⚡ Average FPS: {avg_fps:.1f}")
        print(f"  🕐 Processing time per frame: {avg_time:.3f}s")
        print(f"  ⏱️ Total processing time: {stats['total_processing_time']:.1f}s")

    if stats["pose_distribution"]:
        print("  📈 Pose classification:")
        total_poses = sum(stats["pose_distribution"].values())
        for pose_class, count in stats["pose_distribution"].items():
            percentage = (count / total_poses) * 100 if total_poses > 0 else 0
            print(f"    - {pose_class.upper()}: {count} ({percentage:.1f}%)")

    if stats["dangerous_situations"] > 0:
        danger_rate = (
            (stats["dangerous_situations"] / stats["poses_classified"] * 100)
            if stats["poses_classified"] > 0
            else 0
        )
        print(
            f"  ⚠️ Dangerous situations: {stats['dangerous_situations']} "
            f"({danger_rate:.1f}%)"
        )

    if stats["fall_alerts"] > 0:
        print(f"  🚨 Fall alerts: {stats['fall_alerts']}")


def display_overall_results(overall_stats):
    """
    Display comprehensive summary of all video test results.

    Provides aggregated statistics across all processed videos including:
    - Total processing metrics and performance analysis
    - Overall pose classification distribution
    - System-wide safety assessment
    - Performance efficiency metrics and ratios
    """
    print("\n🌟 OVERALL RESULTS")
    print("=" * 40)
    print(f"📹 Videos processed: {overall_stats['total_videos']}")
    print(f"🖼️ Total frames: {overall_stats['total_frames']}")
    print(f"👥 Total people detected: {overall_stats['total_people']}")
    print(f"🤸 Total poses classified: {overall_stats['total_poses']}")

    if overall_stats["total_processing_time"] > 0:
        overall_fps = (
            overall_stats["total_frames"] / overall_stats["total_processing_time"]
        )
        avg_processing_time = (
            overall_stats["total_processing_time"] / overall_stats["total_frames"]
        )
        print(f"⚡ Overall FPS: {overall_fps:.1f}")
        print(f"🕐 Average processing time: {avg_processing_time:.3f}s per frame")
        print(
            f"⏱️ Total processing time: "
            f"{overall_stats['total_processing_time']:.1f}s"
        )

    if overall_stats["pose_distribution"]:
        print("\n📊 Overall pose distribution:")
        total_poses = sum(overall_stats["pose_distribution"].values())
        for pose_class, count in overall_stats["pose_distribution"].items():
            percentage = (count / total_poses) * 100 if total_poses > 0 else 0
            print(f"  - {pose_class.upper()}: {count} ({percentage:.1f}%)")

    if overall_stats["dangerous_situations"] > 0:
        danger_rate = (
            (overall_stats["dangerous_situations"] / overall_stats["total_poses"] * 100)
            if overall_stats["total_poses"] > 0
            else 0
        )
        print(
            f"\n⚠️ Total dangerous situations: "
            f"{overall_stats['dangerous_situations']} ({danger_rate:.1f}%)"
        )

    if overall_stats["fall_alerts"] > 0:
        print(f"🚨 Total fall alerts: {overall_stats['fall_alerts']}")

    # Performance summary
    print("\n📈 PERFORMANCE SUMMARY:")
    if overall_stats["total_people"] > 0:
        poses_per_person = overall_stats["total_poses"] / overall_stats["total_people"]
        print(f"  - Poses per person detected: {poses_per_person:.1f}")

    if overall_stats["total_frames"] > 0:
        people_per_frame = overall_stats["total_people"] / overall_stats["total_frames"]
        poses_per_frame = overall_stats["total_poses"] / overall_stats["total_frames"]
        print(f"  - Average people per frame: {people_per_frame:.1f}")
        print(f"  - Average poses per frame: {poses_per_frame:.1f}")


if __name__ == "__main__":
    main()
