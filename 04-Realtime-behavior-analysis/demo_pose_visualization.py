#!/usr/bin/env python3
"""
Interactive Demo for Pose Estimation Visualization
Gelişmiş overlay sistemi ile interaktif pose estimation demo'su.
"""

import sys
import time
from pathlib import Path

import cv2

# Add src directory to path
sys.path.append("src")

from config import Config  # noqa: E402
from detection import HumanDetector, HumanTracker  # noqa: E402
from pose import PoseEstimator  # noqa: E402
from visualization import PoseOverlay  # noqa: E402


def interactive_demo():
    """Interactive pose estimation demo with advanced visualization."""

    print("🎬 Interactive Pose Estimation Demo")
    print("=" * 50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Show statistics")
    print("  'r' - Reset statistics")
    print("  'o' - Toggle overlay modes")
    print("  'p' - Pause/Resume")
    print("  'SPACE' - Next frame (when paused)")
    print("=" * 50)

    # Initialize configuration
    config = Config()
    config.apply_optimized_settings(target_fps=25.0)

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

    # Find video files
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

    # Select video
    if len(video_files) == 1:
        selected_video = video_files[0]
    else:
        while True:
            try:
                choice = input(f"\nSelect video (1-{len(video_files)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(video_files):
                    selected_video = video_files[idx]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    print(f"\n🎬 Starting demo with: {selected_video.name}")

    # Run interactive demo
    run_demo(selected_video, detector, tracker, pose_estimator, overlay_system, config)


def run_demo(video_path, detector, tracker, pose_estimator, overlay_system, config):
    """Run the interactive demo."""

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

    # Demo state
    paused = False
    overlay_mode = 0  # 0: Full, 1: Minimal, 2: Keypoints only
    overlay_modes = ["Full Overlay", "Minimal", "Keypoints Only"]

    # Statistics
    session_stats = {
        "frames_processed": 0,
        "total_people": 0,
        "pose_distribution": {},
        "dangerous_situations": 0,
        "fall_alerts": 0,
        "start_time": time.time(),
    }

    frame_num = 0

    print("\n🚀 Demo started! Press 'h' for help")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("📹 End of video reached")
                break
            frame_num += 1

        start_time = time.time()

        # Process frame
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        pose_results = pose_estimator.estimate_pose(frame, tracks) if tracks else []

        # Update session statistics
        session_stats["frames_processed"] += 1
        session_stats["total_people"] += len(tracks)

        for pose_result in pose_results:
            pose_class = pose_result["pose_class"]
            session_stats["pose_distribution"][pose_class] = (
                session_stats["pose_distribution"].get(pose_class, 0) + 1
            )

            if pose_result["is_dangerous"]:
                session_stats["dangerous_situations"] += 1

            if "fall_alert" in pose_result:
                session_stats["fall_alerts"] += 1

        # Create visualization based on mode
        if overlay_mode == 0:  # Full overlay
            frame_info = {
                "frame_count": frame_num,
                "fps": fps,
                "processing_time": time.time() - start_time,
            }
            result_frame = overlay_system.draw_comprehensive_overlay(
                frame.copy(), pose_results, frame_info
            )

        elif overlay_mode == 1:  # Minimal
            result_frame = frame.copy()
            result_frame = detector.draw_detections(
                result_frame, detections, color=(0, 255, 0)
            )
            result_frame = tracker.draw_tracks(result_frame, tracks, color=(255, 0, 0))

            # Simple pose labels
            for pose_result in pose_results:
                body_center = pose_result.get("body_center", (0, 0))
                if body_center != (0, 0):
                    pose_class = pose_result["pose_class"]
                    track_id = pose_result["track_id"]
                    color = config.POSE_COLORS.get(pose_class, (255, 255, 255))

                    label = f"ID:{track_id} {pose_class}"
                    cv2.putText(
                        result_frame,
                        label,
                        (int(body_center[0] - 40), int(body_center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

        else:  # Keypoints only
            result_frame = frame.copy()
            if pose_results:
                result_frame = pose_estimator.draw_keypoints(result_frame, pose_results)

        # Add mode indicator
        mode_text = f"Mode: {overlay_modes[overlay_mode]} (Press 'o' to change)"
        cv2.putText(
            result_frame,
            mode_text,
            (10, result_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Add pause indicator
        if paused:
            cv2.putText(
                result_frame,
                "PAUSED - Press 'p' to resume",
                (result_frame.shape[1] // 2 - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        # Display frame
        cv2.imshow("Pose Estimation Demo", result_frame)

        # Handle key presses
        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
        elif key == ord(" ") and paused:
            # Next frame when paused
            continue
        elif key == ord("o"):
            overlay_mode = (overlay_mode + 1) % len(overlay_modes)
            print(f"🎨 Overlay mode: {overlay_modes[overlay_mode]}")
        elif key == ord("s"):
            show_statistics(session_stats, pose_results)
        elif key == ord("r"):
            session_stats = reset_statistics()
            print("🔄 Statistics reset")
        elif key == ord("h"):
            show_help()

    cap.release()
    cv2.destroyAllWindows()

    # Final summary
    show_final_summary(session_stats)


def show_statistics(session_stats, current_pose_results):
    """Show current statistics."""
    print("\n📊 CURRENT STATISTICS:")
    print(f"  Frames processed: {session_stats['frames_processed']}")
    print(f"  Total people detected: {session_stats['total_people']}")
    print(f"  Current people: {len(current_pose_results)}")

    if session_stats["pose_distribution"]:
        print("  Pose distribution:")
        for pose_class, count in session_stats["pose_distribution"].items():
            print(f"    - {pose_class}: {count}")

    print(f"  Dangerous situations: {session_stats['dangerous_situations']}")
    print(f"  Fall alerts: {session_stats['fall_alerts']}")

    elapsed_time = time.time() - session_stats["start_time"]
    print(f"  Session duration: {elapsed_time:.1f}s")


def reset_statistics():
    """Reset session statistics."""
    return {
        "frames_processed": 0,
        "total_people": 0,
        "pose_distribution": {},
        "dangerous_situations": 0,
        "fall_alerts": 0,
        "start_time": time.time(),
    }


def show_help():
    """Show help information."""
    print("\n❓ HELP:")
    print("  'q' - Quit demo")
    print("  's' - Show statistics")
    print("  'r' - Reset statistics")
    print("  'o' - Toggle overlay modes (Full/Minimal/Keypoints)")
    print("  'p' - Pause/Resume playback")
    print("  'SPACE' - Next frame (when paused)")
    print("  'h' - Show this help")


def show_final_summary(session_stats):
    """Show final session summary."""
    print("\n🏁 FINAL SESSION SUMMARY:")
    print("=" * 40)

    elapsed_time = time.time() - session_stats["start_time"]
    avg_fps = (
        session_stats["frames_processed"] / elapsed_time if elapsed_time > 0 else 0
    )

    print(f"📈 Session duration: {elapsed_time:.1f}s")
    print(f"📊 Frames processed: {session_stats['frames_processed']}")
    print(f"⚡ Average FPS: {avg_fps:.1f}")
    print(f"👥 Total people detected: {session_stats['total_people']}")

    if session_stats["pose_distribution"]:
        print("🤸 Pose distribution:")
        total_poses = sum(session_stats["pose_distribution"].values())
        for pose_class, count in session_stats["pose_distribution"].items():
            percentage = (count / total_poses) * 100 if total_poses > 0 else 0
            print(f"  - {pose_class}: {count} ({percentage:.1f}%)")

    print(f"⚠️ Dangerous situations: {session_stats['dangerous_situations']}")
    print(f"🚨 Fall alerts: {session_stats['fall_alerts']}")

    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    interactive_demo()
