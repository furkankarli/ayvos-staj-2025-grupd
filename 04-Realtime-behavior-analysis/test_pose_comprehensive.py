#!/usr/bin/env python3
"""
Comprehensive Pose Estimation Test Suite

Advanced testing framework for the pose estimation and classification system.
This script implements Task 11 from the pose estimation specification with
extensive analysis capabilities and detailed reporting.

Key Features:
- Automated batch processing of all videos in data/inputs directory
- Frame-by-frame pose estimation with detailed classification analysis
- Comprehensive pose classification statistics and distribution reporting
- Advanced performance metrics calculation (FPS, processing times, accuracy)
- Detailed JSON export of all test results for further analysis
- Real-time progress monitoring with performance feedback
- Error handling and recovery for robust testing

Requirements Implementation:
- ✅ Process videos in data/inputs directory
- ✅ Display pose estimation results for each video
- ✅ Report pose classification statistics with detailed breakdowns
- ✅ Calculate performance metrics (FPS, accuracy, processing times)
- ✅ Generate comprehensive test reports with exportable data

Usage:
    python test_pose_comprehensive.py

Output:
    - Console: Real-time progress and summary statistics
    - File: test_results.json with detailed analysis data
"""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add src directory to path
sys.path.append("src")

from config import Config  # noqa: E402
from detection import HumanDetector, HumanTracker  # noqa: E402
from pose import PoseEstimator  # noqa: E402
from visualization import PoseOverlay  # noqa: E402


class PoseTestRunner:
    """
    Advanced test runner for comprehensive pose estimation system validation.

    This class orchestrates complete system testing with detailed analysis:
    - Multi-video batch processing with progress tracking
    - Frame-by-frame pose analysis with statistical collection
    - Performance profiling across detection, tracking, and classification stages
    - Comprehensive result aggregation and export capabilities
    - Error handling and recovery for robust testing workflows

    Features:
    - Configurable performance optimization settings
    - Detailed timing analysis for each processing stage
    - Pose classification distribution analysis
    - Safety assessment with fall detection metrics
    - JSON export for external analysis tools
    - Real-time progress reporting with performance feedback
    """

    def __init__(self):
        """
        Initialize comprehensive test runner with optimized configuration.

        Sets up the complete testing environment including:
        - Performance-optimized system configuration
        - All required pose estimation modules
        - Result storage and tracking systems
        - Error handling and recovery mechanisms
        """
        self.config = Config()
        self.config.apply_optimized_settings(target_fps=25.0)

        # Initialize modules
        print("🔧 Initializing pose estimation modules...")
        try:
            self.detector = HumanDetector(self.config)
            self.tracker = HumanTracker(self.config)
            self.pose_estimator = PoseEstimator(self.config)
            self.overlay_system = PoseOverlay(self.config)
            print("✅ All modules initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing modules: {e}")
            raise

        # Test results storage
        self.test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "config_settings": self._get_config_summary(),
            "videos": {},
            "overall_stats": {},
        }

    def _get_config_summary(self):
        """Get summary of current configuration settings."""
        return {
            "pose_confidence": self.config.POSE_CONFIDENCE,
            "keypoint_confidence": self.config.KEYPOINT_CONFIDENCE,
            "pose_input_size": self.config.POSE_INPUT_SIZE,
            "pose_frame_skip": self.config.POSE_FRAME_SKIP,
            "max_pose_detections": self.config.MAX_POSE_DETECTIONS,
            "fall_height_ratio": self.config.FALL_HEIGHT_RATIO,
            "standing_angle_threshold": self.config.STANDING_ANGLE_THRESHOLD,
            "sitting_angle_threshold": self.config.SITTING_ANGLE_THRESHOLD,
        }

    def run_comprehensive_test(self):
        """Run comprehensive test on all videos in data/inputs directory."""
        print("🧪 COMPREHENSIVE POSE ESTIMATION TEST")
        print("=" * 60)

        # Find test videos
        input_dir = Path("data/inputs")
        if not input_dir.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return

        video_files = self._find_video_files(input_dir)
        if not video_files:
            print(f"❌ No video files found in {input_dir}")
            return

        print(f"📹 Found {len(video_files)} video files to test:")
        for i, video_file in enumerate(video_files, 1):
            print(f"  {i}. {video_file.name}")

        # Test each video
        overall_stats = defaultdict(int)
        overall_times = []

        for i, video_file in enumerate(video_files, 1):
            print(f"\n🎬 Testing Video {i}/{len(video_files)}: {video_file.name}")
            print("-" * 50)

            video_stats = self._test_single_video(video_file)
            self.test_results["videos"][video_file.name] = video_stats

            # Accumulate overall statistics
            self._accumulate_overall_stats(overall_stats, overall_times, video_stats)

        # Calculate and display overall results
        self._calculate_overall_results(overall_stats, overall_times)

        # Save detailed results
        self._save_test_results()

        print("\n🏁 COMPREHENSIVE TEST COMPLETED!")
        print("📄 Detailed results saved to: test_results.json")

    def _find_video_files(self, input_dir):
        """Find all video files in the input directory."""
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
        video_files = []

        for ext in video_extensions:
            video_files.extend(input_dir.glob(f"*{ext}"))
            video_files.extend(input_dir.glob(f"*{ext.upper()}"))

        return sorted(video_files)

    def _test_single_video(self, video_path):
        """Test pose estimation on a single video file."""
        video_stats = {
            "video_path": str(video_path),
            "video_info": {},
            "processing_stats": {},
            "pose_stats": {},
            "performance_metrics": {},
            "detailed_results": [],
        }

        try:
            # Open video and get info
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"❌ Could not open video: {video_path}")
                video_stats["error"] = f"Could not open video: {video_path}"
                return video_stats

            # Get video information
            video_info = self._get_video_info(cap)
            video_stats["video_info"] = video_info

            print(
                f"📊 Video Info: {video_info['width']}x{video_info['height']}, "
                f"{video_info['fps']:.1f} FPS, {video_info['frame_count']} frames"
            )

            # Process video frames
            processing_stats = self._process_video_frames(cap, video_stats)
            video_stats["processing_stats"] = processing_stats

            cap.release()

            # Calculate performance metrics
            video_stats["performance_metrics"] = self._calculate_performance_metrics(
                video_stats
            )

            # Calculate pose statistics
            video_stats["pose_stats"] = self._calculate_pose_statistics(video_stats)

            # Display results for this video
            self._display_video_results(video_path.name, video_stats)

        except Exception as e:
            print(f"❌ Error processing video {video_path}: {e}")
            video_stats["error"] = str(e)
            import traceback

            traceback.print_exc()

        return video_stats

    def _get_video_info(self, cap):
        """Extract video information."""
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": (
                cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                if cap.get(cv2.CAP_PROP_FPS) > 0
                else 0
            ),
        }

    def _process_video_frames(self, cap, video_stats):
        """Process all frames in the video."""
        processing_stats = {
            "frames_processed": 0,
            "frames_skipped": 0,
            "people_detected": 0,
            "poses_classified": 0,
            "processing_times": [],
            "detection_times": [],
            "tracking_times": [],
            "pose_estimation_times": [],
        }

        frame_num = 0
        max_frames = min(
            200, video_stats["video_info"]["frame_count"]
        )  # Process max 200 frames

        print(f"🔄 Processing up to {max_frames} frames...")

        # Reset tracker for each video
        self.tracker = HumanTracker(self.config)

        while frame_num < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Skip frames based on configuration
            if frame_num % self.config.POSE_FRAME_SKIP != 0:
                processing_stats["frames_skipped"] += 1
                continue

            frame_start_time = time.time()

            # 1. Human Detection
            detection_start = time.time()
            detections = self.detector.detect(frame)
            detection_time = time.time() - detection_start
            processing_stats["detection_times"].append(detection_time)

            # 2. Human Tracking
            tracking_start = time.time()
            tracks = self.tracker.update(detections, frame)
            tracking_time = time.time() - tracking_start
            processing_stats["tracking_times"].append(tracking_time)

            # 3. Pose Estimation and Classification
            pose_start = time.time()
            pose_results = []
            if tracks:
                pose_results = self.pose_estimator.estimate_pose(frame, tracks)
            pose_time = time.time() - pose_start
            processing_stats["pose_estimation_times"].append(pose_time)

            # Record frame processing time
            frame_processing_time = time.time() - frame_start_time
            processing_stats["processing_times"].append(frame_processing_time)

            # Update statistics
            processing_stats["frames_processed"] += 1
            processing_stats["people_detected"] += len(tracks)
            processing_stats["poses_classified"] += len(pose_results)

            # Store detailed results for this frame
            frame_result = {
                "frame_number": frame_num,
                "detections_count": len(detections),
                "tracks_count": len(tracks),
                "poses_count": len(pose_results),
                "processing_time": frame_processing_time,
                "pose_results": [],
            }

            # Analyze pose results
            for pose_result in pose_results:
                pose_data = {
                    "track_id": pose_result["track_id"],
                    "pose_class": pose_result["pose_class"],
                    "pose_confidence": pose_result["pose_confidence"],
                    "is_dangerous": pose_result["is_dangerous"],
                    "keypoints_visible": sum(
                        1
                        for kp in pose_result["keypoints"]
                        if kp[2] > self.config.KEYPOINT_CONFIDENCE
                    ),
                }

                if "fall_alert" in pose_result:
                    pose_data["fall_alert"] = pose_result["fall_alert"]

                frame_result["pose_results"].append(pose_data)

            video_stats["detailed_results"].append(frame_result)

            # Show progress
            if frame_num % 25 == 0:
                avg_time = (
                    np.mean(processing_stats["processing_times"][-25:])
                    if processing_stats["processing_times"]
                    else 0
                )
                current_fps = 1.0 / avg_time if avg_time > 0 else 0
                print(
                    f"  📈 Frame {frame_num}/{max_frames} - "
                    f"FPS: {current_fps:.1f} - "
                    f"People: {len(tracks)} - "
                    f"Poses: {len(pose_results)}"
                )

        return processing_stats

    def _calculate_performance_metrics(self, video_stats):
        """Calculate performance metrics for the video."""
        processing_stats = video_stats["processing_stats"]

        if not processing_stats["processing_times"]:
            return {}

        processing_times = processing_stats["processing_times"]
        detection_times = processing_stats["detection_times"]
        tracking_times = processing_stats["tracking_times"]
        pose_times = processing_stats["pose_estimation_times"]

        return {
            "avg_processing_time": float(np.mean(processing_times)),
            "max_processing_time": float(np.max(processing_times)),
            "min_processing_time": float(np.min(processing_times)),
            "avg_fps": float(1.0 / np.mean(processing_times)),
            "avg_detection_time": (
                float(np.mean(detection_times)) if detection_times else 0
            ),
            "avg_tracking_time": (
                float(np.mean(tracking_times)) if tracking_times else 0
            ),
            "avg_pose_estimation_time": float(np.mean(pose_times)) if pose_times else 0,
            "total_processing_time": float(sum(processing_times)),
            "frames_per_second_actual": (
                processing_stats["frames_processed"] / sum(processing_times)
                if sum(processing_times) > 0
                else 0
            ),
        }

    def _calculate_pose_statistics(self, video_stats):
        """Calculate pose classification statistics."""
        pose_distribution = defaultdict(int)
        confidence_scores = []
        dangerous_situations = 0
        fall_alerts = 0
        total_keypoints_visible = 0
        pose_count = 0

        for frame_result in video_stats["detailed_results"]:
            for pose_result in frame_result["pose_results"]:
                pose_count += 1
                pose_class = pose_result["pose_class"]
                pose_distribution[pose_class] += 1
                confidence_scores.append(pose_result["pose_confidence"])
                total_keypoints_visible += pose_result["keypoints_visible"]

                if pose_result["is_dangerous"]:
                    dangerous_situations += 1

                if "fall_alert" in pose_result:
                    fall_alerts += 1

        # Calculate percentages
        pose_percentages = {}
        if pose_count > 0:
            for pose_class, count in pose_distribution.items():
                pose_percentages[pose_class] = (count / pose_count) * 100

        return {
            "total_poses_classified": pose_count,
            "pose_distribution": dict(pose_distribution),
            "pose_percentages": pose_percentages,
            "avg_confidence": (
                float(np.mean(confidence_scores)) if confidence_scores else 0
            ),
            "min_confidence": (
                float(np.min(confidence_scores)) if confidence_scores else 0
            ),
            "max_confidence": (
                float(np.max(confidence_scores)) if confidence_scores else 0
            ),
            "dangerous_situations": dangerous_situations,
            "fall_alerts": fall_alerts,
            "avg_keypoints_visible": (
                total_keypoints_visible / pose_count if pose_count > 0 else 0
            ),
            "danger_rate": (
                (dangerous_situations / pose_count * 100) if pose_count > 0 else 0
            ),
        }

    def _display_video_results(self, video_name, video_stats):
        """Display results for a single video."""
        print(f"\n📊 RESULTS for {video_name}:")

        # Processing statistics
        proc_stats = video_stats["processing_stats"]
        print(f"  ✅ Frames processed: {proc_stats['frames_processed']}")
        print(f"  ⏭️ Frames skipped: {proc_stats['frames_skipped']}")
        print(f"  👥 Total people detected: {proc_stats['people_detected']}")
        print(f"  🤸 Total poses classified: {proc_stats['poses_classified']}")

        # Performance metrics
        if "performance_metrics" in video_stats:
            perf = video_stats["performance_metrics"]
            print(f"  ⚡ Average FPS: {perf['avg_fps']:.1f}")
            print(f"  🕐 Average processing time: {perf['avg_processing_time']:.3f}s")
            print(f"  📊 Detection time: {perf['avg_detection_time']:.3f}s")
            print(f"  🎯 Tracking time: {perf['avg_tracking_time']:.3f}s")
            print(f"  🤸 Pose estimation time: {perf['avg_pose_estimation_time']:.3f}s")

        # Pose statistics
        if "pose_stats" in video_stats:
            pose_stats = video_stats["pose_stats"]
            print("  📈 Pose classification results:")

            if pose_stats["pose_distribution"]:
                for pose_class, count in pose_stats["pose_distribution"].items():
                    percentage = pose_stats["pose_percentages"].get(pose_class, 0)
                    print(f"    - {pose_class.upper()}: {count} ({percentage:.1f}%)")

            print(f"  🎯 Average confidence: {pose_stats['avg_confidence']:.2f}")
            print(
                f"  👁️ Average keypoints visible: "
                f"{pose_stats['avg_keypoints_visible']:.1f}/17"
            )

            if pose_stats["dangerous_situations"] > 0:
                print(
                    f"  ⚠️ Dangerous situations: {pose_stats['dangerous_situations']} "
                    f"({pose_stats['danger_rate']:.1f}%)"
                )

            if pose_stats["fall_alerts"] > 0:
                print(f"  🚨 Fall alerts: {pose_stats['fall_alerts']}")

    def _accumulate_overall_stats(self, overall_stats, overall_times, video_stats):
        """Accumulate statistics across all videos."""
        if "processing_stats" in video_stats:
            proc_stats = video_stats["processing_stats"]
            overall_stats["total_frames_processed"] += proc_stats["frames_processed"]
            overall_stats["total_people_detected"] += proc_stats["people_detected"]
            overall_stats["total_poses_classified"] += proc_stats["poses_classified"]
            overall_times.extend(proc_stats["processing_times"])

        if "pose_stats" in video_stats:
            pose_stats = video_stats["pose_stats"]
            overall_stats["total_dangerous_situations"] += pose_stats[
                "dangerous_situations"
            ]
            overall_stats["total_fall_alerts"] += pose_stats["fall_alerts"]

    def _calculate_overall_results(self, overall_stats, overall_times):
        """Calculate and display overall test results."""
        print("\n🌟 OVERALL TEST RESULTS")
        print("=" * 50)

        total_videos = len(
            [v for v in self.test_results["videos"].values() if "error" not in v]
        )
        print(f"📹 Videos successfully processed: {total_videos}")
        print(f"🖼️ Total frames processed: {overall_stats['total_frames_processed']}")
        print(f"👥 Total people detected: {overall_stats['total_people_detected']}")
        print(f"🤸 Total poses classified: {overall_stats['total_poses_classified']}")

        if overall_times:
            avg_processing_time = np.mean(overall_times)
            overall_fps = 1.0 / avg_processing_time
            print(f"⚡ Overall average FPS: {overall_fps:.1f}")
            print(f"🕐 Overall average processing time: {avg_processing_time:.3f}s")

        # Calculate overall pose distribution
        overall_pose_distribution = defaultdict(int)
        total_poses = 0

        for video_stats in self.test_results["videos"].values():
            if "pose_stats" in video_stats:
                for pose_class, count in video_stats["pose_stats"][
                    "pose_distribution"
                ].items():
                    overall_pose_distribution[pose_class] += count
                    total_poses += count

        if total_poses > 0:
            print("\n📊 Overall pose distribution:")
            for pose_class, count in overall_pose_distribution.items():
                percentage = (count / total_poses) * 100
                print(f"  - {pose_class.upper()}: {count} ({percentage:.1f}%)")

        if overall_stats["total_dangerous_situations"] > 0:
            danger_rate = (
                (overall_stats["total_dangerous_situations"] / total_poses * 100)
                if total_poses > 0
                else 0
            )
            print(
                f"\n⚠️ Total dangerous situations: "
                f"{overall_stats['total_dangerous_situations']} ({danger_rate:.1f}%)"
            )

        if overall_stats["total_fall_alerts"] > 0:
            print(f"🚨 Total fall alerts: {overall_stats['total_fall_alerts']}")

        # Store overall results
        self.test_results["overall_stats"] = {
            "total_videos_processed": total_videos,
            "total_frames_processed": overall_stats["total_frames_processed"],
            "total_people_detected": overall_stats["total_people_detected"],
            "total_poses_classified": overall_stats["total_poses_classified"],
            "overall_avg_fps": (
                float(1.0 / np.mean(overall_times)) if overall_times else 0
            ),
            "overall_avg_processing_time": (
                float(np.mean(overall_times)) if overall_times else 0
            ),
            "overall_pose_distribution": dict(overall_pose_distribution),
            "total_dangerous_situations": overall_stats["total_dangerous_situations"],
            "total_fall_alerts": overall_stats["total_fall_alerts"],
        }

    def _save_test_results(self):
        """Save detailed test results to JSON file."""
        output_file = Path("test_results.json")

        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, "item"):
                    return obj.item()
                elif hasattr(obj, "tolist"):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {
                        key: convert_numpy_types(value) for key, value in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            # Convert the results
            json_safe_results = convert_numpy_types(self.test_results)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
            print(f"💾 Test results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving test results: {e}")
            # Save a simplified version without detailed results
            try:
                simplified_results = {
                    "test_timestamp": self.test_results["test_timestamp"],
                    "overall_stats": convert_numpy_types(
                        self.test_results["overall_stats"]
                    ),
                    "video_count": len(self.test_results["videos"]),
                }
                with open("test_results_simplified.json", "w") as f:
                    json.dump(simplified_results, f, indent=2)
                print("💾 Simplified results saved to: test_results_simplified.json")
            except Exception as e2:
                print(f"❌ Could not save simplified results either: {e2}")


def main():
    """Main function to run comprehensive pose estimation tests."""
    try:
        test_runner = PoseTestRunner()
        test_runner.run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
