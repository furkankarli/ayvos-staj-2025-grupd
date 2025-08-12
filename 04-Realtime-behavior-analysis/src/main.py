#!/usr/bin/env python3
"""
Main execution script for realtime behavior analysis system.
Integrates all modules: detection, pose estimation, analysis, and segmentation.
"""

import cv2
import numpy as np
from pathlib import Path

from config import Config
from detection import HumanDetector, HumanTracker
from pose import PoseEstimator
from analysis import BehaviorAnalyzer
from segmentation import Segmentator
from utils import VideoInput

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
        while True:
            frame = video.read_frame()
            if frame is None:
                break
                
            frame_count += 1
            print(f"Processing frame {frame_count}")
            
            # 1. Human Detection
            detections = detector.detect(frame)
            print(f"  - Detected {len(detections)} humans")
            
            # 2. Human Tracking
            tracks = tracker.update(detections, frame)
            print(f"  - Tracking {len(tracks)} humans")
            
            # 3. Draw results
            frame_with_detections = detector.draw_detections(frame, detections, color=(255, 0, 0))
            frame_with_tracks = tracker.draw_tracks(frame_with_detections, tracks, color=(0, 255, 0))
            
            # 4. Display frame
            cv2.imshow('Realtime Behavior Analysis', frame_with_tracks)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Limit processing for demo (remove this in production)
            if frame_count > 100:  # Process only first 100 frames for demo
                break
        
        video.release()
        cv2.destroyAllWindows()
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()