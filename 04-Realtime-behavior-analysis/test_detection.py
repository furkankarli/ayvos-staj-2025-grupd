#!/usr/bin/env python3
"""
Test script for detection and tracking modules.
This script tests the basic functionality without requiring video input.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from detection import HumanDetector, HumanTracker

def create_test_frame():
    """Create a test frame with some shapes to simulate people."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw some rectangles to simulate people
    cv2.rectangle(frame, (100, 100), (200, 300), (255, 255, 255), -1)  # Person 1
    cv2.rectangle(frame, (300, 150), (400, 350), (255, 255, 255), -1)  # Person 2
    cv2.rectangle(frame, (500, 200), (600, 400), (255, 255, 255), -1)  # Person 3
    
    return frame

def test_detection():
    """Test the detection module."""
    print("Testing Detection Module...")
    
    try:
        config = Config()
        detector = HumanDetector(config)
        
        # Create test frame
        test_frame = create_test_frame()
        
        # Test detection (this will work if YOLO model is available)
        print("  - Testing detection on synthetic frame...")
        detections = detector.detect(test_frame)
        print(f"  - Detections: {len(detections)}")
        
        # Test drawing
        print("  - Testing detection visualization...")
        frame_with_detections = detector.draw_detections(test_frame, detections)
        
        # Test stats
        stats = detector.get_detection_stats(detections)
        print(f"  - Detection stats: {stats}")
        
        print("  ✓ Detection module test completed")
        return True
        
    except Exception as e:
        print(f"  ✗ Detection module test failed: {e}")
        return False

def test_tracking():
    """Test the tracking module."""
    print("Testing Tracking Module...")
    
    try:
        config = Config()
        tracker = HumanTracker(config)
        
        # Create test detections
        test_detections = [
            {'bbox': (100, 100, 100, 200), 'conf': 0.9, 'class': 'person'},
            {'bbox': (300, 150, 100, 200), 'conf': 0.8, 'class': 'person'},
            {'bbox': (500, 200, 100, 200), 'conf': 0.7, 'class': 'person'}
        ]
        
        # Test tracking update
        print("  - Testing tracking update...")
        tracks = tracker.update(test_detections)
        print(f"  - Tracks: {len(tracks)}")
        
        # Test tracking stats
        stats = tracker.get_tracking_stats(tracks)
        print(f"  - Tracking stats: {stats}")
        
        print("  ✓ Tracking module test completed")
        return True
        
    except Exception as e:
        print(f"  ✗ Tracking module test failed: {e}")
        return False

def test_integration():
    """Test detection and tracking integration."""
    print("Testing Integration...")
    
    try:
        config = Config()
        detector = HumanDetector(config)
        tracker = HumanTracker(config)
        
        # Create test frame
        test_frame = create_test_frame()
        
        # Full pipeline test
        print("  - Running full detection -> tracking pipeline...")
        detections = detector.detect(test_frame)
        tracks = tracker.update(detections, test_frame)
        
        print(f"  - Detections: {len(detections)}, Tracks: {len(tracks)}")
        
        # Test visualization
        frame_with_detections = detector.draw_detections(test_frame, detections)
        frame_with_tracks = tracker.draw_tracks(frame_with_detections, tracks)
        
        print("  ✓ Integration test completed")
        return True
        
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Detection and Tracking Module Tests")
    print("=" * 50)
    
    # Test individual modules
    detection_ok = test_detection()
    tracking_ok = test_tracking()
    integration_ok = test_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Detection Module: {'✓ PASS' if detection_ok else '✗ FAIL'}")
    print(f"Tracking Module:  {'✓ PASS' if tracking_ok else '✗ FAIL'}")
    print(f"Integration:      {'✓ PASS' if integration_ok else '✗ FAIL'}")
    
    if all([detection_ok, tracking_ok, integration_ok]):
        print("\n🎉 All tests passed! Your detection and tracking system is ready.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    print("\nNote: If detection tests fail, make sure you have a YOLO model file")
    print("in the models/ directory (e.g., yolov8n.pt)")

if __name__ == "__main__":
    main()
