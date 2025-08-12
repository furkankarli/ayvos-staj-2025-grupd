"""
Video Input Module
RTSP veya video dosyasından görüntü akışı almak için temel yapı.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

class VideoInput:
    """Video input handler for various sources."""
    
    def __init__(self, source: Union[str, int]):
        """
        Initialize video input.
        
        Args:
            source: Video file path, RTSP URL, or camera index
        """
        self.source = source
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        
    def open(self) -> bool:
        """
        Open video source.
        
        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Video kaynağı açılamadı: {self.source}")
            
        # Video özelliklerini al
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return True
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read next frame from video source.
        
        Returns:
            Frame as numpy array or None if failed
        """
        if self.cap is None:
            raise RuntimeError("Video kaynağı açık değil.")
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        return frame
    
    def read_frame_resized(self, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Read and resize frame to target size.
        
        Args:
            target_size: Target (width, height)
            
        Returns:
            Resized frame or None if failed
        """
        frame = self.read_frame()
        if frame is None:
            return None
            
        return cv2.resize(frame, target_size)
    
    def get_frame_info(self) -> dict:
        """
        Get video frame information.
        
        Returns:
            Dictionary with frame information
        """
        if self.cap is None:
            return {}
            
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'current_frame': int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        }
    
    def set_frame_position(self, frame_number: int) -> bool:
        """
        Set frame position.
        
        Args:
            frame_number: Target frame number
            
        Returns:
            True if successful
        """
        if self.cap is None:
            return False
            
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def is_opened(self) -> bool:
        """
        Check if video source is opened.
        
        Returns:
            True if opened
        """
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
            self.cap = None
