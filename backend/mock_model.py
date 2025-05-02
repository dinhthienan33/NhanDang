"""
Mock implementation of the rain removal model for testing purposes.
This allows the system to be tested without requiring the actual PyTorch model.
"""

import cv2
import numpy as np
import os

class MockRainRemovalModel:
    """Mock implementation of the rain removal model."""
    
    def __init__(self, model_path: str):
        """Initialize the mock model."""
        self.model_path = model_path
        print(f"Mock model initialized. Pretending to load from: {model_path}")
        
    def process_image(self, img: np.ndarray) -> np.ndarray:
        """
        Process an image to simulate rain removal.
        This is a simple implementation that applies basic image processing
        to simulate the effect of rain removal.
        """
        # Convert to grayscale then back to BGR to reduce saturation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Apply a slight blur to simulate droplet removal
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Enhance contrast slightly
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Blend with original for a subtle effect
        result = cv2.addWeighted(enhanced, 0.7, img, 0.3, 0)
        
        # Add a watermark to make it clear this is a mock result
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, 'MOCK RESULT - NOT REAL MODEL', 
                    (10, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        return result 