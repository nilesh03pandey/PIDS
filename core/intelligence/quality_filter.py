import cv2
import numpy as np

class QualityFilter:
    def __init__(self, min_size=40, blur_thresh=100):
        self.min_size = min_size
        self.blur_thresh = blur_thresh

    def check(self, frame, bbox):
        """
        Check if a crop defined by bbox is of sufficient quality for Re-ID.
        Returns (is_good, score)
        """
        x, y, w, h = map(int, bbox)
        
        # 1. Size Check
        if w < self.min_size or h < self.min_size:
            return False, 0.0
            
        # 2. Boundary Check
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x+w), min(h_img, y+h)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, 0.0
            
        # 3. Blur Check (Laplacian Variance)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if score < self.blur_thresh:
            return False, score
            
        return True, score
