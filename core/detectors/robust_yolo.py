import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

class RobustDetector:
    def __init__(self, weights="yolov11n.pt", device=None, base_conf=0.4, use_temporal_smoothing=True):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RobustDetector] Loading YOLO from {weights} on {self.device}...")
        self.model = YOLO(weights)
        self.model.to(self.device)
        
        # Robustness parameters
        self.base_conf = base_conf
        self.use_temporal_smoothing = use_temporal_smoothing
        
        # Temporal buffer: list of (centroid_x, centroid_y) for last N frames
        # Used to slightly boost confidence of consistent detections or suppress ghosts
        self.history = deque(maxlen=5) 
        
    def adapt_threshold(self, num_detections):
        """
        Dynamically adjust confidence threshold based on scene density.
        If crowded, raise threshold to reduce False Positives.
        If sparse, lower threshold to catch missed detections.
        """
        if num_detections > 20:
            return min(self.base_conf + 0.15, 0.85)
        elif num_detections > 10:
            return min(self.base_conf + 0.05, 0.7)
        return self.base_conf

    def detect(self, frame, img_size=320, classes=[0]):
        """
        Run detection with robustness enhancements.
        Returns list of [x1, y1, w, h, score, class_id]
        """
        # 1. Inference
        results = self.model(frame, verbose=False, imgsz=img_size, classes=classes)
        
        raw_dets = []
        if not results:
            return []
            
        # Extract raw boxes first
        for r in results[0].boxes:
            box = r.xyxy[0].cpu().numpy() # x1, y1, x2, y2
            conf = float(r.conf[0])
            cls = int(r.cls[0])
            raw_dets.append((box, conf, cls))
            
        # 2. Adaptive Thresholding
        # current_thresh = self.adapt_threshold(len(raw_dets)) 
        # We replace simple density-based threshold with Distance-Aware Thresholding
        
        final_dets = []
        current_centroids = []
        
        for (box, conf, cls) in raw_dets:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            area = w * h
            
            # Distance-Aware Confidence
            # Small objects (far away) -> Lower threshold allowed
            # Large objects (close) -> Higher threshold required to reduce FPs
            if area < 3000: # Very small / Far
                required_conf = 0.25
            elif area < 10000: # Mid range
                required_conf = 0.35
            else: # Close range
                required_conf = 0.50
                
            # Filter nonsensical boxes
            if w < 5 or h < 10: 
                continue
                
            # Temporal Consistency Check
            cx, cy = x1 + w/2, y1 + h/2
            matched_history = False
            
            if self.use_temporal_smoothing and self.history:
                for past_frame_centroids in self.history:
                    for (pcx, pcy) in past_frame_centroids:
                        dist = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                        if dist < 50: 
                            matched_history = True
                            break
                    if matched_history: break
            
            # Boost confidence if temporally consistent
            effective_conf = conf * 1.2 if matched_history else conf
            
            if effective_conf >= required_conf:
                final_dets.append(([x1, y1, w, h], conf, 'person')) 
                current_centroids.append((cx, cy))
                
        # Update history
        if self.use_temporal_smoothing:
            self.history.append(current_centroids)
            
        return final_dets
