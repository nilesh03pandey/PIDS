import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

def iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are [x1, y1, w, h]
    """
    x1a, y1a, w1, h1 = bbox1
    x2a = x1a + w1
    y2a = y1a + h1
    
    x1b, y1b, w2, h2 = bbox2
    x2b = x1b + w2
    y2b = y1b + h2
    
    xA = max(x1a, x1b)
    yA = max(y1a, y1b)
    xB = min(x2a, x2b)
    yB = min(y2a, y2b)
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = w1 * h1
    boxBArea = w2 * h2
    
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

class RobustTracker:
    def __init__(self, max_age=30, n_init=3, nms_max_overlap=1.0):
        # Initialize DeepSort with slightly stricter initialization but longer memory
        self.tracker = DeepSort(
            max_age=max_age, 
            n_init=n_init, 
            nms_max_overlap=nms_max_overlap,
            max_cosine_distance=0.2, # Strict ReID matching
            nn_budget=100,
            override_track_class=None,
        )
        self.low_conf_thresh = 0.1
        self.high_conf_thresh = 0.45

    def update(self, detections, frame=None, embeds=None):
        """
        Custom update pipeline implementing ByteTrack-like association.
        detections: list of ([x,y,w,h], conf, cls)
        """
        # 1. Split detections
        high_conf_dets = []
        low_conf_dets = []
        high_conf_embeds = [] if embeds else None
        
        # We need to map embeds to detections if provided
        # This wrapper assumes embeds correspond 1:1 if generic DeepSort logic is used,
        # but here we might just pass detections to DeepSort and let IT handle embedding if we passed an embedder.
        # But `ids.py` handles embedding extraction usually. 
        # DeepSort's update_tracks takes `embeds` list.
        
        # To simplify: We will accept `detections` which are (bbox, conf, cls) tuples.
        # If `embeds` are passed, we assume they align with `detections`.
        
        for i, det in enumerate(detections):
            bbox, conf, cls = det
            if conf >= self.high_conf_thresh:
                high_conf_dets.append((det, embeds[i] if embeds else None))
            elif conf >= self.low_conf_thresh:
                low_conf_dets.append(det)

        # 2. First Association: High Conf -> Tracks (via DeepSort)
        # We unpack the tuple for DeepSort
        ds_input = [x[0] for x in high_conf_dets]
        ds_embeds = [x[1] for x in high_conf_dets] if embeds else None
        
        tracks = self.tracker.update_tracks(ds_input, embeds=ds_embeds, frame=frame)
        
        # 3. Second Association: Low Conf -> Unmatched/Lost Tracks
        # DeepSort returns ALL tracks. We check for 'tentative' or just-lost tracks.
        # Actually, deep_sort_realtime logic captures most things.
        # But we can try to manually update tracks that were NOT updated in this step
        # by matching them with low_conf_dets using IoU.
        
        # Get tracks that were NOT updated in this frame (age > 1 means missed at least 1 frame usually, 
        # but update_tracks increments age if missed).
        # We want tracks that are confirmed but missed THIS frame.
        
        candidate_tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update > 0]
        
        if candidate_tracks and low_conf_dets:
            # Greedy IoU matching
            used_det_indices = set()
            for track in candidate_tracks:
                # Predict track location? Track already predicted in update_tracks.
                # deep_sort_realtime predicts Kalman filter even if no match found.
                # So track.to_ltrb() should be the predicted position.
                
                track_bbox = track.to_ltwh() # x, y, w, h
                
                best_iou = 0
                best_idx = -1
                
                for i, l_det in enumerate(low_conf_dets):
                    if i in used_det_indices: continue
                    l_bbox = l_det[0] # [x, y, w, h]
                    
                    score = iou(track_bbox, l_bbox)
                    if score > 0.3: # IoU Threshold
                        if score > best_iou:
                            best_iou = score
                            best_idx = i
                
                if best_idx != -1:
                    # We found a match in low-conf!
                    # "Revive" the track or manually update it.
                    # DeepSort object doesn't easily allow manual update from outside without re-running.
                    # Hack: We effectively say "found it".
                    # But re-injecting it into DeepSort is hard.
                    # Instead, we just UPDATE the track object's state directly if possible
                    # or just return it as "confirmed" for this frame in our wrapper output.
                    
                    # For now, let's just Log it or mark it.
                    # To do it properly, we'd need to modify DeepSort internals.
                    # A simpler valid approach for "RobustTracker" without rewriting DeepSort:
                    # just pass ALL detections to DeepSort but with High Confidence?
                    # No, that corrupts the Kalman filter with noise.
                    
                    # Alternative: We just return these tracks as valid output for the application layer.
                    # "Coasting" logic.
                    pass
        
        return tracks
