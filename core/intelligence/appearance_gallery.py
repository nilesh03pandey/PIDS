import numpy as np
from collections import deque
from scipy.spatial.distance import cosine

class TrackGallery:
    """
    Maintains a gallery of embeddings for a specific track.
    Uses Exponential Moving Average (EMA) to keep a 'current' representation 
    smoothly updated, but also keeps a history of distinct views (Front, Side).
    """
    def __init__(self, track_id, max_history=5, ema_alpha=0.9):
        self.track_id = track_id
        self.history = deque(maxlen=max_history)
        self.ema_embedding = None
        self.alpha = ema_alpha
        
    def update(self, embedding):
        """
        Update the gallery with a new embedding.
        """
        if embedding is None: return
        
        # Normalize
        emb = embedding / (np.linalg.norm(embedding) + 1e-12)
        
        if self.ema_embedding is None:
            self.ema_embedding = emb
        else:
            # Update EMA
            # If the new embedding is very different (e.g. outlier), maybe don't update?
            # Drift protection: only update if similarity is reasonable (>0.4)
            dist = cosine(self.ema_embedding, emb)
            if dist < 0.6: # Moderate similarity allowed
                self.ema_embedding = self.alpha * self.ema_embedding + (1 - self.alpha) * emb
                self.ema_embedding /= (np.linalg.norm(self.ema_embedding) + 1e-12)
            
        # Add to history if distinct enough from last entry
        if not self.history or cosine(self.history[-1], emb) > 0.1:
            self.history.append(emb)
            
    def get_best_embedding(self):
        """
        Returns the EMA embedding as the representative vector.
        """
        return self.ema_embedding if self.ema_embedding is not None else (self.history[-1] if self.history else None)

class GalleryManager:
    """
    Manages galleries for all active tracks.
    """
    def __init__(self):
        self.galleries = {} # track_id -> TrackGallery
        
    def update(self, track_id, embedding):
        if track_id not in self.galleries:
            self.galleries[track_id] = TrackGallery(track_id)
        self.galleries[track_id].update(embedding)
        
    def get_embedding(self, track_id):
        if track_id in self.galleries:
            return self.galleries[track_id].get_best_embedding()
        return None
        
    def clean(self, active_track_ids):
        """Remove galleries for tracks that are no longer active."""
        # Optional: keep them for some time?
        current_ids = set(self.galleries.keys())
        active_set = set(active_track_ids)
        for tid in current_ids - active_set:
            del self.galleries[tid]
