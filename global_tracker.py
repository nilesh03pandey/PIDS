import time
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict, deque
import threading

# --- Constants ---
MAX_GLOBAL_TRACKS = 1000
GLOBAL_TRACK_TIMEOUT = 30.0  # seconds a global track survives without updates
SAME_CAM_TIMEOUT = 2.0       # seconds before we consider a track "lost" on a camera
REID_MATCH_TH = 0.4          # Threshold for matching features across cameras
SPATIAL_TIME_WINDOW = 10.0   # seconds window to search for re-entry in other cameras

class GlobalTrackManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.next_global_id = 1
        
        # map: global_id -> {
        #   'last_seen': timestamp,
        #   'features': deque of features (to keep a history),
        #   'avg_feature': np.array,
        #   'current_cam': cam_name or None (if lost),
        #   'history': list of (cam_name, timestamp, box)
        # }
        self.tracks = {} 
        
        # map: (cam_name, local_tid) -> global_id
        self.local_to_global = {}

    def _get_avg_feature(self, features):
        if not features: return None
        # Simple average and normalize
        feat_mat = np.array(features)
        avg = np.mean(feat_mat, axis=0)
        return avg / (np.linalg.norm(avg) + 1e-12)

    def _find_matching_global_id(self, feature, cam_name, timestamp):
        """
        Search for a matching global track that is NOT currently active on this camera
        and fits spatiotemporal constraints.
        """
        best_gid = None
        best_dist = REID_MATCH_TH

        for gid, info in self.tracks.items():
            # Constraint 1: Don't match with a track that is currently active on THIS camera 
            # (unless the local tracker broke and created a new ID, but we assume DeepSort handles short breaks).
            # We principally look for tracks that were last seen on *other* cameras or are "lost".
            
            # Simple check: If recently seen on this camera, skip? 
            # Actually, if track ID changed locally, we might want to merge. 
            # But for cross-camera, we care about *other* cameras.
            
            # Temporal Constraint: MATCH must be within reasonable time
            time_diff = timestamp - info['last_seen']
            if abs(time_diff) > SPATIAL_TIME_WINDOW:
                continue

            # Feature Match
            dist = cosine(feature, info['avg_feature'])
            if dist < best_dist:
                best_dist = dist
                best_gid = gid
        
        return best_gid

    def update_track(self, cam_name, local_tid, feature, bbox, timestamp=None):
        if timestamp is None: timestamp = time.time()
        
        with self.lock:
            # 0. Clean up old tracks periodically (simple version: every 100 updates or check at start)
            # self._cleanup_stale_tracks(timestamp)

            key = (cam_name, local_tid)
            
            # 1. Check if this local track is already assigned
            if key in self.local_to_global:
                gid = self.local_to_global[key]
                # Update existing global track
                self._update_existing_global_track(gid, feature, cam_name, timestamp, bbox)
                return gid
            
            # 2. It's a new local track. Try to match with existing global tracks.
            gid = self._find_matching_global_id(feature, cam_name, timestamp)
            
            if gid is not None:
                # Match found! Link local -> global
                self.local_to_global[key] = gid
                self._update_existing_global_track(gid, feature, cam_name, timestamp, bbox)
                print(f"[GlobalTracker] Linked {cam_name}:{local_tid} -> Global ID {gid}")
                return gid
            
            # 3. No match found. Create new global track.
            gid = self.next_global_id
            self.next_global_id += 1
            self.local_to_global[key] = gid
            
            self.tracks[gid] = {
                'created_at': timestamp,
                'last_seen': timestamp,
                'features': deque([feature], maxlen=10),
                'avg_feature': feature, # already normalized? assume yes
                'current_cam': cam_name,
                'history': [] # can append (cam_name, timestamp)
            }
            print(f"[GlobalTracker] New Global ID {gid} started at {cam_name}:{local_tid}")
            return gid

    def _update_existing_global_track(self, gid, feature, cam_name, timestamp, bbox):
        t = self.tracks[gid]
        # Check for camera transition
        if t['current_cam'] is not None and t['current_cam'] != cam_name:
             print(f"[TRANSITION] ID {gid} transitioned from {t['current_cam']} to {cam_name}")
        
        t['last_seen'] = timestamp
        t['current_cam'] = cam_name
        
        # Update features
        t['features'].append(feature)
        # Re-calc average (can optimize to running average)
        t['avg_feature'] = self._get_avg_feature(t['features'])

    def get_global_id(self, cam_name, local_tid):
        with self.lock:
            return self.local_to_global.get((cam_name, local_tid), None)
            
# Create a singleton instance to be used by main app
global_tracker = GlobalTrackManager()
