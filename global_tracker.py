import time
import numpy as np
from collections import deque
import threading
from scipy.spatial.distance import cdist

# --- Constants ---
MAX_GLOBAL_TRACKS = 1000
GLOBAL_TRACK_TIMEOUT = 30.0  # seconds a global track survives without updates
SAME_CAM_TIMEOUT = 2.0       # seconds before we consider a track "lost" on a camera
REID_MATCH_TH = 0.5          # Threshold for ArcFace (0.5 is usually good for cosine dist)
SPATIAL_TIME_WINDOW = 10.0   # seconds window to search for re-entry in other cameras

class GlobalTrackManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.next_global_id = 1
        
        # Main storage for global tracks
        # { gid: {
        #     'last_seen': float,
        #     'avg_feature': np.array(512), 
        #     'current_cam': str,
        #     'history': list
        #   }
        # }
        self.tracks = {} 
        
        # Fast lookup for active local tracks to global IDs
        # { (cam_name, local_tid): global_id }
        self.local_to_global = {}

    def _get_avg_feature(self, features):
        if not features: return None
        feat_mat = np.array(features)
        avg = np.mean(feat_mat, axis=0)
        return avg / (np.linalg.norm(avg) + 1e-12)

    def _find_matching_global_id_vectorized(self, feature, cam_name, timestamp):
        """
        Vectorized search for matching global track.
        """
        if not self.tracks:
            return None, 1.0

        # 1. candidate filtering
        candidate_gids = []
        candidate_feats = []
        
        for gid, info in self.tracks.items():
            # Skip tracks active on SAME camera (unless lost for > SAME_CAM_TIMEOUT)
            if info['current_cam'] == cam_name and (timestamp - info['last_seen'] < SAME_CAM_TIMEOUT):
                continue
            
            # Skip tracks active too long ago (temporal constraint)
            if (timestamp - info['last_seen']) > SPATIAL_TIME_WINDOW:
                continue
                
            candidate_gids.append(gid)
            candidate_feats.append(info['avg_feature'])
            
        if not candidate_gids:
            return None, 1.0
            
        # 2. Vectorized Cosine Distance
        # feature shape: (512,). candidate_feats: (N, 512)
        # cdist expects 2D arrays. 
        query = feature.reshape(1, -1)
        gallery = np.array(candidate_feats)
        
        dists = cdist(query, gallery, metric='cosine')[0] # shape (N,)
        
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        if min_dist < REID_MATCH_TH:
            return candidate_gids[min_idx], min_dist
            
        return None, min_dist

    def update_track(self, cam_name, local_tid, feature, bbox, timestamp=None):
        if timestamp is None: timestamp = time.time()
        
        with self.lock:
            key = (cam_name, local_tid)
            
            # 1. Check if this local track is already assigned
            if key in self.local_to_global:
                gid = self.local_to_global[key]
                self._update_existing(gid, feature, cam_name, timestamp, bbox)
                return gid
            
            # 2. It's a new local track. Try to match.
            gid, dist = self._find_matching_global_id_vectorized(feature, cam_name, timestamp)
            
            if gid is not None:
                # Match found! Link local -> global
                self.local_to_global[key] = gid
                self._update_existing(gid, feature, cam_name, timestamp, bbox)
                # print(f"[Global] Linked {cam_name}:{local_tid} -> Global ID {gid} (dist {dist:.2f})")
                return gid
            
            # 3. No match found. Create new global track.
            gid = self.next_global_id
            self.next_global_id += 1
            self.local_to_global[key] = gid
            
            self.tracks[gid] = {
                'created_at': timestamp,
                'last_seen': timestamp,
                'features': deque([feature], maxlen=10),
                'avg_feature': feature, 
                'current_cam': cam_name,
                'history': []
            }
            # print(f"[Global] New Global ID {gid} started at {cam_name}:{local_tid}")
            return gid

    def _update_existing(self, gid, feature, cam_name, timestamp, bbox):
        if gid not in self.tracks: return # Should not happen
        t = self.tracks[gid]
        
        # if t['current_cam'] != cam_name:
        #      print(f"[TRANSITION] ID {gid} transitioned from {t['current_cam']} to {cam_name}")
        
        t['last_seen'] = timestamp
        t['current_cam'] = cam_name
        t['features'].append(feature)
        
        # Running average update for speed? Or simpler full re-avg
        # Full re-avg is fast enough for 512 dims and len=10
        t['avg_feature'] = self._get_avg_feature(t['features'])

    def get_global_id(self, cam_name, local_tid):
        with self.lock:
            return self.local_to_global.get((cam_name, local_tid), None)
            
# Create a singleton instance to be used by main app
global_tracker = GlobalTrackManager()

