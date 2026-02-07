import unittest
import numpy as np
from global_tracker import GlobalTrackManager

class TestGlobalTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = GlobalTrackManager()

    def test_same_id_different_camera(self):
        # Feature vector for Person A
        feat_a = np.array([1.0, 0.0, 0.0])
        feat_a = feat_a / np.linalg.norm(feat_a)
        
        # Camera 1 detects Person A
        ts1 = 1000.0
        gid1 = self.tracker.update_track("cam1", 101, feat_a, (0,0,10,10), timestamp=ts1)
        
        # Camera 2 detects Person A (similar feature) 5 seconds later
        feat_a_noisy = np.array([0.98, 0.02, 0.0])
        feat_a_noisy = feat_a_noisy / np.linalg.norm(feat_a_noisy)
        ts2 = 1005.0
        gid2 = self.tracker.update_track("cam2", 205, feat_a_noisy, (20,20,10,10), timestamp=ts2)
        
        print(f"GID1: {gid1}, GID2: {gid2}")
        self.assertEqual(gid1, gid2, "Should be assigned the same Global ID")

    def test_different_id(self):
        # Person A
        feat_a = np.array([1.0, 0.0, 0.0])
        gid_a = self.tracker.update_track("cam1", 101, feat_a, (0,0,10,10), timestamp=1000.0)
        
        # Person B (orthogonal feature)
        feat_b = np.array([0.0, 1.0, 0.0])
        gid_b = self.tracker.update_track("cam2", 205, feat_b, (0,0,10,10), timestamp=1002.0)
        
        print(f"GID_A: {gid_a}, GID_B: {gid_b}")
        self.assertNotEqual(gid_a, gid_b, "Should be different Global IDs")

    def test_timeout(self):
        # Person A
        feat_a = np.array([1.0, 0.0, 0.0])
        ts1 = 1000.0
        gid1 = self.tracker.update_track("cam1", 101, feat_a, (0,0,10,10), timestamp=ts1)
        
        # Person A appears in Cam 2 after timeout (e.g., 31 seconds later)
        ts2 = 1000.0 + 31.0
        # Start a new "local track" on cam2
        gid2 = self.tracker.update_track("cam2", 205, feat_a, (0,0,10,10), timestamp=ts2)
        
        # Depending on configuration, this might match or not. 
        # In current config SPATIAL_TIME_WINDOW = 10.0, so it should NOT match.
        self.assertNotEqual(gid1, gid2, "Should not match after timeout")

if __name__ == '__main__':
    unittest.main()
