
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

tracker = DeepSort(embedder=None, embedder_gpu=False)

# Frame 1: Detections WITH embeddings
dets1 = [ ([0,0,10,10], 0.9, 'face') ]
embeds1 = [ np.random.rand(512) ]
tracks = tracker.update_tracks(dets1, embeds=embeds1)
print(f"Frame 1 tracks: {len(tracks)}")

# Frame 2: Detections WITHOUT embeddings
dets2 = [ ([1,1,10,10], 0.9, 'face') ]
try:
    tracks = tracker.update_tracks(dets2, embeds=None)
    print(f"Frame 2 tracks (No Embeds): {len(tracks)}")
    if tracks:
        print(f"Track status: {tracks[0].is_confirmed()}")
except Exception as e:
    print(f"Error without embeds: {e}")
