import threading
import time
import queue
import cv2
import numpy as np
from collections import deque, defaultdict

class FrameGrabber:
    """
    Threaded frame capture to ensure the main loop always gets the latest frame.
    Drops stale frames if processing is slower than capture.
    """
    def __init__(self, source, width=640, height=480):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        # Set resolution if it's a webcam
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
        self.q = queue.Queue(maxsize=2)
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            
            # Keep only the latest frame
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        if not self.running and self.q.empty():
            return None
        try:
            return self.q.get(timeout=1.0) # Wait up to 1s for a frame
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

class EmbeddingCache:
    """
    LRU Cache for Re-ID embeddings.
    Key: (camera_name, track_id)
    Value: (embedding, timestamp)
    
    Prevents re-calculating embeddings for the same track detection in every frame.
    """
    def __init__(self, max_size=200, ttl=2.0):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl # Time to live in seconds (or frames? let's do seconds)

    def get(self, key):
        if key in self.cache:
            emb, ts = self.cache[key]
            if time.time() - ts < self.ttl:
                return emb
            else:
                del self.cache[key]
        return None

    def put(self, key, embedding):
        if len(self.cache) >= self.max_size:
            # Simple eviction: remove oldest (by insertion order, standard dict is ordered in Py3.7+)
            # Ideally use OrderedDict or LRU lib, but dict popitem(last=False) works too
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = (embedding, time.time())

class BenchmarkStats:
    """
    Tracks FPS and latency of pipeline stages.
    """
    def __init__(self):
        self.stage_times = defaultdict(lambda: deque(maxlen=100))
        self.id_switches = 0
        self.start_time = None

    def start_frame(self):
        self.start_time = time.perf_counter()
        return self.start_time

    def record_stage(self, stage_name, duration):
        self.stage_times[stage_name].append(duration)

    def end_frame(self):
        total = time.perf_counter() - self.start_time
        self.stage_times["total"].append(total)

    def get_stats(self):
        stats = {}
        for stage, times in self.stage_times.items():
            if times:
                stats[f"avg_{stage}_ms"] = (sum(times) / len(times)) * 1000
        calculate_fps = 1000.0 / stats["avg_total_ms"] if "avg_total_ms" in stats and stats["avg_total_ms"] > 0 else 0
        stats["fps"] = calculate_fps
        return stats
