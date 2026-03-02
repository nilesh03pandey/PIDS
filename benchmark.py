import time
import cv2
import numpy as np
import threading
from pipeline import BenchmarkStats
# v2.0 Modules
from core.detectors.robust_yolo import RobustDetector
from core.trackers.robust_tracker import RobustTracker
from core.intelligence.quality_filter import QualityFilter
import GPUtil
import os
import torch

def get_gpu_usage():
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].load * 100, gpus[0].memoryUsed
    except Exception:
        pass
    return 0, 0

def benchmark_pipeline(video_path, max_frames=500):
    print(f"Starting benchmark v2.0 (Sentinel) on {video_path}...")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize components
    try:
        detector = RobustDetector(device=DEVICE)
        tracker = RobustTracker(max_age=30, n_init=3)
        quality_filter = QualityFilter()
        stats = BenchmarkStats()
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    gpu_loads = []
    
    print("Warmup...")
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            detector.detect(frame)
            
    print("Benchmarking...")
    start_total = time.time()
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        stats.start_frame()
        
        # 1. Detect (Robust)
        t0 = time.perf_counter()
        detections = detector.detect(frame, img_size=320)
        stats.record_stage("detect", time.perf_counter() - t0)
        
        # 2. Track (Robust/Byte)
        t1 = time.perf_counter()
        tracks = tracker.update(detections, frame=frame)
        stats.record_stage("track", time.perf_counter() - t1)
        
        # 3. Intelligent Re-ID Simulation
        t2 = time.perf_counter()
        for track in tracks:
            if not track.is_confirmed(): continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            # Quality Check overhead
            is_good, _ = quality_filter.check(frame, (x1, y1, x2-x1, y2-y1))
            if is_good:
                pass # Simulating decision to run Re-ID (which is the expensive part)
                # In real app, we'd run extractor here.
                # For benchmark, we care about the overhead of the new logic.
        stats.record_stage("reid_logic", time.perf_counter() - t2)
        
        stats.end_frame()
        
        if frame_count % 10 == 0:
            load, mem = get_gpu_usage()
            gpu_loads.append(load)
            fps = stats.get_stats().get('fps', 0)
            print(f"Frame {frame_count}: FPS {fps:.2f}, GPU {load:.1f}%")

    end_total = time.time()
    duration = end_total - start_total
    
    final_stats = stats.get_stats()
    avg_gpu = sum(gpu_loads) / len(gpu_loads) if gpu_loads else 0
    
    print("\n" + "="*40)
    print("SENTINEL v2.0 BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Frames: {frame_count}")
    print(f"Total Time:   {duration:.2f} s")
    print(f"Average FPS:  {frame_count / duration:.2f}")
    print(f"Pipeline FPS: {final_stats.get('fps', 0):.2f} (excluding read time)")
    print("-" * 20)
    print(f"Avg Detect ms: {final_stats.get('avg_detect_ms', 0):.2f}")
    print(f"Avg Track ms:  {final_stats.get('avg_track_ms', 0):.2f}")
    print(f"Avg Logic ms:  {final_stats.get('avg_reid_logic_ms', 0):.2f}") # Logic overhead
    print(f"Avg GPU Util:  {avg_gpu:.1f}%")
    print("="*40)

if __name__ == "__main__":
    # Create a dummy video if needed, or use existing
    video_path = "vid3.mp4" 
    if not os.path.exists(str(video_path)):
        print(f"Video {video_path} not found. Using webcam (0) for short test.")
        video_path = 0
        
    benchmark_pipeline(video_path, max_frames=200)
