# app.py
import os
import cv2
from audit_manager import audit_logger # NEW: Import Audit Logger
import time
import torch
import threading
import numpy as np
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
from pymongo import MongoClient
import datetime
from collections import defaultdict, deque
from numpy.linalg import norm
import winsound
import tkinter as tk
from tkinter import messagebox
import re
# --- Custom Modules ---
# from audit_manager import audit_logger # REPLACED by AuditLedger
from global_tracker import global_tracker 
from pipeline import FrameGrabber, BenchmarkStats

# NEW CORE MODULES
from core.detectors.robust_yolo import RobustDetector
from core.trackers.robust_tracker import RobustTracker
from core.intelligence.quality_filter import QualityFilter
from core.intelligence.appearance_gallery import GalleryManager
from core.intelligence.behavior_engine import BehaviorEngine
from core.forensics.audit_ledger import AuditLedger

# --- Configuration ---
# YOLO_WEIGHTS = "yolov11n.pt" # Handled by RobustDetector
REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_PATH = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- NEW PERFORMANCE OPTIMIZATIONS ---
CAM_RESOLUTION_W = 640
CAM_RESOLUTION_H = 480
PROCESS_EVERY_N_FRAMES = 1

# --- Tuning Parameters ---
STRICT_TH = 0.35      
LOOSE_TH = 0.55       
RATIO_MARGIN = 0.75   
# YOLO_CONF_TH = 0.4 # Handled by RobustDetector

# re-ID dynamic update params
REID_INTERVAL = 3 # Increased frequency but filtered by QualityFilter
MIN_CONF_ASSIGN = 0.55
CONF_MARGIN = 0.18
CONSECUTIVE_UPDATES = 3
EMA_ALPHA_TRACK = 0.85

TILE_W, TILE_H = 480, 360
UNKNOWN_SAVE_DIR = "unknown_crops"
os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)

# --- Database Setup ---
client = MongoClient("mongodb://localhost:27017/")
db = client["person_reid"]
people_col = db["people"]
logs_col = db["logs"]
history_col = db["track_history"]
access_col = db["access_control"]
alerts_col = db["alerts"]

print(f"Using device: {DEVICE}")

# Initialize Components
audit_ledger = AuditLedger("secure_audit.jsonl")
detector = RobustDetector(device=DEVICE)
# feature extractor remains separate for now
extractor = FeatureExtractor(
    model_name=REID_MODEL_NAME,
    model_path=REID_MODEL_PATH,
    device=DEVICE
)

# Zone Config (Dummy for now)
ZONE_CONFIG = {
    "RedZone": [100, 100, 300, 300] # x1, y1, x2, y2
}


# (All functions from trigger_alert to dist_to_conf remain exactly the same)
# --- Alerting and Logging Functions ---

def play_aggressive_sound():
    """Play a sequence of aggressive beeps to demand attention."""
    def _beep():
        for _ in range(5):
            winsound.Beep(2000, 150) # High pitch, fast
            time.sleep(0.05)
            winsound.Beep(1500, 150)
            time.sleep(0.05)
    threading.Thread(target=_beep, daemon=True).start()

def trigger_alert(frame, cam_name, tid):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🚨 [ALERT] UNKNOWN detected on {cam_name} (track {tid}) at {ts}")
    snap_path = f"alert_snapshots/{cam_name}_{tid}_{int(time.time())}.jpg"
    os.makedirs("alert_snapshots", exist_ok=True)
    cv2.imwrite(snap_path, frame)
    
    play_aggressive_sound() # LOUD ALERT
    
    # --- Crypto Audit ---
    audit_ledger.log("ALERT_UNKNOWN", {"camera": cam_name, "track_id": tid, "snapshot": snap_path})
    # ----------------------
    
    show_popup(cam_name, tid)
    
def trigger_zone_alert(name, cam_name, frame, bbox):
    ts = datetime.datetime.now()
    os.makedirs("alert_snapshots", exist_ok=True)
    x, y, w, h = bbox
    crop = frame[y:y+h, x:x+w]
    thumb_name = f"{cam_name}_{name}_{int(time.time())}.jpg"
    thumb_path = os.path.join("alert_snapshots", thumb_name)
    cv2.imwrite(thumb_path, crop)
    alerts_col.insert_one({
        "timestamp": ts, "person_name": name, "camera_name": cam_name,
        "status": "Unauthorized Zone Entry", "thumbnail": thumb_path
    })
    print(f"🚫 [ZONE ALERT] {name} entered restricted camera: {cam_name} at {ts}")
    
    # --- Crypto Audit ---
    audit_ledger.log("ALERT_ZONE", {"person": name, "camera": cam_name, "thumbnail": thumb_path})
    # ----------------------
    
    play_aggressive_sound() # LOUD ALERT
    show_popup(cam_name, f"{name} - Unauthorized Access")

last_log_times = {}
LOG_INTERVAL = 5

def log_person_event(name, cam_name, tid, frame, bbox):
    now = time.time()
    key = (cam_name, tid, name)
    if key in last_log_times and now - last_log_times[key] < LOG_INTERVAL:
        return
    last_log_times[key] = now
    ts = datetime.datetime.now()
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    person_crop = frame[y1:y2, x1:x2]
    thumb_dir = os.path.join("static", "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_name = f"{cam_name}_{tid}_{int(time.time())}.jpg"
    thumb_path = os.path.join(thumb_dir, thumb_name)
    cv2.imwrite(thumb_path, person_crop)
    web_thumb_path = f"/static/thumbnails/{thumb_name}"
    history_col.insert_one({
        "timestamp": ts, "person_name": name, "camera_name": cam_name,
        "track_id": tid, "thumbnail": web_thumb_path
    })
    
    # --- Crypto Audit ---
    audit_ledger.log("PERSON_DETECTED", {"person": name, "camera": cam_name, "track_id": tid})
    # ----------------------
    
    print(f"📋 [DB] Logged {name} on {cam_name}, track {tid} at {ts}")
    
def check_access_permission(person_name, cam_name, frame, bbox):
    if not person_name or person_name == "UNKNOWN" or re.search(r"\s*\(\?\)$", person_name.strip()):
        return
    access_doc = access_col.find_one({"camera_name": cam_name})
    allowed_people = access_doc.get("allowed_people", []) if access_doc else ["*"]
    if allowed_people != ["*"] and person_name not in allowed_people:
        trigger_zone_alert(person_name, cam_name, frame, bbox)

def show_popup(cam_name, tid):
    def popup():
        root = tk.Tk()
        root.title("🚨 Security Alert!")
        root.geometry("300x150+100+100")
        root.attributes("-topmost", True)
        msg = tk.Label(root, text=f"Alert on camera: {cam_name}\n\nDetails: {tid}", fg="red", font=("Arial", 12), justify="center")
        msg.pack(expand=True, padx=10, pady=20)
        root.after(5000, root.destroy)
        root.mainloop()
    threading.Thread(target=popup, daemon=True).start()

def l2norm(v: np.ndarray):
    return v / (np.linalg.norm(v) + 1e-12)

def extract_feature_from_crop(frame, box, margin=0.05):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * margin), int(bh * margin)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w - 1, x2 + pad_x), min(h - 1, y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        feat_t = extractor([crop_rgb])
    feat = feat_t[0].cpu().numpy().flatten()
    return l2norm(feat)

def load_known_people():
    people = []
    for doc in people_col.find():
        feats = [l2norm(np.array(f, dtype=np.float32).flatten()) for f in doc.get("features", [])]
        if feats: people.append({"name": doc["name"], "role": doc.get("role", "Unknown"), "features": feats})
    print(f"[DB] Loaded {len(people)} known people")
    return people

def match_person(feat: np.ndarray, known_people):
    if feat is None or not known_people: return None, None, None, False
    f = l2norm(feat)
    best_person, best_dist, second_dist = None, 1.0, 1.0
    for person in known_people:
        dists = [float(cosine(f, ex)) for ex in person["features"]]
        if not dists: continue
        min_d = min(dists)
        if min_d < best_dist:
            second_dist, best_dist, best_person = best_dist, min_d, person
        elif min_d < second_dist:
            second_dist = min_d
    if best_person is None: return None, None, None, False
    ratio_ok = (best_dist / (second_dist + 1e-12)) < RATIO_MARGIN
    strong = (best_dist < STRICT_TH) and ratio_ok
    weak = (best_dist < LOOSE_TH) and ratio_ok
    if strong: return best_person["name"], best_person["role"], best_dist, True
    if weak: return best_person["name"] + " (?)", best_person["role"], best_dist, False
    return None, None, None, False

def dist_to_conf(dist):
    if dist is None: return 0.0
    if dist <= STRICT_TH: return 1.0
    if dist >= LOOSE_TH: return 0.0
    return float((LOOSE_TH - dist) / (LOOSE_TH - STRICT_TH))


# --- Main Processing Loop ---
latest_frames = {}
frames_lock = threading.Lock()
last_alert_time = {}  # ✅ FIX: initialize the dictionary globally


def process_camera(source, cam_name, known_reload_interval=30):
    global last_alert_time

    print(f"[{cam_name}] starting, source={source}")
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RESOLUTION_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION_H)

    # Initialize Modules per camera
    # RobustTracker wraps DeepSort with tuned params
    tracker = RobustTracker(max_age=50, n_init=3) 
    quality_filter = QualityFilter()
    gallery = GalleryManager()
    behavior = BehaviorEngine(zone_config=ZONE_CONFIG)
    
    track_info = {}
    track_vote = defaultdict(lambda: deque(maxlen=CONSECUTIVE_UPDATES))
    known_people = load_known_people()
    last_known_load = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_name}] stream ended")
            break
        frame_idx += 1

        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            with frames_lock:
                latest_frames[cam_name] = cv2.resize(frame, (TILE_W, TILE_H))
            time.sleep(0.01)
            continue

        if time.time() - last_known_load > known_reload_interval:
            known_people = load_known_people()
            last_known_load = time.time()

        # 1. Robust Detection
        # returns list of ([x,y,w,h], conf, cls)
        detections = detector.detect(frame, img_size=320)
        
        # 2. Robust Tracking (ByteTrack style association)
        # We pass full detections. Tracker handles high/low conf split if implemented, 
        # or just standard DeepSort if using the wrapper.
        tracks = tracker.update(detections, frame=frame)
        
        # 3. Behavior Analysis
        events = behavior.update(tracks)
        for event in events:
            # Handle Events
            if event['type'] == 'LOITERING':
                # print(f"⚠️ [LOITER] Track {event['track_id']} in {event['zone']} for {event['duration']:.1f}s")
                pass
            elif event['type'] == 'ZONE_CHANGE':
                # Check if entered a restricted zone
                new_zone = event['to']
                tid = event['track_id']
                if new_zone != "General" and new_zone != "None":
                     # Get identity
                    info = track_info.get(tid)
                    name = info["name"] if info else "UNKNOWN"
                    print(f"🚫 [ZONE ENTRY] {name} entered {new_zone}")
                    # We can use trigger_zone_alert but it needs bbox/frame.
                    # We skip full alert for now or implement it if track confirms execution.
                    pass
                
        # 4. Identity & Attributes
        for track in tracks:
            if not track.is_confirmed(): continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            tid = track.track_id
            
            info = track_info.get(tid, {
                "name": "UNKNOWN", "role": "", "conf": 0.0, 
                "last_reid_frame": -999, "last_seen": time.time()
            })
            info["last_seen"] = time.time()

            # --- INTELLIGENT RE-ID ---
            # Instead of Re-ID every N frames blindly, check Quality.
            
            # Check if we should update embedding
            should_run_reid = (frame_idx - info["last_reid_frame"]) >= REID_INTERVAL
            
            feat = None
            if should_run_reid:
                # Quality Check
                is_good, q_score = quality_filter.check(frame, (x1, y1, x2-x1, y2-y1))
                if is_good:
                    feat_raw = extract_feature_from_crop(frame, (x1, y1, x2, y2))
                    if feat_raw is not None:
                        # Update Gallery
                        gallery.update(tid, feat_raw)
                        info["last_reid_frame"] = frame_idx
                        # Get best representative embedding for matching
                        feat = gallery.get_embedding(tid)

            # Match if we have a valid embedding (current or cached)
            # If we didn't run ReID this frame, check if gallery has one
            if feat is None:
                feat = gallery.get_embedding(tid)
            
            if feat is not None:
                # Global Tracker Update
                global_id = global_tracker.update_track(cam_name, tid, feat, (x1, y1, x2, y2))
                info["global_id"] = global_id

                # Identity Matching
                name, role, dist, strong = match_person(feat, known_people)
                new_conf = dist_to_conf(dist)
                
                # Update Identity Logic (similar to before but using gallery feat)
                if info["name"] == "UNKNOWN":
                    if new_conf >= MIN_CONF_ASSIGN: 
                        info.update({"name": name, "role": role, "conf": new_conf})
                else:
                    current_name, current_conf = info["name"], info["conf"]
                    if name == current_name:
                        info["conf"] = max(current_conf, new_conf)
                    elif strong and new_conf > current_conf + CONF_MARGIN:
                        info.update({"name": name, "role": role, "conf": new_conf})
                        
            # Alert Logic
            if info["name"] == "UNKNOWN":
                elapsed = time.time() - info.setdefault("first_unknown_time", time.time())
                if elapsed >= 5 and time.time() - last_alert_time.get(cam_name, 0) > 10:
                    trigger_alert(frame, cam_name, tid)
                    last_alert_time[cam_name] = time.time()
            else:
                info.pop("first_unknown_time", None)
            
            track_info[tid] = info

            # Draw
            display_name = info["name"]
            color = (0, 200, 0) if " (?)" not in display_name and display_name != "UNKNOWN" else (0, 165, 255)
            if display_name == "UNKNOWN": color = (0, 0, 255)
            
            label = f"{display_name}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Log specific person events
            if display_name != "UNKNOWN" and " (?)" not in display_name:
                bbox_wh = (x1, y1, x2-x1, y2-y1)
                log_person_event(display_name, cam_name, tid, frame, bbox_wh)
                check_access_permission(display_name, cam_name, frame, bbox_wh)

        with frames_lock:
            latest_frames[cam_name] = cv2.resize(frame, (TILE_W, TILE_H))

    cap.release()
    print(f"[{cam_name}] exiting")

# (Dashboard and Main Execution remain the same)
def draw_cam_label(frame, cam_name):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (TILE_W, 30), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, cam_name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def dashboard_loop(camera_list):
    n = len(camera_list)
    if n == 0: return
    grid_cols = int(np.ceil(np.sqrt(n)))
    grid_rows = int(np.ceil(n / grid_cols))
    while True:
        with frames_lock:
            frames = []
            for _, cam_name in camera_list:
                f = latest_frames.get(cam_name)
                if f is None: f = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
                else: f = f.copy()
                f = draw_cam_label(f, cam_name)
                frames.append(f)
        while len(frames) < grid_cols * grid_rows:
            frames.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))
        rows = [np.hstack(frames[r*grid_cols:(r+1)*grid_cols]) for r in range(grid_rows)]
        grid = np.vstack(rows)
        cv2.imshow("Dashboard", grid)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"): break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_list = [
        ("vid3.mp4", "Lobby_Cam"),
        ("vid4.mp4", "Entrance_Cam"),
        # (0, "Webcam") 
    ]
    threads = []
    for src, cam in camera_list:
        t = threading.Thread(target=process_camera, args=(src, cam), daemon=True)
        t.start()
        threads.append(t)
    try:
        dashboard_loop(camera_list)
    except KeyboardInterrupt:
        print("Interrupted by user")
    print("Exiting.")