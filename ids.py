# app.py
import os
import cv2
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

# --- Model and Device Configuration ---
YOLO_WEIGHTS = "yolov11n.pt" 
REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_PATH = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- NEW PERFORMANCE OPTIMIZATIONS ---
# 1. Set a lower camera resolution to reduce the amount of data processed
CAM_RESOLUTION_W = 640
CAM_RESOLUTION_H = 480
# 2. Tell YOLO to use a smaller image size for faster inference
YOLO_INFERENCE_SIZE = 320
# 3. Process video frames less frequently to reduce CPU/GPU load
PROCESS_EVERY_N_FRAMES = 1
# --- END PERFORMANCE OPTIMIZATIONS ---

# --- Tuning Parameters ---
STRICT_TH = 0.35      # Cosine distance for a strong match
LOOSE_TH = 0.55       # Cosine distance for a weak match
RATIO_MARGIN = 0.75   # Ratio test: best_dist / second_best_dist must be < this
YOLO_CONF_TH = 0.4    # Detection confidence threshold

# re-ID dynamic update params
REID_INTERVAL = 5
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

print(f"Using device: {DEVICE}") # <<<<<<<<<< CHECK THIS LINE IN YOUR TERMINAL!
yolo_model = YOLO(YOLO_WEIGHTS)
yolo_model.to(DEVICE)

extractor = FeatureExtractor(
    model_name=REID_MODEL_NAME,
    model_path=REID_MODEL_PATH,
    device=DEVICE
)

# (All functions from trigger_alert to dist_to_conf remain exactly the same)
# --- Alerting and Logging Functions (largely unchanged, but with small improvements) ---

def trigger_alert(frame, cam_name, tid):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🚨 [ALERT] UNKNOWN detected on {cam_name} (track {tid}) at {ts}")
    snap_path = f"alert_snapshots/{cam_name}_{tid}_{int(time.time())}.jpg"
    os.makedirs("alert_snapshots", exist_ok=True)
    cv2.imwrite(snap_path, frame)
    threading.Thread(target=lambda: winsound.Beep(1000, 700), daemon=True).start()
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
    print(f" ZONE ALERT] {name} entered restricted camera: {cam_name} at {ts}")
    threading.Thread(target=lambda: winsound.Beep(1200, 800), daemon=True).start()
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
    global last_alert_time  # FIX: Add this line to use the global variable

    print(f"[{cam_name}] starting, source={source}")
    cap = cv2.VideoCapture(source)
    
    # MODIFIED: Set the desired camera resolution for performance
    if isinstance(source, int): # Only set for webcams, not video files
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RESOLUTION_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION_H)

    tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.3)
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

        # MODIFIED: Pass the smaller inference size to the YOLO model
        results = yolo_model(frame, verbose=False, classes=[0], imgsz=YOLO_INFERENCE_SIZE)
        
        detections = []
        for r in results[0].boxes:
            conf = float(r.conf[0])
            if conf >= YOLO_CONF_TH:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if w < 20 or h < 40: continue
                detections.append(([x1, y1, w, h], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed(): continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            tid = track.track_id
            info = track_info.get(tid, {
                "name": "UNKNOWN", "role": "", "conf": 0.0, "last_feat": None,
                "last_reid_frame": -999, "last_seen": time.time()
            })
            info["last_seen"] = time.time()
            
            # This is the section that was causing the error
            if info["name"] == "UNKNOWN":
                elapsed = time.time() - info.setdefault("first_unknown_time", time.time())
                # Now this line will work correctly
                if elapsed >= 5 and time.time() - last_alert_time.get(cam_name, 0) > 10:
                    trigger_alert(frame, cam_name, tid)
                    last_alert_time[cam_name] = time.time()
            else:
                info.pop("first_unknown_time", None)

            # ... rest of the function continues ...
            if (frame_idx - info["last_reid_frame"]) >= REID_INTERVAL:
                feat = extract_feature_from_crop(frame, (x1, y1, x2, y2))
                info["last_reid_frame"] = frame_idx
                if feat is not None:
                    name, role, dist, strong = match_person(feat, known_people)
                    new_conf = dist_to_conf(dist)
                    if info["name"] == "UNKNOWN":
                        if new_conf >= MIN_CONF_ASSIGN: info.update({"name": name, "role": role, "conf": new_conf, "last_feat": feat.copy()})
                        elif name:
                            track_vote[tid].append(name)
                            if list(track_vote[tid]).count(name) >= CONSECUTIVE_UPDATES: info.update({"name": name, "role": role, "conf": new_conf, "last_feat": feat.copy()})
                    else:
                        current_name, current_conf = info["name"], info["conf"]
                        if name == current_name:
                            info["conf"] = max(current_conf, new_conf)
                            info["last_feat"] = EMA_ALPHA_TRACK * info["last_feat"] + (1 - EMA_ALPHA_TRACK) * feat
                            info["last_feat"] = l2norm(info["last_feat"])
                        elif strong and new_conf > current_conf + CONF_MARGIN:
                            info.update({"name": name, "role": role, "conf": new_conf, "last_feat": feat.copy()})
                            track_vote[tid].clear()
                        elif name:
                            track_vote[tid].append(name)
                            counts = {c: track_vote[tid].count(c) for c in set(track_vote[tid])}
                            top_candidate, top_count = max(counts.items(), key=lambda x: x[1])
                            if top_candidate != current_name and top_count >= CONSECUTIVE_UPDATES:
                                rp = next((p for p in known_people if p["name"] == top_candidate), None)
                                info.update({"name": top_candidate, "role": rp["role"] if rp else "", "conf": new_conf, "last_feat": feat.copy()})
            track_info[tid] = info
            
            is_certain_match = info["name"] and " (?)" not in info["name"] and info["name"] != "UNKNOWN"
            if is_certain_match:
                bbox = (x1, y1, x2 - x1, y2 - y1)
                log_person_event(info["name"], cam_name, tid, frame, bbox)
                check_access_permission(info["name"], cam_name, frame, bbox)
            
            display_name = info["name"]
            color = (0, 200, 0) if " (?)" not in display_name and display_name != "UNKNOWN" else (0, 165, 255)
            if display_name == "UNKNOWN": color = (0, 0, 255)
            label = f"{display_name}" + (f" ({info['role']})" if info.get("role") else "")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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