# Advanced Multi-Camera Surveillance with Access Control

This project is a real-time, multi-camera surveillance system that detects, tracks, and re-identifies people across different video streams. It includes **Cross-Camera Re-Identification**, **Zone-Based Access Control**, and **Immutable Audit Logs**. The system leverages state-of-the-art deep learning models for high accuracy and provides a robust alerting system for both unknown individuals and unauthorized zone entries.

## 🚀 Key Features

*   **Multi-Camera Support**: Processes multiple video streams (from files or live cameras) concurrently using threading.
*   **Cross-Camera Re-Identification**:
    *   **Global Tracking**: Assigns a unique, consistent ID to a person as they move between different cameras.
    *   **Transition Logging**: Automatically detects and logs when a person exits one camera view and enters another.
*   **Immutable Audit Logs**:
    *   **Tamper-Evident**: Uses SHA-256 hash chaining to link all log entries.
    *   **Secure Signing**: All logs and file artifacts (snapshots) are signed with an HMAC using a secure key.
    *   **Comprehensive Logging**: Tracks all alerts, administrative actions (Enroll/Edit/Delete), and data exports.
*   **Person Detection & Tracking**:
    *   **YOLO**: Fast and accurate person detection.
    *   **DeepSORT**: Robust local tracking within single camera views.
    *   **OSNet (Re-ID)**: Extracts deep feature vectors to recognize individuals across different times and cameras.
*   **Dual Alert System**:
    *   **Unknown Person Alert**: Triggers if an unknown person remains in view for too long.
    *   **Unauthorized Access Alert**: Triggers if a known person enters a restricted camera zone.
*   **Centralized Dashboard**: Displays all camera feeds in a single grid view.

-----

## 🛠️ How It Works (System Architecture)

1.  **Frame Capture**: The system reads frames from multiple video sources.
2.  **Detection & Local Tracking**: YOLO detects people, and DeepSORT assigns local track IDs.
3.  **Cross-Camera Integration**:
    *   The **`GlobalTrackManager`** receives features from all cameras.
    *   It matches new tracks against existing global identities using Re-ID embedding similarity and spatio-temporal constraints.
    *   If a match is found, the person is assigned their existing **Global ID**.
4.  **Identification**: Features are compared against the **MongoDB `people` collection**.
5.  **Audit Logging**:
    *   Every alert, access violation, or system action is sent to the **`AuditLogger`**.
    *   The logger computes a hash of the event data + the previous log's hash, then signs it with a secret key.
    *   The entry is stored in the **`audit_ledger`** database.
6.  **Visualization**: Frames are annotated with Global IDs and names, then displayed on the dashboard.

-----

## ⚙️ Tech Stack

*   **AI / ML**: `PyTorch`, `ultralytics` (YOLO), `torchreid` (OSNet), `deep-sort-realtime`
*   **Database**: `MongoDB` (Stores People, Access Rules, History, and the Immutable Audit Ledger)
*   **Security**: `HMAC`, `SHA-256` (for Audit Logs)
*   **Core**: `OpenCV-Python`, `NumPy`, `SciPy`, `Flask` (Web Interface)

-----

## 🏁 Getting Started

### Prerequisites

*   Python 3.8+
*   MongoDB installed and running on `localhost:27017`.
*   NVIDIA GPU (Recommended).

### 1. Installation

```bash
git clone https://github.com/adityapatil37/PIDS
cd PIDS

# Create and activate venv
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install dependencies (ensure PyTorch matches your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Database Setup

Ensure MongoDB is running. The application will automatically create:
*   `person_reid` DB: For users, access rules, and history.
*   `audit_ledger` DB: For the immutable audit chain.

### 3. Run the Application

```bash
python ids.py
```

*   **Dashboard**: A window showing live camera feeds with tracking.
*   **Web Interface**: Access `http://localhost:5000` (if `main.py` is running) for enrollment and history.

### 4. Admin Web Interface

Run the Flask app to manage users and view logs:

```bash
python main.py
```

*   **Enrollment**: Upload images to register new people.
*   **History**: View attendance logs.
*   **Access Control**: Configure which cameras specific people can access.
*   **Alerts**: View unauthorized access incidents.

-----

## 🛡️ Security Features

### Audit Log Verification
To verify the integrity of the audit logs and ensure no tampering has occurred:

```bash
python test_audit_log.py
```

This script will:
1.  Traverse the entire hash chain in MongoDB.
2.  Re-compute hashes and HMAC signatures for every entry.
3.  Report any broken links or invalid signatures.

-----

## 📁 Project Structure

```
.
├── ids.py                  # Main surveillance system
├── main.py                 # Flask Web Admin Interface
├── global_tracker.py       # Cross-Camera Tracking Logic
├── audit_manager.py        # Immutable Audit Log Implementation
├── alert_snapshots/        # Signed images of alerts
└── static/thumbnails/      # Signed thumbnails of detections
```

---
## 📜 License & Acknowledgements
Copyright © 2025 Aditya Patil. All rights reserved.