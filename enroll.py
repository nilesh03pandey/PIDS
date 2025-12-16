import os
import cv2
import torch
import numpy as np
import argparse
from pymongo import MongoClient
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
import datetime

# --- CONFIGURATION ---
# Ensure these match your main app.py settings
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "person_reid"
COLLECTION_NAME = "people"

YOLO_WEIGHTS = "yolov8n.pt"  # Use the same YOLO model for consistency
REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_PATH = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- END CONFIGURATION ---


def l2norm(v: np.ndarray):
    """Applies L2 normalization to a vector."""
    return v / (np.linalg.norm(v) + 1e-12)

def extract_feature_from_crop(image_crop, extractor):
    """Extracts a Re-ID feature vector from a single cropped image."""
    if image_crop is None or image_crop.size == 0:
        return None
    
    crop_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        feat_t = extractor([crop_rgb]) # Extractor expects a list of images
    
    feat = feat_t[0].cpu().numpy().flatten()
    return l2norm(feat)


def main(args):
    print(f"Using device: {DEVICE}")

    # 1. Initialize Models
    print("Loading models...")
    yolo_model = YOLO(YOLO_WEIGHTS)
    yolo_model.to(DEVICE)

    reid_extractor = FeatureExtractor(
        model_name=REID_MODEL_NAME,
        model_path=REID_MODEL_PATH,
        device=DEVICE
    )
    print("Models loaded successfully.")

    # 2. Connect to MongoDB
    print(f"Connecting to MongoDB at {MONGO_URI}...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    people_col = db[COLLECTION_NAME]
    print("MongoDB connected.")

    # 3. Process Images in the provided directory
    image_files = [f for f in os.listdir(args.path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"Error: No images found in directory '{args.path}'.")
        return

    print(f"Found {len(image_files)} images. Processing...")
    
    all_features = []
    for filename in image_files:
        image_path = os.path.join(args.path, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"  - Warning: Could not read image {filename}. Skipping.")
            continue

        # Detect people in the image
        results = yolo_model(image, verbose=False, classes=[0])
        
        person_boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(person_boxes) == 0:
            print(f"  - Warning: No person detected in {filename}. Skipping.")
            continue
        
        # Find the largest detected person (heuristic for the main subject)
        largest_box = max(person_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        x1, y1, x2, y2 = map(int, largest_box)
        
        # Crop the person from the image
        person_crop = image[y1:y2, x1:x2]
        
        # Extract Re-ID feature from the crop
        feature = extract_feature_from_crop(person_crop, reid_extractor)
        
        if feature is not None:
            all_features.append(feature.tolist()) # Convert numpy array to list for MongoDB
            print(f"  - Successfully extracted feature from {filename}.")
        else:
            print(f"  - Warning: Could not extract feature from {filename}.")

    if not all_features:
        print("\nError: Could not extract any features. Please use clearer images. Aborting enrollment.")
        return

    # 4. Save to Database
    person_doc = {
        "name": args.name,
        "role": args.role,
        "features": all_features,
        "registered_at": datetime.datetime.now()
    }
    
    # Check if person already exists. If so, append features. Otherwise, insert new.
    existing_person = people_col.find_one({"name": args.name})
    
    if existing_person:
        result = people_col.update_one(
            {"_id": existing_person["_id"]},
            {"$push": {"features": {"$each": all_features}}}
        )
        if result.modified_count > 0:
            print(f"\n✅ Success: Updated existing person '{args.name}' with {len(all_features)} new image features.")
        else:
            print("\nError: Failed to update the person in the database.")
    else:
        result = people_col.insert_one(person_doc)
        print(f"\n✅ Success: Enrolled new person '{args.name}' with ID '{result.inserted_id}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a new person into the Re-ID database.")
    parser.add_argument("--name", type=str, required=True, help="Name of the person.")
    parser.add_argument("--role", type=str, required=True, help="Role of the person (e.g., 'Employee', 'Visitor').")
    parser.add_argument("--path", type=str, required=True, help="Path to the folder containing images of the person.")
    
    args = parser.parse_args()
    
    main(args)