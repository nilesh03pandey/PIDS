# register_person.py
import cv2
import torch
from torchreid.utils import FeatureExtractor
from pymongo import MongoClient
import datetime
import time
import numpy as np

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["person_reid"]
people_col = db["people"]

# Torchreid extractor
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='C:/Users/adity/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def extract_feature(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Failed to load image {img_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feat = extractor([img_rgb])  # torch tensor
    return feat[0].cpu().numpy()


def register_person(name, role, image_paths):
    features = []
    for path in image_paths:
        feat = extract_feature(path)
        if feat is not None:
            # ensure numpy array
            if isinstance(feat, list):
                feat = np.array(feat)
            features.append(feat)

    if not features:
        print(f"[!] No valid images found for {name}, skipping registration.")
        return

    record = {
        "name": name,
        "role": role,
        "features": [f.flatten().tolist() for f in features if f is not None],
        "registered_at": time.time()
    }

    people_col.insert_one(record)
    print(f"[+] Registered {name} ({role}) with {len(features)} samples")




# Example usage
if __name__ == "__main__":
    register_person("adi", "malak", ["samples/adi1.jpeg", "samples/adi2.jpeg", "samples/adi3.jpeg", "samples/adi4.jpeg", "samples/adi5.jpeg", "samples/adi6.jpeg", "samples/adi7.jpeg", "samples/adi8.jpeg", "samples/adi9.jpeg", "samples/adi10.jpeg", "samples/adi11.jpeg", "samples/adi12.jpg", "samples/adi13.jpg"])
    # register_person("aish", "employee9", ["samples/aish.jpeg", "samples/aish1.jpeg", "samples/aish2.jpeg", "samples/aish3.jpeg", "samples/aish4.jpg", "samples/aish5.jpeg", "samples/aish6.jpeg", "samples/aish7.jpeg", "samples/aish8.jpeg", "samples/aish9.jpeg", "samples/aish10.jpeg", "samples/aish11.jpeg", "samples/aish12.jpeg", "samples/aish13.jpeg"])
    # register_person("navya", "employee2", ["samples/nav1.jpeg", "samples/nav2.jpeg", "samples/nav3.jpeg", "samples/nav4.jpg", "samples/nav5.jpeg", "samples/nav6.jpeg", "samples/nav7.jpeg", "samples/nav8.jpeg", "samples/nav9.jpeg", "samples/nav10.jpeg", "samples/nav11.jpg"])


