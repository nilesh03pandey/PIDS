import cv2
import numpy as np
import os
import torch

# FIX: Add PyTorch's bundled CUDA libraries to DLL search path so ONNX Runtime can find them.
# This fixes "LoadLibrary failed with error 126" on Windows when using onnxruntime-gpu.
if os.name == 'nt' and torch.cuda.is_available():
    try:
        libs_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        os.add_dll_directory(libs_path)
        # print(f"[FaceEngine] Added CUDA libs from: {libs_path}")
    except Exception as e:
        print(f"[FaceEngine] Warning: Could not add PyTorch libs to DLL path: {e}")

import insightface
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, provider="CUDAExecutionProvider", det_thresh=0.5, det_size=(640, 640)):
        """
        Initialize InsightFace app with SCRFD detector and ArcFace recognizer.
        """
        # "buffalo_l" contains SCRFD-10G (detection) and ArcFace-R100 (recognition)
        # It's a good balance of speed and accuracy. 
        # For absolute max speed, "buffalo_s" (SCRFD-500M + ArcFace-MobileNet) could be used.
        self.app = FaceAnalysis(name="buffalo_l", providers=[provider])
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.det_thresh = det_thresh

    def detect_and_embed(self, frame):
        """
        Detect faces and extract 512-dim embeddings.
        Returns a list of dicts: {'bbox': [x1, y1, x2, y2], 'embedding': np.array, 'score': float}
        """
        faces = self.app.get(frame)
        results = []
        for face in faces:
            if face.det_score < self.det_thresh:
                continue
            
            bbox = face.bbox.astype(int)
            results.append({
                "bbox": bbox,
                "embedding": face.embedding, # 512-dim normalized vector
                "score": face.det_score,
                "kps": face.kps # Keypoints (eyes, nose, mouth) - useful for alignment if needed
            })
        return results

    def get_face_embedding(self, face_crop):
        """
        Extract embedding from a pre-cropped face image.
        """
        if face_crop is None or face_crop.size == 0:
            return None
            
        # Refine the crop with the detector to get precise landmarks/alignment
        # This is better than raw embedding on unaligned crop
        faces = self.app.get(face_crop)
        
        if not faces:
            # Fallback: if detection fails on crop (maybe too tight?), return None
            # We could try to force embedding extraction but alignment matters a lot for ArcFace.
            return None
            
        # Return the largest face found in the crop
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        return face.embedding
