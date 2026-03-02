
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

print("Keys in app:", dir(app))
if hasattr(app, 'det_model'):
    print("Has det_model")
else:
    print("No det_model")
    
if hasattr(app, 'rec_model'):
    print("Has rec_model")
    
# print models dict if exists
if hasattr(app, 'models'):
    print("Models:", app.models.keys())
