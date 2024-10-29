
import ultralytics
# %%
import os
# %%
from PIL import Image
import supervision as sv
from ultralytics import YOLO

def run_yolo(image_path: str):
    model = YOLO('yolo11n.pt')
    # Loop through all files in the folder
    image = Image.open(image_path)

    # Run YOLO model prediction
    result = model.predict(image, conf=0.4)[0]

    return result