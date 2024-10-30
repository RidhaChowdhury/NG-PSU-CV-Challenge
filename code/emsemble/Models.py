from abc import ABC, abstractmethod
class Model():
    def __init__(self):
        pass

    @abstractmethod
    def detect(self, image_path):
        """
        :param image_path: path to image
        :return: list of detected objects
        """
        pass

from PIL import Image
import supervision as sv
from ultralytics import YOLO
import numpy as np

class YoloModel(Model):
    def __init__(self):
        super().__init__()
        self.model = YOLO('yolo11n.pt')

    def detect(self, image_path: str, conf: float = 0.4):
        image = Image.open(image_path)
        result = self.model.predict(image, conf=0.4)[0]

        # transform output
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class predictions

        boxes = np.array(boxes)  # No conversion needed
        scores = np.array(scores)
        classes = np.array(classes, dtype=int)

        return {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
        }

import math
from PIL import Image
import requests
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import os
import requests
import supervision as sv
class DetrModel(Model):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        torch.set_grad_enabled(False)
        self.CLASSES = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def detect(self, image_path: str, conf: float = 0.4):
        im = Image.open(image_path)
        # mean-std normalize the input image (batch-size: 1)
        img = self.transform(im).unsqueeze(0)

        # propagate through the model
        outputs = self.model(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.85

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

        detr_boxes = bboxes_scaled
        detr_scores = probas[keep].max(-1).values
        detr_classes = probas[keep].argmax(-1)

        boxes = detr_boxes.cpu().numpy() if isinstance(detr_boxes, torch.Tensor) else np.array(detr_boxes)
        scores = detr_scores.cpu().numpy() if isinstance(detr_scores, torch.Tensor) else np.array(detr_scores)
        classes = detr_classes.cpu().numpy().astype(int) if isinstance(detr_classes, torch.Tensor) else np.array(detr_classes, dtype=int)

        output = {
            "boxes": boxes,
            "scores": scores,
            "classes": classes
        }
        return output