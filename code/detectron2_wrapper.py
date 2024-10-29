import sys, os, distutils.core
# %%
import torch, detectron2
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
from cv2 import imshow
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
import matplotlib.pyplot as plt

def run_detectron(image_path: str):
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    setup_logger()

    # Read the image
    im = cv2.imread(image_path)

    # Configure model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # Set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Set the model to run on CPU instead of CUDA
    cfg.MODEL.DEVICE = 'cpu'  # Use 'cpu' instead of 'cuda'

    predictor = DefaultPredictor(cfg)

    # Get the predictions
    outputs = predictor(im)

    # Extract the instances without the mask information
    instances = outputs["instances"].to("cpu")

    # Remove the mask predictions from the output
    instances.remove("pred_masks")
    return instances