import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load an image
image_path = "your_image_path_here.jpg"  # Replace with your image file path
image = cv2.imread(image_path)

# Setup configuration for PointRend (with Mask R-CNN as base)
cfg = get_cfg()
cfg.merge_from_file("configs/PointRend/mask_rcnn_R_50_FPN_3x.yaml")  # Path to PointRend config file
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Threshold for detection
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # Pretrained weights for Mask R-CNN
cfg.freeze()

# Create predictor
predictor = DefaultPredictor(cfg)

# Run inference on the image
outputs = predictor(image)

# Extract predicted segmentation masks and bounding boxes
instances = outputs["instances"].to("cpu")
boxes = instances.pred_boxes.tensor.numpy()  # Bounding boxes (x1, y1, x2, y2)
masks = instances.pred_masks.numpy()  # Segmentation masks

# Assume we provide a prompt in the form of a point or a bounding box
# Example: Define a point (x, y) or a bounding box [x1, y1, x2, y2] (this would be your prompt)
# We simulate using the first predicted bounding box as the prompt for segmentation

point_prompt = (int(boxes[0][0]), int(boxes[0][1]))  # First bounding box's top-left corner (point prompt)
print(f"Using point prompt: {point_prompt}")

# Visualize the segmentation with the bounding box prompt
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(instances.pred_classes.numpy())
v = v.draw_panoptic_segmentation(outputs["instances"].to("cpu"), color="yellow")

# Show the image with segmentation result
cv2.imshow("PointRend Segmentation", v.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
