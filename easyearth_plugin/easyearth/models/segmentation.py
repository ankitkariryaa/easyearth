"""Segmentation models from hugging face"""

# Load model directly
from transformers import AutoImageProcessor, SegformerConfig, AutoModelForSemanticSegmentation
from PIL import Image
from transformers import pipeline
import numpy as np
from transformers import SegformerConfig
import torch 
from pathlib import Path
from typing import Union
try:
    from .base_model import BaseModel
except ImportError:
    # For direct script execution
    from base_model import BaseModel

class Segmentation(BaseModel):
    def __init__(self, model_path: str = "restor/tcd-segformer-mit-b5") -> None:
        """Initialize SegFormer model
        Args:
            model_path: Model identifier or path
        """
        super().__init__(model_path)
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.logger.debug(f"Loading model from {model_path}")
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_path)
        self.logger.debug(f"Model loaded successfully")
        self.semantic_segmentation = pipeline("image-segmentation", model_path)
        self.logger.debug(f"Pipeline created successfully")
        self.config = SegformerConfig.from_pretrained(model_path)
        self.logger.debug(f"Model config loaded successfully")

    def get_masks(self, image: Union[str, Path, Image.Image, np.ndarray]):
        """Get the masks for a given prompt
        Args:
            image: The image to process
        Returns: 
            masks
        """

        self.logger.debug(f"Processing image: {image}")
        if isinstance(image, str) or isinstance(image, Path):
            raw_image = Image.open(image).convert("RGB")
        else:
            raw_image = image

        with torch.no_grad():
            inputs = self.processor(raw_image, return_tensors='pt')
            preds = self.model(pixel_values=inputs.pixel_values)
            target_size = [(raw_image.size[1], raw_image.size[0])]
            masks = self.processor.post_process_semantic_segmentation(preds, target_sizes=target_size)
        return masks
    

if __name__=='__main__':    
    segformer = Segmentation(model_path="restor/tcd-segformer-mit-b5")
    # # Handle local GeoTIFF files
    image_path = "/home/yan/Downloads/data/DJI_0108.JPG"
    image = Image.open(image_path).convert('RGB')
    masks = segformer.get_masks(image)
    geojson = segformer.raster_to_vector(masks, img_transform=None, filename="/tmp/masks.geojson")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.title("RGB")
    plt.imshow(np.array(image[0].numpy()))
    plt.subplot(122)
    plt.title("Prediction")
    plt.imshow(masks, interpolation='nearest')
    plt.show() 