"""SAM model implementation"""

try:
    from .base_model import BaseModel
except ImportError:
    # For direct script execution
    from base_model import BaseModel
from transformers import SamModel, SamProcessor
import torch
from PIL import Image
import numpy as np
from typing import Optional, List, Tuple, Union, Any
import requests
import rasterio
from pathlib import Path

class Sam(BaseModel):
    def __init__(self, model_version: str = "facebook/sam-vit-huge"):
        """Initialize the SAM model
        Args:
            model_version: The model version to use
        """
        super().__init__(model_version)
        self.model = SamModel.from_pretrained(model_version).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_version)

    def get_metadata(self, image):
        """Get the metadata for a given image
        Args:
            image: The image to process
        Returns: The metadata
        """

        with rasterio.open(image) as src:
            metadata = src.meta
        return metadata

    def get_image_embeddings(self, raw_image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Get the image embeddings for a given image
        Args:
            raw_image: The image to process, np.ndarray
        Returns: 
            The image embeddings
        """
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
        image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        return image_embeddings

    def get_masks(self, 
                 image: Union[str, Path, Image.Image, np.ndarray],
                 input_points: Optional[List] = None,
                 input_boxes: Optional[List] = None,
                 input_labels: Optional[List] = None,
                 image_embeddings: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the masks for a given prompt
        Args:
            image: The image to process
            input_points: Optional point prompts
            input_boxes: Optional box prompts
            input_labels: Optional labels
            image_embeddings: Optional pre-computed embeddings
        Returns: 
            Tuple of (masks, scores)
        """
        if isinstance(image, str) or isinstance(image, Path):
            raw_image = Image.open(image).convert("RGB")
        else:
            raw_image = image

        if image_embeddings is None:
            image_embeddings = self.get_image_embeddings(raw_image)

        inputs = self.processor(
            raw_image, 
            input_points=input_points, 
            input_boxes=input_boxes, 
            input_labels=input_labels, 
            return_tensors="pt"
        ).to(self.device)
        
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})

        with torch.no_grad():
            outputs = self.model(**inputs)

        # TODO: should this be on gpu or cpu?
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores.cpu()
        
        return masks, scores

    def raster_to_vector(self, masks: Union[torch.Tensor, np.ndarray], scores: Union[torch.Tensor, np.ndarray], 
                         img_transform: Optional[Any] = None, filename: Optional[str] = None):
        """Extends base raster_to_vector with SAM-specific processing"""
        
        # Get the index of the mask with the highest iou score
        highes_score_idx = torch.argmax(scores, dim=2)
        assert highes_score_idx.shape[-1] == masks[0].shape[0]

        # Change boolean to index id
        depth_ids = torch.arange(masks[0].size(0)).view(-1, 1, 1, 1).expand_as(masks[0])
        masks_id = torch.where(masks[0], depth_ids+1, torch.tensor(0))

        # Get the mask with the highest score
        masks_list = []
        for x, y in enumerate(highes_score_idx):
            masks_list.append(masks_id[x, y, :, :])
        masks_highest = torch.stack(masks_list, dim=0)
        masks_combined = torch.amax(masks_highest, dim=0, keepdim=False).numpy().astype(np.uint8)
        
        assert masks_combined.sum() > 0  # check if there are any masks

        return super().raster_to_vector(masks_combined, img_transform, filename)


if __name__ == "__main__":

    image_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    img_transform = rasterio.transform.from_bounds(0, 0, 1024, 1024, 1024, 1024)

    input_points = [[[820, 1080]]]
    input_boxes = [[[650, 900, 1000, 1250]]]
    multiple_boxes = [[[620, 900, 1000, 1255], [2050, 800, 2400, 1150]]]
    input_labels = [[0]]

    sam = Sam()
    image_embeddings = sam.get_image_embeddings(raw_image)

    # No prompts
    masks, scores = sam.get_masks(raw_image, image_embeddings=image_embeddings)
    geojson = sam.raster_to_vector(masks, scores, img_transform, filename="/tmp/masks_no.geojson")
    # Single points
    masks, scores = sam.get_masks(raw_image, image_embeddings=image_embeddings, input_points=input_points)
    geojson = sam.raster_to_vector(masks, scores, img_transform, filename="/tmp/masks_point.geojson")
    # Single bounding box
    masks, scores = sam.get_masks(raw_image, image_embeddings=image_embeddings, input_boxes=input_boxes)
    geojson = sam.raster_to_vector(masks, scores, img_transform, filename="/tmp/masks_bbox.geojson")
    # Bounding box and point with label 0 to mask out parts of the image
    masks, scores = sam.get_masks(raw_image, image_embeddings=image_embeddings, input_points=input_points, input_boxes=input_boxes, input_labels=input_labels)
    geojson = sam.raster_to_vector(masks, scores, img_transform, filename="/tmp/masks_point_bbox.geojson")
    # Multiple prompts and multiple masks
    masks, scores = sam.get_masks(raw_image, image_embeddings=image_embeddings, input_boxes=multiple_boxes)
    geojson = sam.raster_to_vector(masks, scores, img_transform, filename="/tmp/masks_multibbox.geojson")