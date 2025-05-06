"""Predicts object masks for a given prompts using SAM"""

import torch
from transformers import SamModel, SamProcessor

from PIL import Image
import requests

import rasterio
from rasterio.features import shapes as get_shapes
import numpy as np
import geopandas as gpd

from rasterio import features

class Sam:
    def __init__(self, model_version="facebook/sam-vit-huge"):
        """Initialize the SAM model
        Args:
            model_version: The model version to use
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def get_image_embeddings(self, model, processor, raw_image):
        """Get the image embeddings for a given image
        Args:
            model: The model to use
            processor: The processor to use
            raw_image: The image to process
        Returns: The image embeddings
        """

        inputs = processor(raw_image, return_tensors="pt").to(self.device)
        image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
        return image_embeddings

    def get_masks(self, model, raw_image, processor, image_embeddings, input_points=None, input_boxes=None, input_labels=None, multimask_output=True):
        """Get the masks for a given prompt
        Args:
            model: The model to use
            processor: The processor to use
            image_embeddings: The image embeddings
            input_points: The 2D points
            input_boxes: The bounding boxes
            input_labels: The labels
        Returns: The masks and the scores
        """

        inputs = processor(raw_image, input_points=input_points, input_boxes=input_boxes, input_labels=input_labels, return_tensors="pt").to(self.device)
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=multimask_output)  # TODO: so maybe at the moment do not allow hollow masks where it requires multimask_output=True...

        masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores.cpu()
        return masks, scores

    def raster_to_vector(self, masks, scores, img_transform=None, filename=None):
        """Converts a raster mask to a vector mask
        Args:
            masks: The raster mask
        Returns: The vector mask
        """

        num_masks = masks[0].shape[0]
        num_scores = masks[0].shape[1]

        # get the object id for each mask
        objects_id = torch.arange(num_masks).view(-1, 1, 1, 1).expand_as(
            masks[0])  # the first dimension is the object id
        masks_id = torch.where(masks[0], objects_id + 1, torch.tensor(
            0))  # the second dimension is the mask id, one object may have multiple predicted masks with different confidence scores

        # if multimask_output is True, then we have multiple masks for each object with different scores, then we choose the one with the highest score
        if num_scores > 1:
            # TODO: verity if this is correct
            # Get the index of the mask with the highest iou score
            highes_score_idx = torch.argmax(scores, dim=2)
            assert highes_score_idx.shape[-1] == masks[0].shape[0]

            # Get the mask with the highest score
            masks_list = []
            for obj, sco in enumerate(highes_score_idx[0].tolist()):
                masks_list.append(masks_id[obj, sco, :, :])
            masks_highest = torch.stack(masks_list, dim=0)

            # Convert to the dimensions suitable for super().raster_to_vector
            masks_combined = masks_highest
        else:
            # Convert to the dimensions suitable for super().raster_to_vector
            masks_combined = masks_id.squeeze(1)

        # TODO: is there a better way? this will cause potential problems for overlapping predictions -> for now, the latter prediction will overwrite the former...
        # reduce the dimensions of the masks to 2D by choosing the largest value
        if len(masks_combined.shape) > 2:
            masks_combined = np.max(masks_combined, axis=0)

        # convert tensor to numpy array
        if isinstance(masks_combined, torch.Tensor):
            masks_combined = masks_combined.cpu().numpy()
        # convert to uint8
        masks_combined = (masks_combined > 0).astype(np.uint8)

        if img_transform is not None:
            shape_generator = features.shapes(
                masks_combined,
                mask=masks_combined>0,
                transform=img_transform,
            )
        else:
            shape_generator = features.shapes(
                masks_combined,
                mask=masks_combined>0,
            )

        geojson = [
            {"properties": {"uid": value}, "geometry": polygon}
            for polygon, value in shape_generator
        ]

        if filename:
            # save geojson as .geojson file
            gdf = gpd.GeoDataFrame.from_features(geojson)
            gdf.to_file(filename=filename, driver="GeoJSON")

        return geojson


if __name__ == "__main__":

    image_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    img_transform = rasterio.transform.from_bounds(0, 0, 1024, 1024, 1024, 1024)

    input_points = [[[850, 1100], [2250, 1000]]]
    input_boxes = [[[650, 900, 1000, 1250]]]
    multiple_boxes = [[[620, 900, 1000, 1255], [2050, 800, 2400, 1150]]]
    input_labels = [[0]]
    multiple_points = [[[[850, 1100]], [[2250, 1000]]]] # one mask for each point

    sam = Sam()
    image_embeddings = sam.get_image_embeddings(sam.model, sam.processor, raw_image)

    # No prompts
    masks, scores = sam.get_masks(sam.model, raw_image, sam.processor, image_embeddings)
    geojson = sam.raster_to_vector(masks, scores, img_transform, filename="/tmp/masks_no.geojson")
    # Single points
    masks, scores = sam.get_masks(sam.model, raw_image, sam.processor, image_embeddings, input_points=input_points)
    geojson = sam.raster_to_vector(masks, scores, img_transform, filename="/tmp/masks_point.geojson")
    # Multiple points, and one point one (set of) mask
    masks, scores = sam.get_masks(sam.model, raw_image, sam.processor, image_embeddings, input_points=multiple_points)
    geojson = sam.raster_to_vector(masks, scores, img_transform, filename="/tmp/masks_multipoint.geojson")
    # Single bounding box
    masks, scores = sam.get_masks(sam.model, raw_image, sam.processor, image_embeddings, input_boxes=input_boxes)
    geojson = sam.raster_to_vector(masks, scores, img_transform, filename="/tmp/masks_bbox.geojson")
    # Bounding box and point with label 0 to mask out parts of the image
    masks, scores = sam.get_masks(sam.model, raw_image, sam.processor, image_embeddings, input_points=input_points, input_boxes=input_boxes, input_labels=input_labels)
    geojson = sam.raster_to_vector(masks, scores, img_transform, filename="/tmp/masks_point_bbox.geojson")
    # Multiple prompts and multiple masks
    masks, scores = sam.get_masks(sam.model, raw_image, sam.processor, image_embeddings, input_boxes=multiple_boxes)
    geojson = sam.raster_to_vector(masks, scores, img_transform, filename="/tmp/masks_multibbox.geojson")