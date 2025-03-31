from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import rasterio
import pyproj  # Add this for CRS transformation
from easyearth.models.sam import Sam
from PIL import Image
import requests

def reorgnize_prompts(prompts):
    """
    Reorganize prompts into a dictionary with separate lists for points, labels, boxes, and text

    Args:
        prompts (list): List of prompt dictionaries
    Returns:
        transformed_prompts (dict): Dictionary with separate lists for points, labels, boxes, and text
    """

    transformed_prompts = {
        'points': [],
        'labels': [],
        'boxes': [],
        'text': []
    }

    for prompt in prompts:
        prompt_type = prompt.get('type')
        prompt_data = prompt.get('data', {})
        if prompt_type == 'Point':
            transformed_prompts['points'].append(prompt_data.get('points', []))
            transformed_prompts['labels'].append(prompt_data.get('labels', []))
        elif prompt_type == 'Box':
            transformed_prompts['boxes'].append(prompt_data.get('boxes', []))
        elif prompt_type == 'Text':
            transformed_prompts['text'].append(prompt_data.get('text', []))

    return transformed_prompts


def reproject_prompts(prompts, transform, image_shape):
    """
    Transform all types of prompts from map coordinates to pixel coordinates

    Args:
        prompts (list): List of prompt dictionaries
        transform (affine.Affine): Rasterio transform object
        image_shape (tuple): Image dimensions (height, width)
    """

    height, width = image_shape[:2]

    def clip_coordinates(x, y):
        """Clip coordinates to image boundaries"""
        x = max(0, min(int(x), width - 1))
        y = max(0, min(int(y), height - 1))
        return x, y

    input_points = prompts.get('points', [])
    input_labels = prompts.get('labels', [])
    input_boxes = prompts.get('boxes', [])
    input_text = prompts.get('text', [])

    transformed = {
        'points': [],
        'labels': [],
        'boxes': [],
        'text': []
    }

    if input_points:
        for point in input_points:
            px, py = ~transform * (point[0], point[1])
            px, py = clip_coordinates(px, py)
            transformed['points'].append([px, py])

        transformed['labels'].extend(input_labels)

    if input_boxes:
        for box in input_boxes:
            x1, y1 = ~transform * (box[0], box[1])
            x2, y2 = ~transform * (box[2], box[3])

            x1, y1 = clip_coordinates(x1, y1)
            x2, y2 = clip_coordinates(x2, y2)

            transformed['boxes'].append([
                min(x1, x2),
                min(y1, y2),
                max(x1, x2),
                max(y1, y2)
            ])

    if input_text:
        transformed['text'].extend(input_text)

    return transformed

def predict():
    try:
        # Get the image data from the request
        data = request.get_json()

        # data = json_data
        image_path = data.get('image_path')
        target_crs = data.get('target_crs') if data.get('target_crs') else None  # The CRS for the output masks

        if not image_path:
            return jsonify({
                'status': 'error',
                'message': 'image_path is required'
            }), 400

        if image_path.endswith('.tif'):
            # Open the image using rasterio to get geospatial information
            with rasterio.open(image_path) as src:
                # Get image metadata
                transform = src.transform
                source_crs = src.crs.to_string()

                # Read image data
                image_array = src.read()
                # Transpose from (bands, height, width) to (height, width, bands)
                image_array = np.transpose(image_array, (1, 2, 0))
                # get transform of the target CRS using pyproj
            if target_crs is not None and target_crs != source_crs:
                target_crs = pyproj.CRS.from_string(target_crs)
                transform = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True).transform(transform)
        elif image_path.startswith('http'):
            image_array = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
            image_array = np.array(image_array)
            transform = None
        else:
            image_array = np.array(Image.open(image_path))
            transform = None

        # Generate masks
        sam = Sam()

        # Handle image embeddings
        embedding_path = data.get('embedding_path')
        if embedding_path:
            image_embeddings = np.load(embedding_path)
        else:
            embedding_path = "/home/yan/PycharmProjects/easyearth/tmp/image_embeddings.npy"
            image_embeddings = sam.get_image_embeddings(sam.model, sam.processor, image_array)
            np.save(embedding_path, image_embeddings.cpu().numpy())

        # Process prompts
        prompts = data.get('prompts', [])
        if not prompts:
            return jsonify({
                'status': 'error',
                'message': 'prompts are required'
            }), 400

        # Reorgnize the prompts
        prompts = reorgnize_prompts(prompts)

        # Transform all prompts
        if image_path.endswith('.tif'):
            transformed_prompts = reproject_prompts(
                prompts,
                transform,
                image_array.shape
            )
        else:
            transformed_prompts = prompts

        # Use transformed prompts with SAM
        masks, scores = sam.get_masks(
            sam.model,
            image_array,
            sam.processor,
            image_embeddings,
            input_points=transformed_prompts['points'] if transformed_prompts['points'] else None,
            input_labels=transformed_prompts['labels'] if transformed_prompts['labels'] else None,
            input_boxes=transformed_prompts['boxes'] if transformed_prompts['boxes'] else None,
        )

        # Convert masks to GeoJSON with proper coordinate system
        if masks is not None:
            geojson = sam.raster_to_vector(
                masks,
                scores,
                transform,
                filename="/home/yan/PycharmProjects/easyearth/tmp/masks.geojson"
            )

            return jsonify({
                'status': 'success',
                'features': geojson,
                'crs': target_crs
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'No valid masks generated'
            }), 400

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500