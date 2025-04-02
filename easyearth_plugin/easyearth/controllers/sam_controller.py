from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import rasterio
import pyproj  # Add this for CRS transformation
from easyearth.models.sam import Sam
from PIL import Image
import requests
import os
import logging
import json
from datetime import datetime
import sys

# Create logs directory in the plugin directory
PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PLUGIN_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging configuration
log_file = os.path.join(LOG_DIR, f'sam_server_{datetime.now().strftime("%Y%m%d")}.log')

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),  # 'a' for append mode
        logging.StreamHandler(sys.stdout)  # This will print to Docker logs
    ]
)

# Get logger for this module
logger = logging.getLogger(__name__)

# Log some initial information
logger.info(f"=== Server Starting ===")
logger.info(f"Plugin directory: {PLUGIN_DIR}")
logger.info(f"Log directory: {LOG_DIR}")
logger.info(f"Log file: {log_file}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Python path: {sys.path}")

# Test logging
logger.debug("Debug message test")
logger.info("Info message test")
logger.warning("Warning message test")

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
    """Handle prediction request"""
    try:
        # Get the image data from the request
        data = request.get_json()
        logger.debug("=== Received Request ===")
        logger.debug(json.dumps(data, indent=2))

        # Validate and convert image path
        image_path = data.get('image_path')
        if not image_path:
            return jsonify({
                'status': 'error',
                'message': 'image_path is required'
            }), 400

        # Convert host paths to container paths

        # when it contains /home/yan/Downloads
        if image_path.startswith('/home/yan/Downloads'):
            container_path = image_path.replace('/home/yan/Downloads', '/usr/src/app/data')
            logger.debug(f"Converting path from {image_path} to {container_path}")
            image_path = container_path

        logger.debug(f"Checking image path: {image_path}")
        logger.debug(f"File exists: {os.path.exists(image_path)}")
        logger.debug(f"Current working directory: {os.getcwd()}")
        

        if not os.path.exists(image_path):
            return jsonify({
                'status': 'error',
                'message': f'Image file not found at path: {image_path}. Container path: {container_path if "container_path" in locals() else "N/A"}'
            }), 404

        # Load image with detailed error handling
        try:
            if image_path.startswith(('http://', 'https://')):
                # Handle URL images
                logger.debug(f"Loading image from URL: {image_path}")
                response = requests.get(image_path, stream=True)
                response.raise_for_status()  # Raise error for bad status codes
                image = Image.open(response.raw).convert('RGB')
                image_array = np.array(image)
                transform = None
                source_crs = None
                logger.debug(f"Loaded URL image with shape: {image_array.shape}")
            elif image_path.endswith('.tif'):
                # Handle local GeoTIFF files
                logger.debug(f"Loading GeoTIFF from: {image_path}")
                if not os.path.exists(image_path):
                    return jsonify({
                        'status': 'error',
                        'message': f'GeoTIFF file not found: {image_path}'
                    }), 404
                with rasterio.open(image_path) as src:
                    transform = src.transform
                    source_crs = src.crs.to_string()
                    image_array = src.read()
                    image_array = np.transpose(image_array, (1, 2, 0))
                    logger.debug(f"Loaded GeoTIFF with shape: {image_array.shape}")
            else:
                # Handle local regular images
                logger.debug(f"Loading local image from: {image_path}")
                if not os.path.exists(image_path):
                    return jsonify({
                        'status': 'error',
                        'message': f'Image file not found: {image_path}'
                    }), 404
                image = Image.open(image_path).convert('RGB')
                image_array = np.array(image)
                transform = None
                source_crs = None
                logger.debug(f"Loaded local image with shape: {image_array.shape}")

            # Ensure image is in the correct format
            if len(image_array.shape) == 2:
                # Convert grayscale to RGB
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] > 3:
                # Take only first 3 channels if more than 3
                image_array = image_array[:, :, :3]
            
            logger.debug(f"Final image shape: {image_array.shape}, dtype: {image_array.dtype}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image from URL: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to download image: {str(e)}'
            }), 500
        except Exception as e:
            logger.error("Error loading image:")
            logger.exception(e)
            return jsonify({
                'status': 'error',
                'message': f'Failed to load image: {str(e)}'
            }), 500

        # Process prompts with detailed logging
        try:
            prompts = data.get('prompts', [])
            logger.debug("Processing prompts:")
            logger.debug(json.dumps(prompts, indent=2))

            # Transform prompts
            transformed_prompts = reorgnize_prompts(prompts)
            logger.debug("Transformed prompts:")
            logger.debug(json.dumps(transformed_prompts, indent=2))

            # Initialize SAM
            logger.debug("Initializing SAM model")
            sam = Sam()

            # Get embeddings
            logger.debug("Getting image embeddings")
            embedding_path = data.get('embedding_path')
            save_embeddings = data.get('save_embeddings', False)

            if embedding_path and os.path.exists(embedding_path):
                logger.debug(f"Loading existing embedding from: {embedding_path}")
                image_embeddings = np.load(embedding_path)
            else:
                logger.debug("Generating new embeddings")
                image_embeddings = sam.get_image_embeddings(sam.model, sam.processor, image_array)
                if save_embeddings and embedding_path:
                    logger.debug(f"Saving embeddings to: {embedding_path}")
                    np.save(embedding_path, image_embeddings.cpu().numpy())

            # Get masks
            logger.debug("Getting masks from SAM")
            masks, scores = sam.get_masks(
                sam.model,
                image_array,
                sam.processor,
                image_embeddings,
                input_points=transformed_prompts['points'] if transformed_prompts['points'] else None,
                input_labels=transformed_prompts['labels'] if transformed_prompts['labels'] else None,
                input_boxes=transformed_prompts['boxes'] if transformed_prompts['boxes'] else None,
            )

            if masks is None:
                logger.error("No masks generated")
                return jsonify({
                    'status': 'error',
                    'message': 'No valid masks generated'
                }), 400

            logger.debug(f"Generated {len(masks)} masks")

            # Convert to GeoJSON
            logger.debug("Converting masks to GeoJSON")
            geojson = sam.raster_to_vector(
                masks,
                scores,
                transform,
                filename="/tmp/masks.geojson"
            )

            return jsonify({
                'status': 'success',
                'features': geojson,
                'crs': source_crs
            }), 200

        except Exception as e:
            logger.error("Error in prediction:")
            logger.exception(e)
            return jsonify({
                'status': 'error',
                'message': f'Prediction error: {str(e)}'
            }), 500

    except Exception as e:
        logger.error("Unexpected error in predict():")
        logger.exception(e)
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500