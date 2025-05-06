from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import rasterio
import pyproj
import torch  # Add this for CRS transformation
from easyearth.models.segmentation import Segmentation
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
log_file = os.path.join(LOG_DIR, f'segment_server_{datetime.now().strftime("%Y%m%d")}.log')

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

def predict():
    """Handle prediction request for segmentation models from hugging face."""
    try:
        # Get the image data from the request
        data = request.get_json()

        # Validate and convert image path
        image_path = data.get('image_path')
        if not image_path:
            return jsonify({
                'status': 'error',
                'message': 'image_path is required'
            }), 400
        if not os.path.exists(image_path):
            return jsonify({
                'status': 'error',
                'message': f'Image file not found at path: {image_path}.'
            }), 404

        # Load image with detailed error handling
        try:
            if image_path.startswith(('http://', 'https://')):
                # Handle URL images
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw).convert('RGB')
                image_array = np.array(image)
                transform = None
                source_crs = None
            elif image_path.endswith('.tif'):
                # Handle local GeoTIFF files
                with rasterio.open(image_path) as src:
                    transform = src.transform
                    source_crs = src.crs.to_string()
                    image_array = src.read()
                    image_array = np.transpose(image_array, (1, 2, 0))
            else:
                # Handle local regular images
                logger.debug(f"Loading local image from: {image_path}")
                image = Image.open(image_path).convert('RGB')
                image_array = np.array(image)
                transform = None
                source_crs = None

            # Ensure image is in the correct format
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] > 3:
                image_array = image_array[:, :, :3]

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image from URL: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to download image: {str(e)}'
            }), 500
        except Exception as e:
            logger.error("Error loading image:", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f'Failed to load image: {str(e)}'
            }), 500

        # Process prompts
        try:
            prompts = data.get('prompts', [])
            
            # Initialize SAM
            logger.debug("Initializing Segmentation model")
            segformer = Segmentation()

            # Get masks
            masks = segformer.get_masks(image_array)

            if masks is None:
                return jsonify({
                    'status': 'error',
                    'message': 'No valid masks generated'
                }), 400

            # Convert to GeoJSON
            geojson = segformer.raster_to_vector(
                masks,
                transform,
                filename=f"{PLUGIN_DIR}/tmp/predict-segment_{os.path.basename(image_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"
            )

            return jsonify({
                'status': 'success',
                'features': geojson,
                'crs': source_crs
            }), 200

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Prediction error: {str(e)}'
            }), 500

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500