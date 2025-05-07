import logging
import os
import sys
from datetime import datetime

import requests

from easyearth import logger
import random
from flask import jsonify

# TODO: combine sam_controller and segmentation_controller into one controller

# TODO: standardize the logging function across the project
def setup_logging(name="predict-controller"):
    """
    Set up logging for the application
    """
    # Get environment variables for directories
    APP_DIR = os.environ.get('APP_DIR', '/usr/src/app')
    LOG_DIR = os.environ.get('LOG_DIR', os.path.join(APP_DIR, 'logs'))
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # 'a' for append mode
            logging.StreamHandler(sys.stdout)  # This will print to Docker logs
        ]
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.info(f"Logging to {log_file}")
    return logger

def verify_image_path(image_path):
    """Verify the image path and check if it is a valid URL or local file. Remember to convert the image path the path in the docker container"""
    # TODO: to complete
    if image_path.startswith(('http://', 'https://')):
        # Handle URL images
        try:
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image from URL: {str(e)}")
            return False
    else:
        # Handle local files
        if os.path.isfile(image_path):
            return True
        else:
            logger.error(f"Invalid image path: {image_path}")
            return False

def verify_model_path(model_path):
    """"Verify the model path and check if it is a valid model path from hugging face"""
    # TODO: to complete
    raise NotImplementedError("Model path verification is not implemented yet.")


def generate_random_predictions():
    types = ["Class", "Points", "Polygons"]
    predictions = []
    for _ in range(random.randint(1, 5)):  # Random number of predictions
        prediction_type = random.choice(types)
        if prediction_type == "Class":
            predictions.append({"type": "Class", "data": {"label": "example_class", "confidence": random.random()}})
        elif prediction_type == "Points":
            predictions.append({"type": "Points", "data": [{"x": random.randint(0, 100), "y": random.randint(0, 100)} for _ in range(random.randint(1, 5))]})
        elif prediction_type == "Polygons":
            predictions.append({"type": "Polygons", "data": [[{"x": random.randint(0, 100), "y": random.randint(0, 100)} for _ in range(4)] for _ in range(random.randint(1, 3))]})
    return predictions

def predict():
    response = {
        "model_description": "Mock Image Analysis Model v1.0",
        "predictions": generate_random_predictions()
    }
    return jsonify(response), 200

def ping():
    return jsonify({"message": "Server is alive"}), 200