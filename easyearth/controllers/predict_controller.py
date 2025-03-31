from easyearth import logger
import random
from flask import jsonify


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

def predict(request):
    response = {
        "model_description": "Mock Image Analysis Model v1.0",
        "predictions": generate_random_predictions()
    }
    return jsonify(response), 200

def ping(request):
    return jsonify({"message": "Server is alive"}), 200