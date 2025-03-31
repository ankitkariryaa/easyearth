# Easy Earth project with Flask, Connexion and OpenApi 3


EasyEarth Python project using Flask and Connexion

```http
https://github.com/zalando/connexion
```

## Requirements

* Docker Compose 1.21.2+ (see https://docs.docker.com/compose/install/)
* Python 3.6 +

## Run with Docker Compose in the project root directory

```bash
# building the container
cd /path/to/easyearth
sudo docker-compose build

# starting up a container
sudo docker-compose up
```

## Build the virtual environment

```bash
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
pip3 install -r requirements.txt
pip3 install -r test-requirements.txt
```

## Swagger definition

```http
http://localhost:3781/v1/easyearth/swagger.json  # TODO: not working yet...
```

## Health Check

Check if the server is running, the response should be `Server is alive`
```
curl -X GET http://127.0.0.1:3781/v1/easyearth/ping'

```

Check if the prediction endpoint is working, 
```
curl -X POST http://127.0.0.1:3781/v1/easyearth/predict -H "Content-Type: application/json" -d '{
  "image_path": "/path/to/image.jpg",
  "embedding_path": "/path/to/embedding.bin",
  "prompts": [
    {"type": "Point", "data": {"x": 50, "y": 50}},
    {"type": "Text", "data": {"text": "Example text"}}
  ]
}'

```

Test predictions with SAM model
```
curl -X POST http://127.0.0.1:3781/v1/easyearth/sam-predict -H "Content-Type: application/json" -d '{
  "image_path": "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png",
  "prompts": [
    {
      "type": "Point",
      "data": {
        "points": [[850, 1100]],
        "labels": [1]
      }
    }
  ]
}'

```

## Launch tests

```bash
source venv/bin/activate
tox
```

## How does it work?
on QGIS GUI
1. select what model to use
2. load embeddings or leave it empty
3. create the prompt 
All these info would be sent as a configuration file to the backend for prediction, then the backend would return the mask to the frontend for visualization.
The frontend would then display the mask on the map, the user can then save the mask as a raster or vector file
