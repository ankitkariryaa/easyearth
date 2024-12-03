# Easy Earth project with Flask, Connexion and OpenApi 3


EasyEarth Python project using Flask and Connexion

```http
https://github.com/zalando/connexion
```

## Requirements

* Docker Compose 1.21.2+
* Python 3.6 +

## Run with Docker Compose

```bash
# building the container
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
http://localhost:3781/v1/easyearth/swagger.json
```

## Health Check

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

## Launch tests

```bash
source venv/bin/activate
tox
```
