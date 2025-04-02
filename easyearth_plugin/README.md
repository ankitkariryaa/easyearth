# Easy Earth project with Flask, Connexion and OpenApi 3


EasyEarth - QGIS plugin powered by huggingface, Flask and Connexion

```http
https://github.com/zalando/connexion
```

## Requirements

* Docker Compose 1.21.2+ (see https://docs.docker.com/compose/install/)
* Python 3.6 +
* QGIS

## Installing docker

For linux user, do not use snap to install docker and docker-compose  # TODO: need to see if it is an issue for windows user...
```bash
sudo apt update
sudo apt install docker-compose
```

## Install easyearth plugin on qgis
1. Open QGIS, click Settings -> User profiles -> Open Active Profile Folder -> python -> plugins
2. Copy easyearth_plugin folder to "plugins"
3. Reopen QGIS, click Plugins -> Manage and Install Plugins -> Installed -> click the check box before EasyEarth

## Run EasyEarth
1. Click Start Docker
2. Click Browse image and select an image to play with 
3. Click Start Drawing.


---------- Documentation during development --------------
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