# Easy Earth project with Flask, Connexion and OpenApi 3


EasyEarth - QGIS plugin powered by huggingface, Flask and Connexion 

Check the demo [here](https://drive.google.com/file/d/1AShHsXkYoBj4zltAGkdnzEfKp2GSFFeS/view)


![image](https://github.com/user-attachments/assets/1447e21f-6cb2-4917-8d06-ba9960b78d87)


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

```bash
cd easyearth_plugin  # go to the directory where docker-compose.yml is located
sudo docker-compose build  # build the container, can be skipped if already built
sudo docker-compose up  # start the container
```

Check if the server is running, the response should be `Server is alive`
```bash
curl -X GET http://127.0.0.1:3781/v1/easyearth/ping'

```

Test predictions with prompts using SAM model
```bash
curl -X POST http://127.0.0.1:3781/v1/easyearth/sam-predict -H "Content-Type: application/json" -d '{
  "image_path": "/usr/src/app/user/DJI_0108.JPG",
  "embedding_path": "/usr/src/app/user/embeddings/DJI_0108.pt",
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
Test predictions with no prompts using other segmentation models
```bash
curl -X POST http://127.0.0.1:3781/v1/easyearth/segment-predict -H "Content-Type: application/json" -d '{
  "image_path": "/usr/src/app/user/DJI_0108.JPG",
  "prompts": []            
}'


```

## Launch tests

```bash
source venv/bin/activate
tox
```
