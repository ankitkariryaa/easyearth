# Easy Earth project with Flask, Connexion and OpenApi 3


EasyEarth - QGIS plugin powered by huggingface, Flask and Connexion 

Check the demo [here](https://drive.google.com/file/d/1AShHsXkYoBj4zltAGkdnzEfKp2GSFFeS/view)


![image](https://github.com/user-attachments/assets/1447e21f-6cb2-4917-8d06-ba9960b78d87)


```http
https://github.com/zalando/connexion
```

## Folder structure
```
easyearth
├── easyearth  -> server app
├── easyearth_plugin  -> qgis plugin
│   ├── easyearth  -> server app
│   ├── ...
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

## Download the code and only need the easyearth_plugin folder
```bash
# go to your download directory
cd ~/Downloads
git clone https://github.com/YanCheng-go/easyearth.git
cp -r ./easyearth/easyearth_plugin easyearth_plugin
```

## Build the container

```bash
cd easyearth_plugin  # go to the directory where docker-compose.yml is located
sudo docker-compose up -d # build the container
```

## Install easyearth plugin on qgis
1. Open QGIS, click Settings -> User profiles -> Open Active Profile Folder -> python -> plugins
2. Copy easyearth_plugin folder to "plugins"
3. Reopen QGIS, click Plugins -> Manage and Install Plugins -> Installed -> click the check box before EasyEarth

## Run EasyEarth in QGIS
1. Click Start Docker or in the terminal run `sudo docker-compose up -d` to start the container
    ```bash
    cd easyearth_plugin  # go to the directory where docker-compose.yml is located
    # sudo docker-compose build  # build the container, can be skipped if already built
    sudo docker-compose up -d # start the container
    ```
2. Click Browse image and select an image to play with 
3. Click Start Drawing.

## Run EasyEarth outside QGIS
Start the docker container and send requests to the server using curl or any other HTTP client.
```bash
cd easyearth_plugin  # go to the directory where the repo is located
# sudo docker-compose build  # build the container, can be skipped if already built
sudo docker-compose up -d  # start the container
```

### Health Check
Check if the server is running, the response should be `Server is alive`
```bash
curl -X GET http://127.0.0.1:3781/v1/easyearth/ping'
```
### Use SAM with prompts
Send prompts to the server and get predictions from SAM model. Check the generated geojson in easyearth_plugin/user/tmp/...
```bash
curl -X POST http://127.0.0.1:3781/v1/easyearth/sam-predict -H "Content-Type: application/json" -d '{
  "image_path": "/usr/src/app/user/DJI_0108.JPG",                                                                    
  "prompts": [ 
    {
      "type": "Point",
      "data": {
        "points": [[850, 1100]]
      }
    }  
                      
  ]            
}'

```
### Use models with no prompts
Call other segmentation models with out prompt engineering
```bash
curl -X POST http://127.0.0.1:3781/v1/easyearth/segment-predict -H "Content-Type: application/json" -d '{
  "image_path": "/usr/src/app/user/DJI_0108.JPG",
  "prompts": []
}'
```

### Swagger definition
```http
http://localhost:3781/v1/easyearth/swagger.json  # TODO: not working yet...
```

## Launch tests  # TODO: to add

```bash
source venv/bin/activate
tox
```
