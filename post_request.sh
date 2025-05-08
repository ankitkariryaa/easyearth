curl -X POST http://127.0.0.1:3781/v1/easyearth/sam-predict -H "Content-Type: application/json" -d '{
  "image_path": "/usr/src/app/user/DJI_0108.JPG",
  "embedding_path": "/usr/src/app/user/embeddings/DJI_0108.pt",  # if empty, the code will generate embeddings first
  "prompts": [
    {
      "type": "Point",
      "data": {
        "points": [[850, 1100]],
      }
    }

  ]
}'