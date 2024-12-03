curl -X POST http://127.0.0.1:3781/v1/easyearth/predict -H "Content-Type: application/json" -d '{
  "image_path": "/path/to/image.jpg",
  "embedding_path": "/path/to/embedding.bin",
  "prompts": [
    {"type": "Point", "data": {"x": 50, "y": 50}},
    {"type": "Text", "data": {"text": "Example text"}}
  ]
}'
