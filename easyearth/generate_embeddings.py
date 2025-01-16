'''Generate embeddings for a dataset using a pre-trained model.'''

import easyearth.models.geosam as geosam
from easyearth.models.geosam.image_encoder import ImageEncoder

setting_file = "/content/setting.json"
feature_dir = './'

### parse settings from the setting,json file
settings = geosam.parse_settings_file(setting_file)

### setting file not contains feature_dir, you need add it
settings.update({"feature_dir":feature_dir})

### split settings into init_settings, encode_settings
init_settings, encode_settings = geosam.split_settings(settings)

print(f"settings: {settings}")
print(f"init_settings: {init_settings}")
print(f"encode_settings: {encode_settings}")

img_encoder = ImageEncoder(**init_settings)
img_encoder.encode_image(**encode_settings)