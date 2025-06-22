from PIL import Image
import numpy as np

def preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).convert("RGB")
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)
