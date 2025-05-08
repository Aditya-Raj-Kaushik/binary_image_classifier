import numpy as np
from PIL import Image
import io
import pickle

# Load model and scaler
with open("saved_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

class_map = {0: "Apple", 1: "Orange"}

def preprocess_image_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((64, 64))
    img_array = np.array(image).flatten() / 255.0
    scaled = scaler.transform([img_array])
    return scaled

def predict_fruit(image_file):
    # Preprocess the image bytes and scale
    input_data = preprocess_image_bytes(image_file)
    
    # Predict the class (Apple or Orange)
    prediction = model.predict(input_data)[0][0]
    return class_map[int(prediction)]
