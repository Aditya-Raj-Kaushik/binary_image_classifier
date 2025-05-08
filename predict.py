import numpy as np
from PIL import Image
import io
import pickle


with open("saved_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

class_map = {0: "Apple", 1: "Orange"}

def preprocess_image_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((64, 64))
    img_array = np.array(image).flatten() / 255.0
    scaled = scaler.transform([img_array])
    return scaled

def predict_fruit(image_file, threshold=0.8):
    input_data = preprocess_image_bytes(image_file)
    
    prob = float(model.forward(input_data)[0][0])
    
    confidence = prob if prob >= 0.5 else 1 - prob
    predicted_class = 1 if prob >= 0.5 else 0

    if confidence >= threshold:
        return class_map[predicted_class], confidence
    else:
        return "Unknown", confidence
