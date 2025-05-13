import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import pickle
import cv2

# Load the model and scaler
with open("saved_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# Class label mapping
class_map = {0: "Apple", 1: "Orange"}

def preprocess_image_bytes(image_bytes):
   
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((64, 64))
        img_array = np.array(image).flatten() / 255.0
        scaled = scaler.transform([img_array])
        return scaled
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

def get_bounding_box_image(image_bytes, predicted_class):
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        cv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter small contours 
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Only drawing bounding box for the predicted class
                if predicted_class == 0:  
                    cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                elif predicted_class == 1:  
                    cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        result_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_img)
        return result_pil
    except Exception as e:
        raise ValueError(f"Bounding box drawing failed: {e}")

def image_to_base64(pil_image):
    """Converts a PIL Image to a base64-encoded string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def predict_fruit(image_bytes, threshold=0.8):
    
    try:
        # Preprocessing the image and get the prediction probability
        input_data = preprocess_image_bytes(image_bytes)
        prob = float(model.forward(input_data)[0][0])
        predicted_class = 1 if prob >= 0.5 else 0
        confidence = prob if prob >= 0.5 else 1 - prob

        
        label = class_map[predicted_class] if confidence >= threshold else "Unknown"

        # Drawing bounding boxes only if the predicted class matches
        if label != "Unknown":
            result_pil = get_bounding_box_image(image_bytes, predicted_class)
        else:
            result_pil = Image.open(io.BytesIO(image_bytes))  # No bounding box if  unknown

        # Converting the image with bounding boxes to base64
        base64_img = image_to_base64(result_pil)

        return label, confidence, base64_img
    except Exception as e:
        return "Error", 0.0, "", str(e)
