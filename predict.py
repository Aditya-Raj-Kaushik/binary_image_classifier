import pickle
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Loading model from pickle
with open('saved_model.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

# Map prediction to label
class_map = {0: 'Apple', 1: 'Orange'}

def preprocess_image(image_path, image_size=(64, 64)):
    img = Image.open(image_path).convert('L')
    img = img.resize(image_size)
    flat = np.array(img).flatten() / 255.0
    scaled = scaler.transform([flat])
    return scaled

# Loading and preprocessing new image
image_path = 'test_image.jpg'  
X_new = preprocess_image(image_path)


prediction = model.predict(X_new)[0][0]
print(f"Predicted Class: {class_map[int(prediction)]}")
