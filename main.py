from utils import load_images
import kagglehub
import os

# Download and locate dataset
dataset_path = kagglehub.dataset_download("kipshidze/apple-vs-orange-binary-classification")

# The actual images are inside 'fruit-dataset' folder
data_path = os.path.join(dataset_path, "fruit-dataset")

# Load images
X, y = load_images(data_path, image_size=(64, 64))

print("Loaded images:", X.shape)
print("Loaded labels:", y.shape)
print("Sample labels:", y[:5])
