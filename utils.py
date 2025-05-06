import os
from PIL import Image
import numpy as np

def load_images(data_path, image_size=(64, 64)):
    X = []  # This will hold all the image data
    y = []  # This will hold the corresponding labels (0 for Apple, 1 for Orange)
    
    # Telling the code what label to give to each folder
    class_map = {'Apple': 0, 'Orange': 1}
    
    # Go through each folder (Apple and Orange)
    for folder in ['Apple', 'Orange']:
        folder_path = os.path.join(data_path, folder)  # Create the full path to the folder
        label = class_map[folder]  # Get the number that represents the class (0 or 1)
        
        # Now, look at each image file inside that folder
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)  # Full path to the image file
            try:
                # Open the image and convert it to grayscale (black & white)
                img = Image.open(img_path).convert('L')
                # Resize the image to 64x64 pixels so all images are the same size
                img = img.resize(image_size)
                # Turn the image into a 1D array and scale pixel values between 0 and 1
                X.append(np.array(img).flatten() / 255.0)
                # Save the label (either 0 or 1)
                y.append(label)
            except Exception as e:
                # If the image can't be read or processed, skip it and print the reason
                print(f"Skipping {img_path}: {e}")

    # Convert our lists into proper NumPy arrays that the neural network can understand
    return np.array(X), np.array(y).reshape(-1, 1)
