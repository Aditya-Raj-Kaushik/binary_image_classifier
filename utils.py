from PIL import Image, ImageEnhance
import numpy as np
import os
import random

def augment_image(img):
    # horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # rotation between -20 to 20 degrees
    angle = random.uniform(-20, 20)
    img = img.rotate(angle)

    # brightness adjustment 
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random zoom 
    if random.random() > 0.5:
        w, h = img.size
        scale = random.uniform(0.8, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h))

    return img

def load_images(data_path, image_size=(64, 64), augment=True, augment_factor=4):
    X = []
    y = []

    class_map = {'Apple': 0, 'Orange': 1}

    for folder in ['Apple', 'Orange']:
        folder_path = os.path.join(data_path, folder)
        label = class_map[folder]

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(image_size)
                X.append(np.array(img).flatten() / 255.0)
                y.append(label)

                # Apply augmentations
                if augment:
                    for _ in range(augment_factor):
                        aug_img = augment_image(img)
                        X.append(np.array(aug_img).flatten() / 255.0)
                        y.append(label)

            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    return np.array(X), np.array(y).reshape(-1, 1)

