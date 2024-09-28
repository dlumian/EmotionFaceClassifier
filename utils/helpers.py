import os
import json
import cv2
import numpy as np

def load_images_and_labels(folder):
    images = []
    labels = []
    for emotion in os.listdir(folder):
        # Skip hidden files and directories (those that start with a dot)
        if emotion.startswith('.'):
            continue
        label_folder = os.path.join(folder, emotion)
        for filename in os.listdir(label_folder):
            # Skip hidden files
            if filename.startswith('.'):
                continue
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = cv2.imread(os.path.join(label_folder, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = img.flatten()  # Flatten the image
                    img = img / 255.0  # Normalize
                    images.append(img)
                    labels.append(emotion)
    return np.array(images), np.array(labels)