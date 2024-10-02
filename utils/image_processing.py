import os
import cv2
import glob
import numpy as np
import pandas as pd


def create_image_dataframe(base_path, search_structure):
    """
    Create a DataFrame with metadata for images in a hierarchical directory structure.

    Parameters:
    - base_path (str): The base directory path where the dataset is located.
    - search_structure (dict): A dictionary where each key is a column label (e.g., 'usage', 'emotion')
                               and the value is a list of directory names to search for that column.

    Returns:
    - df (pd.DataFrame): DataFrame containing full paths and metadata columns.
    """
    data = []

    def recursive_search(current_path, structure_keys, metadata):
        if len(structure_keys) == 0:
            image_paths = glob.glob(os.path.join(current_path, '*.jpg'))
            for img_path in image_paths:
                metadata_copy = metadata.copy()
                metadata_copy['full_path'] = img_path
                data.append(metadata_copy)
        else:
            current_key = structure_keys[0]
            sub_dirs = search_structure[current_key]
            for sub_dir in sub_dirs:
                next_path = os.path.join(current_path, sub_dir)
                if os.path.exists(next_path):
                    metadata[current_key] = sub_dir
                    recursive_search(next_path, structure_keys[1:], metadata)

    recursive_search(base_path, list(search_structure.keys()), {})
    return pd.DataFrame(data)

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