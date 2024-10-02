import os
import numpy as np
from PIL import Image

def convert_pixels_to_array(pixels):
    'Reshape pixel arrays into 2-D format from flat vector'
    array = np.array([int(x) for x in pixels.split(' ')]).reshape(48,48)
    array = np.array(array, dtype='uint8')
    return array

def save_image(row):
    # Set path, verify dirs exist
    base_path = os.path.join('data', 'fer2013')
    emot = row['emotion']
    dir_path = os.path.join(base_path, row['usage'], emot)
    os.makedirs(dir_path, exist_ok=True)
    # Final output name
    f_name = f"{emot}-{row['emo_count_id']}.jpg"
    final_path = os.path.join(dir_path, f_name)
    # Convert and save the array to grayscale
    img = Image.fromarray(row['image'].astype('uint8'), 'L')
    img.save(final_path)
