import os
import numpy as np
from PIL import Image

def convert_pixels_to_array(pixels):
    """
    Convert a space-separated string of pixels into a 2-D numpy array.

    Parameters:
        pixels (str): A string containing space-separated pixel values.

    Returns:
        numpy.ndarray: A 2-D numpy array with shape (48, 48) and dtype 'uint8'.
    """
    # Convert the space-separated string into a list of integers
    pixel_values = [int(x) for x in pixels.split(' ')]

    # Reshape the list into a 48x48 2-D array
    array = np.array(pixel_values).reshape(48, 48)

    # Ensure the array is of type 'uint8'
    'Reshape pixel arrays into 2-D format from flat vector'
    pixel_values = [int(x) for x in pixels.split(' ')]
    return np.array(pixel_values, dtype='uint8').reshape(48, 48)

def save_image(row):
    # Construct directory path and ensure it exists
    """
    Save an image to a specified directory path based on its emotion and usage.

    Parameters:
        row (pandas.Series): A row containing image data and metadata, 
                             with keys 'emotion', 'usage', 'emo_count_id', and 'image'.

    Returns:
        str: The file path where the image was saved.
    """
    
    base_dir = 'data'
    emotion_label = row['emotion']
    directory_path = os.path.join(base_dir, row['usage'], emotion_label)
    os.makedirs(directory_path, exist_ok=True)
    
    # Construct the final image file name and path
    file_name = f"{emotion_label}-{row['emo_count_id']}.jpg"
    image_path = os.path.join(directory_path, file_name)
    
    # Convert the image array to grayscale and save
    image = Image.fromarray(row['image'].astype('uint8'), 'L')
    image.save(image_path)
    
    return image_path
