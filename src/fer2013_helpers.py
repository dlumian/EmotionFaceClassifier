import os
import numpy as np
import pandas as pd
from PIL import Image

def create_img(row):
    base_path = os.path.join('data', 'fer2013')
    usage = row['usage']
    emot = row['emotion']
    dir_path = os.path.join(base_path, usage, emot)
    os.makedirs(dir_path, exist_ok=True)

    f_name = f"{emot}-{row['emo_count_id']}.jpg"

    # Convert the array to grayscale
    arr = row['image']
    tmp_img = Image.fromarray(arr.astype('uint8'), 'L')

    # Combined path
    final_path = os.path.join(dir_path, f_name)
    # Save the grayscale image as a JPG
    tmp_img.save(final_path)