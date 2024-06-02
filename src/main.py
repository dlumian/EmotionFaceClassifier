import os
import ast
import json
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

def check_directory_name(target_name) -> bool:
    """
    Check if the current directory name matches the target_name.
    If not, move up a directory and repeat the check.
    
    Args:
        target_name (str): The directory name to match.
        
    Returns:
        bool: True if the current directory name matches the target_name, False otherwise.
    """
    # Get the current directory path
    current_dir = os.getcwd()
    
    # Extract the directory name from the path
    current_dir_name = os.path.basename(current_dir)
    
    # Check if the current directory name matches the target_name
    if current_dir_name == target_name:
        print(f'Directory set to {current_dir}, matches target dir sting {target_name}.')
        return True
    else:
        # Move up a directory
        os.chdir('..')
        # Check if we have reached the root directory
        if os.getcwd() == current_dir:
            return False
        # Recursively call the function to check the parent directory
        return check_directory_name(target_name)


def convert_pixels_to_array(pixels):
    'Reshape pixel arrays into correct format for FER2013 csv input'
    array = np.array([int(x) for x in pixels.split(' ')]).reshape(48,48)
    array = np.array(array, dtype='uint8')
    return array

# Convert string pixel data to numpy arrays
def str_to_array(pixel_str):
    return np.array(ast.literal_eval(pixel_str), dtype=np.uint8)

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

