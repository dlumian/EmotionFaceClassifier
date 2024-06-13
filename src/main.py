import os
import json
import inspect
import joblib
import importlib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

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

def str_to_array(pixel_str):
    'Convert string pixel data to numpy arrays'
    clean_strs = pixel_str.replace(',', '').split()
    return np.array(clean_strs, dtype=np.uint8)

def load_config(file_path):
    ' Opens and loads json file'
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def create_models(model_config):
    'Function to create model instances from the configuration'
    models = {}
    for model_name, model_info in model_config.items():
        module = importlib.import_module(model_info['module'])
        model_class = getattr(module, model_info['class'])
        models[model_name] = model_class(**model_info['params'])
    return models

def get_classification_metrics(y_true, y_pred):
    'calculates metrics on classifier performance'
    metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
    return metrics

def get_module_path(class_object):
    """Get the module path from a class object.

    Args:
        class_object: The class object.

    Returns:
        The module path, or None if the module path cannot be determined.
    """

    module = inspect.getmodule(class_object)
    if module is None:
        return None

    return module.__file__

def save_model(model, filename):
    'Saves model based on type'
    module_path = get_module_path(model)

    if '_logistic' in module_path:
        joblib.dump(model, filename)
    elif 'tree/_classes' in module_path:
        joblib.dump(model, filename)
    elif 'ensemble/_forest' in module_path:
        joblib.dump(model, filename)
    elif 'ensemble/_gb' in module_path:
        joblib.dump(model, filename)
    elif 'xgboost' in module_path:
        json_path = filename.replace('.pkl', '.json')
        model.save_model(filename)
    elif 'lightgbm' in module_path:
        model.save_model(filename)
    else:
        print(f'Module path: {module_path}.')
        raise ValueError("Unsupported library")

def to_json(data, file_path):
    'Writes data to filepath with nice formatting'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)