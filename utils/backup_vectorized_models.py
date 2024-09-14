import os
import sys
import joblib
import pickle
import inspect
import importlib

import numpy as np
import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier 
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

module_name = 'utils'

try:
    __import__(module_name)
except ImportError:
    # If the module cannot be imported, append the directory to sys.path
    module_path = './../'
    if module_path not in sys.path:
        sys.path.append(module_path)
    try:
        __import__(module_name)
        print(f"Successfully imported {module_name} after appending {module_path} to sys.path.")
    except ImportError:
        print(f"Failed to import {module_name} even after appending {module_path} to sys.path.")
else:
    print(f"{module_name} is already available.")

from utils.helpers import (
    load_config
)

# Function to dynamically import and instantiate models
def instantiate_model(model_name, params):
    if model_name in ['RandomForestClassifier', 'SVC']:
        module = importlib.import_module('sklearn.ensemble' if model_name == 'RandomForestClassifier' else 'sklearn.svm')
    elif model_name.startswith('LGBM'):
        module = importlib.import_module('lightgbm')
    elif model_name.startswith('XGB'):
        module = importlib.import_module('xgboost')
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    model_class = getattr(module, model_name)
    
    # Instantiate the model with the parameters
    model = model_class(**params)
    return model

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

def get_classification_metrics(y_true, y_pred):
    'calculates metrics on classifier performance'
    metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
    return metrics

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
        model.save_model(json_path)
    elif 'lightgbm' in module_path:
        txt_path = filename.replace('.pkl', '.txt')
        model.booster_.save_model(txt_path)
    else:
        print(f'Module path: {module_path}.')
        raise ValueError("Unsupported library")

def train_vec_model(label, model_params, X_train, y_train):
    # Create output dir
    vec_model_dir = os.path.join('models', 'vectorized')
    os.makedirs(vec_model_dir, exist_ok=True)

    print(f"Running {label} model...")    
    # Set dirs and filepaths
    model_output_dir = os.path.join(vec_model_dir, label)
    model_output_path = os.path.join(model_output_dir, 'mdl.pkl')
    metrics_ouput_path = os.path.join(model_output_dir, 'train_metrics.csv')
    cm_ouput_path = os.path.join(model_output_dir, 'train_confusion_matrix.png')

    os.makedirs(model_output_dir, exist_ok=True)
    
    # fit, save, predict
    model = instantiate_model(label, model_params)

    model.fit(X_train, y_train)
    save_model(model, filename=model_output_path)

    model_preds = model.predict(X_train)
    model_results = get_classification_metrics(y_train, model_preds)

    # Aggregate metrics and save to model dir
    pd.DataFrame(model_results, index=[0]).to_csv(metrics_ouput_path)
    model_metrics.append({label: model_results})

    # Confusion matrix
    int_labels = [int(i) for i in emo_dict.keys()]
    str_labels = [i for i in emo_dict.values()]
    
    cm_disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_train,
        y_pred=model_preds, 
        cmap='Blues',
        labels=int_labels,
        display_labels=str_labels
    )    
    plt.tight_layout()
    plt.savefig(cm_ouput_path, pad_inches=5)