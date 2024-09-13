import os
import numpy as np
import pandas as pd
import pickle
import importlib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

import json
import importlib

# Load the JSON file
with open('models.json', 'r') as file:
    models_dict = json.load(file)

# Function to save the trained model to a pickle file
def save_model(model, filename):    
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

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

# # Iterate through the models in the JSON file and instantiate them
# models = {}
# for model_name, params in models_dict.items():
#     model = instantiate_model(model_name, params)
#     models[model_name] = model


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