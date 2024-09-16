import os
import sys
import joblib
import inspect
import importlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

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
    load_config,
    to_json,
    check_directory_name,
    load_images_and_labels
)


def create_train_test_splits(data_path_dict, usage='Training'):
    img_arr_list = []
    lbl_arr_list =[]
    for _, data_path in data_path_dict.items():
        full_data_path = os.path.join(data_path, usage)
        imgs_arr, labels_arr = load_images_and_labels(full_data_path)
        img_arr_list.append(imgs_arr)
        lbl_arr_list.append(labels_arr)
    concatenated_img_arr = np.concatenate(img_arr_list, axis=0)
    concatenated_label_arr = np.concatenate(lbl_arr_list, axis=0)
    print(f'Shape of X array is: {concatenated_img_arr.shape}.')
    print(f'Shape of y array is: {concatenated_label_arr.shape}.')
    return concatenated_img_arr, concatenated_label_arr

def instantiate_model(model_config):
    'Function to create model instances from the configuration'
    module = importlib.import_module(model_config['module'])
    model_class = getattr(module, model_config['class'])
    model= model_class(**model_config['params'])
    return model

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
        model.save_model(json_path)
    elif 'lightgbm' in module_path:
        txt_path = filename.replace('.pkl', '.txt')
        model.booster_.save_model(txt_path)
    else:
        print(f'Module path: {module_path}.')
        raise ValueError("Unsupported library")



def train_and_evaluate_model(model_name, model_params, X_train, y_train, labels, output_dir=os.path.join('models', 'vectorized')):
    """Train the model, evaluate it, and save the metrics."""
    model  = instantiate_model(model_params)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_train)
    report = classification_report(y_train, y_pred, target_names=labels, output_dict=True)
    matrix = confusion_matrix(y_train, y_pred)
    
    # Save metrics
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    report_path = os.path.join(model_dir, 'classification_report.json')
    to_json(report, report_path)

    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(os.path.join(model_dir, 'confusion_matrix.csv'))
    
    # Extract summary metrics for the DataFrame
    accuracy = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }

def main(data_path, labels_path, output_dir, models):
    """Main function to load data, train models, and generate summary."""
    # Load the data and labels once
    X_train, y_train, labels = load_data_and_labels(data_path, labels_path)

    # List to store the summary metrics for all models
    summary_metrics = []

    # Train, evaluate each model, and save the metrics
    for model_name, model_instance in models.items():
        print(f"Training and evaluating {model_name}")
        metrics = train_and_evaluate_model(model_instance, model_name, X_train, y_train, labels, output_dir)
        summary_metrics.append(metrics)

    # Create a summary DataFrame
    summary_df = pd.DataFrame(summary_metrics)

    # Save the summary DataFrame to a CSV file
    summary_df.to_csv(os.path.join(output_dir, 'model_summary.csv'), index=False)

    # Display the summary DataFrame
    print(summary_df)

if __name__ == "__main__":
    # Ensure current working directory
    main_dir = 'EmotionFaceClassifier'
    check_directory_name(main_dir)

    # Load input data paths from JSON
    input_data_paths = load_config('./configs/input_mappings.json')
    input_paths = input_data_paths["img_directories"]

    X_train, y_train = create_train_test_splits(input_paths, usage='Training')

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Set output paths
    intermediate_data_path = os.path.join('data', 'intermediate_data')
    os.makedirs(intermediate_data_path, exist_ok=True)

    train_imgs_path = os.path.join(intermediate_data_path, 'combined_train_images.npy')
    train_labels_path = os.path.join(intermediate_data_path, 'combined_train_labels.npy')
    train_labels_encode_path = os.path.join(intermediate_data_path, 'combined_train_labels_encoded.npy')

    # Save the combined datasets
    np.save(train_imgs_path, X_train)
    np.save(train_labels_path, y_train)
    np.save(train_labels_encode_path, y_train_encoded)

    # Load model params from JSON
    vectorized_models = load_config('./configs/vectorized_models.json')
    print(vectorized_models.keys())

    # Execute the main function
    main(data_path, labels_path, output_dir, models)
