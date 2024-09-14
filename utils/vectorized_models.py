import os
import json
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
    check_directory_name,
    load_images_and_labels
)


def create_train_test_splits(data_path_dict, usage='Training'):
    img_df_list = []
    lbl_df_list =[]
    for _, data_path in data_path_dict.items():
        full_data_path = os.path.join(data_path, usage)
        imgs_df, labels_df = load_images_and_labels(full_data_path)
        img_df_list.append(imgs_df)
        lbl_df_list.append(labels_df)
    concatenated_img_df = pd.concat(img_df_list, ignore_index=True)
    concatenated_label_df = pd.concat(lbl_df_list, ignore_index=True)
    return concatenated_img_df, concatenated_label_df

def train_and_evaluate_model(model_params, X_train, y_train, labels):
    """Train the model, evaluate it, and save the metrics."""
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_train)
    report = classification_report(y_train, y_pred, target_names=labels, output_dict=True)
    matrix = confusion_matrix(y_train, y_pred)
    
    # Save metrics
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
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

    # Define your models
    models = {
        'model_1': model_1_instance,
        'model_2': model_2_instance,
        # Add more models as needed
    }

    # Execute the main function
    main(data_path, labels_path, output_dir, models)
