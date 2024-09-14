import os
import json
import pandas as pd
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
    load_config
)

# def load_data_and_labels(data_path, labels_path):
#     """Load the training data and labels from the specified paths."""
#     # Load X and y data
#     X_train = ...  # Load your X_train data here
#     y_train = ...  # Load your y_train data here
    
#     # Load labels from JSON
#     with open(labels_path, 'r') as f:
#         labels = json.load(f)
    
#     return X_train, y_train, labels

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
    # Define paths
    data_path = 'path/to/data'
    labels_path = 'path/to/labels.json'
    output_dir = 'path/to/save/model/metrics'


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
