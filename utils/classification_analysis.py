import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.base import clone

def load_model_info(json_path):
    with open(json_path, 'r') as file:
        model_info = json.load(file)
    return model_info

def drop_category(X, y, category_to_drop):
    mask = y != category_to_drop
    return X[mask], y[mask]

def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def fit_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def collect_metrics(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return report, matrix

def run_classification_analysis(json_path, X_train, X_test, y_train, y_test, category_to_drop=None, normalize=True):
    model_info = load_model_info(json_path)
    model = clone(model_info['model'])

    if category_to_drop:
        X_train, y_train = drop_category(X_train, y_train, category_to_drop)
        X_test, y_test = drop_category(X_test, y_test, category_to_drop)

    if normalize:
        X_train, X_test, scaler = normalize_data(X_train, X_test)

    y_pred = fit_and_predict(model, X_train, y_train, X_test)
    report, matrix = collect_metrics(y_test, y_pred)

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

# Example usage
if __name__ == "__main__":
    # Load your data here
    # X_train, X_test, y_train, y_test = ...

    json_path = 'path_to_your_model_info.json'
    run_classification_analysis(json_path, X_train, X_test, y_train, y_test, category_to_drop='Disgust')