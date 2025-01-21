import os
import logging
from datetime import datetime
from datascifuncs.tidbit_tools import print_json, write_json
from sklearn.metrics import classification_report, confusion_matrix
from .analysis_tools import instantiate_model, normalize_data, timeit, suppress_warnings

def fit_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def collect_metrics(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return report, matrix

def run_classification_analysis(json_path, X_train, X_test, y_train, y_test, category_to_drop=None, normalize=True):

    if normalize:
        X_train, X_test, scaler = normalize_data(X_train, X_test)

    y_pred = fit_and_predict(model, X_train, y_train, X_test)
    report, matrix = collect_metrics(y_test, y_pred)

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

def generate_analysis_paths(analysis_config):
    model_type = analysis_config['class']
    normalizer = analysis_config['normalization']

    base_dir = 'models/vectorized_classification'
    dir_name = f"{model_type.lower()}_{normalizer}"
    result_dir = os.path.join(base_dir, dir_name)

    log_filename = f'run_details.log'
    details_filename = f'analysis_details.json'
    predictions_filename = f'predictions.csv'
    cm_filename = 'confusion_matrix.png'

    log_path = os.path.join(result_dir, log_filename)
    json_path = os.path.join(result_dir, details_filename)
    preds_path = os.path.join(result_dir, predictions_filename)
    component_path = os.path.join(result_dir, cm_filename)

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    paths_dict = {
        'base_dir': base_dir,
        'result_dir': result_dir,
        'log_path': log_path,
        'json_path': json_path,
        'preds_path': preds_path,
        'component_path': component_path
    }

    analysis_config['paths'] = paths_dict
    return analysis_config

@suppress_warnings
@timeit
def run_single_analysis(X, y, analysis_config):
    json_path = analysis_config['paths']['json_path']
    preds_path = analysis_config['paths']['preds_path']
    log_path = analysis_config['paths']['log_path']

    # Check if both json_path and npz_path exist
    if os.path.exists(json_path) and os.path.exists(preds_path):
        print(f"Analysis already exists. Skipping analysis.")
        logging.info(f"Analysis already exists. Skipping analysis.")
        print('\n\n')
        return None

    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        force=True) 
    logging.info(f"Analysis start time: {datetime.now()}.")

    print(f"Analysis settings:")
    print_json(analysis_config)

    model = instantiate_model(analysis_config)
    X_normalized, scaler = normalize_data(X, analysis_config['normalization'])




    final_results = []

    for category in unique_categories:
        category_results = run_category_analysis(
            category=category, model=model,
            X_normalized=X_normalized, y=y, 
            component_values=component_values
        )
        final_results.append(category_results)

    write_json(analysis_config, json_path)
    print(f"Analysis settings saved to {json_path}.")

    save_analysis_data(final_results, npz_path)
    print(f"Analysis data saved to {npz_path}.")

    logging.info(f"Analysis settings saved to {json_path}.")
    logging.info(f"Analysis data saved to {npz_path}.")

    model_type = analysis_config['class']
    normalizer = analysis_config['normalization']

    row_labels = {}
    row_labels['Components'] = analysis_config['components_for_reconstruction']

    component_dict = extract_image_dict(final_results=final_results, data_key='component_reconstructions')
    component_path = analysis_config['paths']['component_path']
    component_title = f'{model_type} Component Averages'

    box_text = f'Normalization: {normalizer}.\nTotal Components: {total_components}.'

    plot_face_matrix(
        image_dict=component_dict,
        row_labels=row_labels,
        group_colors=analysis_config['color_map'],
        save_path=component_path,
        main_title=component_title,
        box_text=box_text
    )
    print(f"Saved matrix image of component averages to: {component_path}.")
    logging.info(f"Component image matrix saved to {component_path}.")

    logging.info(f"Analysis end time: {datetime.now()}.")
    print(f'Current analysis complete.')
    return final_results
# # Example usage
# if __name__ == "__main__":
#     # Load your data here
#     # X_train, X_test, y_train, y_test = ...

#     json_path = 'path_to_your_model_info.json'
#     run_classification_analysis(json_path, X_train, X_test, y_train, y_test, category_to_drop='Disgust')