import os
import logging
import numpy as np
from copy import deepcopy
from datetime import datetime
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from datascifuncs.tidbit_tools import print_json, write_json
from .preprocessing import plot_face_matrix
from .analysis_tools import instantiate_model, normalize_data, timeit

def generate_analysis_paths(analysis_config):
    model_type = analysis_config['class']
    total_components = analysis_config['total_components']
    normalizer = analysis_config['normalization']

    base_dir = 'models/unsupervised'
    dir_name = f"{model_type.lower()}_{normalizer}_{total_components}"
    result_dir = os.path.join(base_dir, dir_name)

    log_filename = f'analysis_log.log'
    details_filename = f'analysis_details.json'
    data_filename = f'analysis_data.npz'
    component_matrix_filename = 'component_averages_matrix.png'
    reconstruction_matrix_filename = 'reconstruction_matrix.png'

    log_path = os.path.join(result_dir, log_filename)
    json_path = os.path.join(result_dir, details_filename)
    npz_path = os.path.join(result_dir, data_filename)
    component_path = os.path.join(result_dir, component_matrix_filename)
    recons_path = os.path.join(result_dir, reconstruction_matrix_filename)

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    paths_dict = {
        'base_dir': base_dir,
        'result_dir': result_dir,
        'log_path': log_path,
        'json_path': json_path,
        'npz_path': npz_path,
        'component_path': component_path,
        'recons_path': recons_path
    }

    # Add the paths dictionary to the analysis_config
    analysis_config['paths'] = paths_dict
    return analysis_config

def reconstruct_images(model, X, n_components):
    X_copy = deepcopy(X)
    subset_model = deepcopy(model)
    # Select the top n_components principal components
    subset_model.components_[n_components:, :]=0
    # Reconstruct the images using the selected components
    logging.info(f'X copy shape: {X_copy.shape}.')
    logging.info(f'Subset model components: {subset_model.components_.shape}.')
    zeroed_components = np.all(subset_model.components_ == 0, axis=1).sum()
    logging.info(f'Subset model all 0 components: {zeroed_components}.')
    subset_X_copy = subset_model.transform(X_copy)
    logging.info(f'Transformed X shape: {subset_X_copy.shape}.')
    reconstructed_images = subset_model.inverse_transform(subset_X_copy)
    return reconstructed_images

def calculate_metrics(original, reconstruction):
    mse = mean_squared_error(original, reconstruction)
    psnr = 10 * np.log10((255**2) / mse)
    ssim_value = ssim(original, reconstruction, data_range=original.max() - original.min(), multichannel=True)
    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_value
    }

def run_component_reconstruction(model, recon_components):
    # Create average image of components
    selected_components = model.components_[:recon_components,:]
    avg_selected_components = np.mean(selected_components, axis=0)
    return np.reshape(avg_selected_components, (48,48))

def run_X_reconstruction(model, scaler, original_X, recon_components):
    # Reconstruct all images
    reconstructed_X = reconstruct_images(model, original_X, recon_components)
    if scaler:
        reconstructed_X = scaler.inverse_transform(reconstructed_X)   
    reconstructed_X_image = np.reshape(np.mean(reconstructed_X, axis=0), (48, 48))
    reconstructed_X_metrics = calculate_metrics(original_X, reconstructed_X)
    return reconstructed_X_image, reconstructed_X_metrics

def run_category_analysis(category, model, scaler, X_normalized, y, component_values):

    model_category = clone(model)
    total_components = model_category.n_components

    valid_components_values = []
    component_avg_images = []
    avg_recon_images = []
    recon_metrics = []

    if category == 'Overall':
        X_category = deepcopy(X_normalized)
    else:
        X_category = deepcopy(X_normalized[y == category])

    features_category = model_category.fit_transform(X_category)
    print(f'Running category: {category}.')
    print(f'Shape of features is: {features_category.shape}.')

    for recon_components in component_values:
        if recon_components > total_components:
            raise ValueError(f"Requested components ({recon_components}) exceed total_components ({total_components}).")
        else:
            valid_components_values.append(recon_components)
            component_recon = run_component_reconstruction(model=model_category, recon_components=recon_components)
            component_avg_images.append(component_recon)

            X_avg_recon, X_recon_metrics = run_X_reconstruction(
                model=model_category, scaler=scaler, original_X=X_category, recon_components=recon_components)
            avg_recon_images.append(X_avg_recon)
            recon_metrics.append(X_recon_metrics)

    category_results={
        'category': category,
        'valid_component_values': valid_components_values,
        'component_reconstructions': component_avg_images,
        'avg_recon_images' : avg_recon_images,
        'metrics_recon_dicts' : recon_metrics
    }
    print(f"Analysis complete for category: {category}")
    return category_results

def save_analysis_data(data_list, output_file):
    data_dict = {}
    for data in data_list:
        category = data['category']
        valid_component_values = np.array(data['valid_component_values'])
        component_reconstructions = np.array(data['component_reconstructions'])
        avg_recon_images = np.array(data['avg_recon_images'])
        metrics_recon_dicts = data['metrics_recon_dicts']

        data_dict[category] = {
            'valid_component_values': valid_component_values,
            'component_reconstructions': component_reconstructions,
            'avg_recon_images': avg_recon_images,
            'metrics_recon_dicts': metrics_recon_dicts
        }

    np.savez_compressed(output_file, **data_dict)

def extract_image_dict(final_results, data_key):
    image_dict = {}
    for results in final_results:
        image_dict[results['category']] = results[data_key]
    return image_dict    

@timeit
def run_single_analysis(X, y, analysis_config):
    json_path = analysis_config['paths']['json_path']
    npz_path = analysis_config['paths']['npz_path']

    # Check if both json_path and npz_path exist
    if os.path.exists(json_path) and os.path.exists(npz_path):
        print(f"Analysis already exists. Skipping analysis.")
        logging.info(f"Analysis already exists. Skipping analysis.")
        return None

    unique_categories = np.unique(y)
    unique_categories = np.insert(unique_categories, 0, 'Overall')
    component_values = analysis_config['components_for_reconstruction']
    total_components = analysis_config['total_components']

    log_path = analysis_config['paths']['log_path']

    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        force=True) 
    logging.info(f"Analysis state time: {datetime.now()}.")

    print(f"Analysis settings:")
    print_json(analysis_config)

    analysis_config['params']['n_components'] = total_components
    model = instantiate_model(analysis_config)
    X_normalized, scaler = normalize_data(X, analysis_config['normalization'])

    final_results = []

    for category in unique_categories:
        category_results = run_category_analysis(
            category=category, model=model, scaler=scaler,
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
    recons_dict = extract_image_dict(final_results=final_results, data_key='avg_recon_images')

    component_path = analysis_config['paths']['component_path']
    reconstruction_path = analysis_config['paths']['recons_path']

    component_title = f'{model_type} Component Averages'
    reconstructions_title = f'{model_type} Reconstructed Image Averages'

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

    plot_face_matrix(
        image_dict=recons_dict,
        row_labels=row_labels,
        group_colors=analysis_config['color_map'],
        save_path=reconstruction_path,
        main_title=reconstructions_title,
        box_text=box_text
    )
    print(f"Saved matrix image of component averages to: {reconstruction_path}.")
    logging.info(f"Component image matrix saved to {reconstruction_path}.")

    logging.info(f"Analysis end time: {datetime.now()}.")
    print(f'Current analysis complete.')
    return final_results
