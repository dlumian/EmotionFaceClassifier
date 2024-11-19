import os
import time
import logging
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from functools import wraps
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datascifuncs.tidbit_tools import load_json, write_json, print_json
from .analysis_tools import instantiate_model


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Log the timing
        logging.info(f'Function {func.__name__} Took {total_time:.4f} seconds')
        
        # Print the timing (optional)
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        
        return result
    return timeit_wrapper

def load_image(file_path, flatten=False):
    """Load a single image file and convert it to a numpy array."""
    with Image.open(file_path) as img:
        img = np.array(img)
        if flatten:
            img = img.flatten()
        return np.array(img)
    
def create_X_y(df, img_path_column, label_column, flatten=True):
    X = np.array([load_image(path, flatten=flatten) for path in df[img_path_column]])    
    y = df[label_column].values
    return X, y

def normalize_data(data, normalizer='none'):
    """Normalize the data using the specified method."""
    if normalizer == 'none':
        return data
    elif normalizer == 'minmax':
        return MinMaxScaler().fit_transform(data)
    elif normalizer == 'standard':
        return StandardScaler().fit_transform(data)
    else:
        raise ValueError(f"Unknown normalization method: {normalizer}")
    
def calculate_metrics(original, reconstruction):
    mse = mean_squared_error(original, reconstruction)
    psnr = 10 * np.log10((255**2) / mse)
    ssim_value = ssim(original, reconstruction, data_range=original.max() - original.min(), multichannel=True)
    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_value
    }

def feature_reconstruction(model, features, recon_components):
    partial_features = np.copy(features)
    if recon_components == partial_features.shape[1]:
        return model.inverse_transform(partial_features)
    else:
        partial_features[:,recon_components:]=0
        return model.inverse_transform(partial_features)

def compile_metrics(category_results, file_path, valid_components_values):
    metrics_df = pd.DataFrame()
    for cr in category_results:
        cat = cr.get('category')
        mets = cr.get('metrics_recon_dicts')
        if cat is not None and mets is not None:
            for comp, met_dict in zip(valid_components_values, mets):
                ser = pd.Series()
                ser['Category'] = cat
                ser['Components'] = comp
                for met, val in met_dict.items():
                    ser[met] = val
                metrics_df = pd.concat([metrics_df, ser.to_frame().T], axis=0)
    metrics_df.to_csv(file_path, index=False)
    print(f"Metrics saved to {file_path}")

@timeit
def run_single_analysis(X, y, analysis_config):
    unique_categories = np.unique(y)
    unique_categories = np.insert(unique_categories, 0, 'Overall')
    component_values = analysis_config['components_for_reconstruction']
    total_components = analysis_config['total_components']

    log_path = analysis_config['paths']['log_path']

    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        force=True) 
    logging.info(f"Analysis state time: {datetime.datetime.now()}.")

    print(f"Analysis settings:")
    print_json(analysis_config)

    analysis_config['params']['n_components'] = total_components
    model = instantiate_model(analysis_config)
    X_normalized = normalize_data(X, analysis_config['normalization'])

    all_results = []

    for category in unique_categories:
        model_category = clone(model)
        valid_components_values = []
        category_recon_avg_images = []
        metrics_recon_dicts = []

        if category == 'Overall':
            X_category = np.copy(X_normalized)
        else:
            X_category = X_normalized[y == category]

        features_category = model_category.fit_transform(X_category)
        print(f'Running category: {category}.')
        print(f'Shape of features is: {features_category.shape}.')

        for recon_components in component_values:
            if recon_components > total_components:
                raise ValueError(f"Requested components ({recon_components}) exceed total_components ({total_components}).")
            else:
                valid_components_values.append(recon_components)
                recon_images = feature_reconstruction(model_category, features_category, recon_components)
                category_recon_avg_images.append(np.mean(recon_images, axis=0))
                metrics_recon_dicts.append(calculate_metrics(X_category, recon_images))

        category_results={
            'category': category,
            'valid_component_values': valid_components_values,
            'metrics_recon_dicts' : metrics_recon_dicts,
            'avg_recon_images' : category_recon_avg_images
        }
        all_results.append(category_results)
        # category_avg_images[category]=category_recon_avg_images

    np_results = np.array(all_results, dtype=object)
    np_analysis_config = np.array(analysis_config, dtype=object)
    final_results = np.append(np_analysis_config, np_results)

    json_path = analysis_config['paths']['analysis_json']
    npz_path = analysis_config['paths']['avg_reconstructions_file']

    write_json(analysis_config, json_path)
    print(f"Analysis settings saved to {json_path}")
    np.savez_compressed(npz_path, final_results)
    print(f"Averaged reconstructions saved to {npz_path}")
    # compile_metrics(category_results, metrics_file, valid_components_values)
    logging.info(f"Analysis end time: {datetime.datetime.now()}.")

    return final_results

def generate_analysis_paths(analysis_config):
    base_dir = 'models/unsupervised'
    model_type = analysis_config['class']
    total_components = analysis_config['total_components']
    normalizer = analysis_config['normalization']

    dir_name = f"{model_type.lower()}_{normalizer}_{total_components}"
    result_dir = os.path.join(base_dir, dir_name)
    log_dir = os.path.join(result_dir, 'logs')
    log_filename = f'log_{dir_name}.log'
    log_path = os.path.join(log_dir, log_filename)
    analysis_json = os.path.join(result_dir, f"{dir_name}_info.json")
    metrics_file = os.path.join(result_dir, f"{dir_name}_metrics.csv")
    avg_reconstructions_file = os.path.join(result_dir, f"{model_type.lower()}_avg_reconstructions.npz")

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    paths_dict = {
        'base_dir': base_dir,
        'result_dir': result_dir,
        'log_dir': log_dir,
        'log_path': log_path,
        'analysis_json': analysis_json,
        'metrics_file': metrics_file,
        'avg_reconstructions_file': avg_reconstructions_file,
    }

    # Add the paths dictionary to the analysis_config
    analysis_config['paths'] = paths_dict

    return analysis_config

def load_arrays_if_file_exists(file_path):
    if os.path.exists(file_path):
        arrays = np.load(file_path)
        return dict(arrays)
    else:
        return None

def run_multiple_analyses(
        df, 
        config_path='configs/unsupervised_models_test.json', 
        analysis_types=['PCA', 'NMF', 'FastICA'], 
        img_path_column='img_path', 
        label_column='emotion'
    ):
    config = load_json(config_path)
    X, y = create_X_y(df, img_path_column, label_column)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
# def create_enhanced_matrix_plot(results_array, emotion_colors):
#     avg_categories = results['avg_categories']
#     unique_categories = results['unique_categories']
#     component_values = results['component_values']
    
#     # colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

#     print(f"Plotting for {analysis_type} with {normalizer} normalization")
#     print(f"Number of rows (component values): {len(component_values)}")
#     print(f"Number of columns (categories): {len(unique_categories)}")

#     n_rows = len(component_values)
#     n_cols = len(unique_categories)
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
#     fig.suptitle(f'{analysis_type} Reconstructions', fontsize=30, fontweight='bold', y=1.05)
    
#     # Ensure axes is 2D even if there's only one row
#     if n_rows == 1:
#         axes = axes.reshape(1, -1)
    
#     # Add column titles (emotions)
#     for j, cat in enumerate(unique_categories):
#         axes[0, j].set_title(cat, color=emotion_colors[cat], fontsize=24, fontweight='bold', pad=10)

#     for i, (n_components, avg_category) in enumerate(zip(component_values, avg_categories)):
#         for j, (cat, avg_image) in enumerate(zip(unique_categories, avg_category)):
#             print(f"Plotting row {i}, column {j}")
#             print(f"Image shape: {avg_image.shape}")
#             print(f"Image range: {avg_image.min():.4f} to {avg_image.max():.4f}")
            
#             ax = axes[i, j]
#             im = ax.imshow(avg_image.reshape(48, 48), cmap='gray')
            
#             if j == 0:  # First column
#                 ax.set_ylabel(f'{component_values[i]}', fontsize=24, rotation=0, ha='right', va='center')
#                 ax.yaxis.set_label_coords(-0.1, 0.5)  # Adjust label position
#             ax.set_xticks([])  # Remove x-axis ticks
#             ax.set_yticks([])  # Remove y-axis ticks
            
#             # Add colored border
#             for spine in ax.spines.values():
#                 spine.set_edgecolor(emotion_colors[cat])
#                 spine.set_linewidth(8)
    
#     # Add text box with details
#     n_components = results['n_components']
#     textstr = f'Analysis: {analysis_type}\nNormalization: {normalizer}\nComponents: {n_components}'
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     fig.text(0.95, 1.02, textstr, transform=fig.transFigure, fontsize=14,
#              verticalalignment='top', bbox=props)
    
#     plt.tight_layout()
#     plt.subplots_adjust(left=0.1, top=0.9, bottom=0.05, right=0.95)
#     return fig