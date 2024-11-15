import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datascifuncs.tidbit_tools import load_json, write_json
from .analysis_tools import instantiate_model

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
    partial_features[:,recon_components:]=0
    return model.inverse_transform(partial_features)

def compile_metrics(category_results, file_path):
    metric_dict = {}
    for cr in category_results:
        cat = cr.get('category')
        mets = cr.get('metrics_recon_dicts')
    if cat is not None and mets is not None:    
        metric_dict[cat] = mets
        metrics_df = pd.DataFrame(metric_dict)
        metrics_df.to_csv(file_path, index=False)
        print(f"Metrics saved to {file_path}")
    else:
        print("No valid category or metrics found in category_results.")

def run_single_analysis(X, y, analysis_config):
    base_dir = 'models/unsupervised'
    os.makedirs(base_dir, exist_ok=True)
    model_type = analysis_config['class']
    total_components = analysis_config['total_components']
    component_values = analysis_config['components_for_reconstruction']
    normalizer = analysis_config['normalization']
    unique_categories = np.unique(y)
    unique_categories = np.insert(unique_categories, 0, 'Overall')

    dir_name = f"{model_type.lower()}_{normalizer}_{total_components}"
    result_dir = os.path.join(base_dir, dir_name)
    analysis_json = os.path.join(result_dir, f"{dir_name}_info.json")
    metrics_file = os.path.join(result_dir, f"{dir_name}_metrics.csv")
    avg_reconstructions_file = os.path.join(result_dir, f"{model_type.lower()}_avg_reconstructions.npz")
    
    results = {
        'model': model_type,
        'results_dir': result_dir,
        'n_components': total_components,
        'component_values': component_values,
        'unique_categories': unique_categories,
        'model_config': analysis_config
    }

    if os.path.exists(metrics_file):
        print(f"Metrics file for {dir_name} already exists. Skipping...")
        return results

    print(f"Analysis type: {model_type}")
    print(f"Normalization: {normalizer}")
    print(f"Total components: {total_components}")
    print(f"Parameters: {analysis_config['params']}")
    os.makedirs(result_dir, exist_ok=True)

    analysis_config['params']['n_components'] = total_components
    model = instantiate_model(analysis_config)
    X_normalized = normalize_data(X, normalizer)

    category_results = []
    category_avg_images = {}

    for category in unique_categories:
        model_category = clone(model)
        valid_components_values = []
        category_recon_avg_images = []
        metrics_recon_dicts = []

        if category == 'Overall':
            X_category = X_normalized
        else:
            X_category = X_normalized[y == category]

        features_category = model_category.fit_transform(X_category)

        for recon_components in component_values:
            if recon_components > total_components:
                raise ValueError(f"Requested components ({recon_components}) exceed total_components ({total_components}).")
            else:
                valid_components_values.append(recon_components)
                recon_images = feature_reconstruction(model_category, features_category, recon_components)
                category_recon_avg_images.append(np.mean(recon_images, axis=0))
                metrics_recon_dicts.append(calculate_metrics(X_category, recon_images))

        iteration_results={
            'category': category,
            'valid_component_values': valid_components_values,
            'metrics_recon_dicts' : metrics_recon_dicts
        }
        category_results.append(iteration_results)
        category_avg_images[category]=category_recon_avg_images

    write_json(category_results, analysis_json)
    print(f"Analysis settings saved to {analysis_json}")
    np.savez_compressed(avg_reconstructions_file, **category_avg_images)
    print(f"Averaged reconstructions saved to {avg_reconstructions_file}")
    compile_metrics(category_results, metrics_file)

    return category_avg_images


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

