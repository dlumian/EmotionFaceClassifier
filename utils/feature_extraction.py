import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from datascifuncs.tidbit_tools import load_json, write_json

def load_image(file_path):
    """Load a single image file and convert it to a numpy array."""
    with Image.open(file_path) as img:
        return np.array(img)

def load_images(df, flatten=False):
    """Load images from file paths and convert to 48x48 grayscale."""
    images = []
    for file_path in df['img_path']:
        img = load_image(file_path, flatten=flatten)
        images.append(img)
    return np.array(images)

def create_X_y(df, img_path_column, label_column, flatten=True, img_size=None):
    """
    Create X and y from a DataFrame containing image paths and labels.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing image paths and labels
    img_path_column (str): Name of the column containing image file paths
    label_column (str): Name of the column containing labels
    flatten (bool): Whether to flatten the image data (default: True)
    img_size (tuple): Desired image size (height, width). If None, original size is kept (default: None)
    
    Returns:
    X (np.array): Array of images (flattened or as matrices)
    y (np.array): Array of labels
    """
    def process_image(path):
        img = load_image(path)
        if img_size:
            img = Image.fromarray(img).resize(img_size)
            img = np.array(img)
        return img

    X = np.array([process_image(path) for path in df[img_path_column]])
    
    if flatten:
        X = X.reshape(X.shape[0], -1)  # Flatten the images
    
    y = df[label_column].values
    return X, y

def normalize_data(data, normalizer):
    """Normalize the data using the specified method."""
    if normalizer == 'none':
        return data
    elif normalizer == 'minmax':
        return MinMaxScaler().fit_transform(data)
    elif normalizer == 'standard':
        return StandardScaler().fit_transform(data)
    else:
        raise ValueError(f"Unknown normalization method: {normalizer}")

def extract_features(data, analysis_type, n_components, config):
    """Perform feature extraction using the specified analysis type."""
    model_config = config[analysis_type]
    model_class = getattr(__import__(model_config['module'], fromlist=[model_config['class']]), model_config['class'])
    model = model_class(n_components=n_components, **model_config['params'])
    features = model.fit_transform(data)
    print(f"Extracted features shape: {features.shape}")
    print(f"Model n_components: {model.n_components_}")
    print(f"Model components shape: {model.components_.shape}")
    return model, features

def partial_reconstruction(model, features, component_values):
    """Perform partial reconstructions for the specified component values."""
    print(f"Input features shape: {features.shape}")
    reconstructions = []
    n_samples, n_components = features.shape
    
    for value in component_values:
        partial_features = np.zeros_like(features)
        partial_features[:, :value] = features[:, :value]
        print(f"Partial features shape for {value} components: {partial_features.shape}")
        
        reconstruction = model.inverse_transform(partial_features)
        print(f"Reconstruction shape for {value} components: {reconstruction.shape}")
        reconstructions.append(reconstruction)
    
    return reconstructions

def average_reconstructions(reconstructions, y):
    """Average reconstructions for the whole dataset and each category."""
    avg_all = [np.mean(reconstruction, axis=0) for reconstruction in reconstructions]
    unique_categories = np.unique(y)
    print(f"Number of unique categories: {len(unique_categories)}")
    print(f"Unique categories: {', '.join(unique_categories)}")
    avg_categories = [
        [np.mean(reconstruction[y == cat], axis=0) for reconstruction in reconstructions]
        for cat in unique_categories
    ]
    return avg_all, avg_categories, unique_categories

from skimage.metrics import structural_similarity as ssim

def calculate_metrics(original, reconstructed):
    """Calculate reconstruction metrics."""
    mse = mean_squared_error(original, reconstructed)
    psnr = 10 * np.log10((255**2) / mse)
    ssim_value = ssim(original, reconstructed, data_range=original.max() - original.min(), multichannel=True)
    return mse, ssim_value, psnr

def run_analysis(df, img_path_column, label_column, config_path, analysis_type, subset_condition=None, flatten=True, img_size=None):
    """
    Run the complete analysis pipeline for a given analysis type.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing image paths and labels
    img_path_column (str): Name of the column containing image file paths
    label_column (str): Name of the column containing labels
    config_path (str): Path to the configuration file
    analysis_type (str): Type of analysis to perform (e.g., 'PCA', 'NMF', 'FastICA')
    subset_condition (str, optional): A string representing a condition to subset the DataFrame
    flatten (bool): Whether to flatten the image data (default: True)
    img_size (tuple): Desired image size (height, width). If None, original size is kept (default: None)
    
    Returns:
    dict: Results of the analysis
    """
    config = load_json(config_path)  # Assuming you're using your own load_json function
    analysis_config = next(a for a in config['Analyses'] if a['type'] == analysis_type)
    
    # Apply subset condition if provided
    if subset_condition:
        df = df.query(subset_condition)
    
    X, y = create_X_y(df, img_path_column, label_column, flatten=flatten, img_size=img_size)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # If X is not flattened, flatten it for the unsupervised analysis
    if not flatten:
        X_flattened = X.reshape(X.shape[0], -1)
    else:
        X_flattened = X
    print(f"X_flattened shape: {X_flattened.shape}")
    
    results = {}
    for normalizer in analysis_config['normalization']:
        normalized_X = normalize_data(X_flattened, normalizer)
        print(f"Normalized X shape: {normalized_X.shape}")
        model, features = extract_features(normalized_X, analysis_type, analysis_config['total_components'], config)
        
        reconstructions = partial_reconstruction(model, features, analysis_config['components_for_reconstruction'])
        
        avg_all, avg_categories, unique_categories = average_reconstructions(reconstructions, y)
        
        # Calculate metrics for each partial reconstruction
        metrics = {
            'mse': [],
            'ssim': [],
            'psnr': []
        }
        for reconstruction in reconstructions:
            mse, ssim, psnr = calculate_metrics(normalized_X, reconstruction)
            metrics['mse'].append(mse)
            metrics['ssim'].append(ssim)
            metrics['psnr'].append(psnr)
        
        results[normalizer] = {
            'model': model,
            'features': features,
            'reconstructions': reconstructions,
            'avg_all': avg_all,
            'avg_categories': avg_categories,
            'unique_categories': unique_categories,
            'metrics': metrics,
            'y': y,
            'component_values': analysis_config['components_for_reconstruction']
        }
    
    return results, X, y

def create_enhanced_matrix_plot(results, analysis_type, emotion_colors, normalization):
    reconstructions = results['reconstructions']
    y = results['y']
    component_values = results['component_values']
    n_components = len(component_values)
    unique_categories = results['unique_categories']
    n_categories = len(unique_categories)
    
    fig, axes = plt.subplots(n_components, n_categories, figsize=(n_categories*3, n_components*3))
    fig.suptitle(f'{analysis_type} Reconstructions', fontsize=30, fontweight='bold', y=1.05)
    
    # Ensure axes is 2D even if there's only one row
    if n_components == 1:
        axes = axes.reshape(1, -1)
    
    # Add column titles (emotions)
    for j, cat in enumerate(unique_categories):
        axes[0, j].set_title(cat, color=emotion_colors[cat], fontsize=24, fontweight='bold', pad=10)

    for i in range(n_components):
        for j, cat in enumerate(unique_categories):
            avg_reconstruction = np.mean(reconstructions[i][y == cat], axis=0)
            im = axes[i, j].imshow(avg_reconstruction.reshape(48, 48), cmap='gray')
            
            if j == 0:  # First column
                axes[i, j].set_ylabel(f'{component_values[i]}', fontsize=24, rotation=0, ha='right', va='center')
                axes[i, j].yaxis.set_label_coords(-0.1, 0.5)  # Adjust label position
            axes[i, j].set_xticks([])  # Remove x-axis ticks
            axes[i, j].set_yticks([])  # Remove y-axis ticks
            
            # Add colored border
            for spine in axes[i, j].spines.values():
                spine.set_edgecolor(emotion_colors[cat])
                spine.set_linewidth(8)
    
    # Add text box with details
    textstr = f'Total components: {results["model"].n_components_}\nNormalization: {normalization}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.95, 1.02, textstr, transform=fig.transFigure, fontsize=20,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Add 'Components' label to the left of the plot
    fig.text(0.02, 0.5, 'Components', va='center', rotation='vertical', fontsize=24)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, top=0.9, bottom=0.05, right=0.95)
    return fig

def perform_unsupervised_analysis(df, img_path_column, label_column, config_path, analysis_types, emotion_colors):
    base_dir = 'models/unsupervised'
    os.makedirs(base_dir, exist_ok=True)

    for analysis_type in analysis_types:
        results, X, y = run_analysis(df, img_path_column, label_column, config_path, analysis_type)
        
        for normalizer, result in results.items():
            total_components = result['model'].n_components_
            dir_name = f"{analysis_type.lower()}_{normalizer}_{total_components}"
            result_dir = os.path.join(base_dir, dir_name)
            
            # Skip if results directory already exists
            if os.path.exists(result_dir):
                print(f"Results for {dir_name} already exist. Skipping...")
                continue
            
            os.makedirs(result_dir, exist_ok=True)
            
            # Save plot
            fig = create_enhanced_matrix_plot(result, analysis_type, emotion_colors, normalizer)
            fig.savefig(os.path.join(result_dir, f"{dir_name}_plot.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Save analysis settings and results
            analysis_info = {
                'analysis_type': analysis_type,
                'normalizer': normalizer,
                'total_components': total_components,
                'component_values': result['component_values']
            }
            write_json(analysis_info, os.path.join(result_dir, f"{dir_name}_info.json"))
            
            # Create and save metrics CSV
            metrics_df = pd.DataFrame({
                'components': result['component_values'],
                'mse': result['metrics']['mse'],
                'ssim': result['metrics']['ssim']
            })
            metrics_df.to_csv(os.path.join(result_dir, f"{dir_name}_metrics.csv"), index=False)
            
            print(f"Analysis results for {dir_name} saved successfully.")
