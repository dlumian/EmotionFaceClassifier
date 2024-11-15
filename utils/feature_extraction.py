import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from datascifuncs.tidbit_tools import load_json, write_json
from .analysis_tools import instantiate_model

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

def partial_reconstruction(model, features, component_values, original_data):
    reconstructions = []
    metrics = []
    n_samples, n_features = original_data.shape
    for n_components in component_values:
        # Create a zero-filled array with the same shape as the original features
        partial_features = np.zeros((n_samples, model.n_components_))
        # Fill in the used components
        partial_features[:, :n_components] = features[:, :n_components]
        
        reconstruction = model.inverse_transform(partial_features)
        reconstructions.append(reconstruction)
        metrics.append(calculate_metrics(original_data, reconstruction, n_components))
    return reconstructions, metrics

def calculate_metrics(original, reconstruction, n_components):
    mse = mean_squared_error(original, reconstruction)
    psnr = 10 * np.log10((255**2) / mse)
    ssim_value = ssim(original, reconstruction, data_range=original.max() - original.min(), multichannel=True)
    return {
        'Components': n_components,
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_value
    }

def run_analysis(X, y, analysis_type, normalizer, analyses_config, config):
    n_components = analyses_config['total_components']
    component_values = analyses_config['components_for_reconstruction']

    # If X is not flattened, flatten it for the unsupervised analysis
    if len(X.shape) > 2:
        X_flattened = X.reshape(X.shape[0], -1)
    else:
        X_flattened = X  

    X_normalized = normalize_data(X_flattened, normalizer)

    # Create the model
    model_config = config[analysis_type]
    model_config['n_componenets'] = n_components
    model = instantiate_model(model_config)

    # Perform fit_transform for all X
    features_all = model.fit_transform(X_normalized)
    reconstructions_all, metrics_all = partial_reconstruction(model, features_all, component_values, X_normalized)

    # Initialize results
    avg_categories = [np.mean(reconstruction, axis=0) for reconstruction in reconstructions_all]
    metrics = [dict(m, Category='Overall') for m in metrics_all]

    # Perform fit_transform for each category
    # unique_categories = np.unique(y)
    unique_categories = np.array(["Angry", "Happy", "Surprise"])

    for category in unique_categories:
        X_category = X_normalized[y == category]
        model_category = clone(model)  # Create a new instance of the model for each category
        features_category = model_category.fit_transform(X_category)
        reconstructions_category, metrics_category = partial_reconstruction(model_category, features_category, component_values, X_category)
        avg_categories.append([np.mean(reconstruction, axis=0) for reconstruction in reconstructions_category])
        metrics.extend([dict(m, Category=category) for m in metrics_category])

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics)
   
    # Prepare results
    results = {
        'model': model,
        'features': features_all,
        'reconstructions': reconstructions_all,
        'avg_categories': avg_categories,
        'unique_categories': np.insert(unique_categories, 0, 'Overall'),
        'metrics': metrics_df,
        'y': y,
        'component_values': component_values,
        'n_components': n_components,
        'model_config': model_config
    }

    return results, metrics_df

def create_enhanced_matrix_plot(results, analysis_type, emotion_colors, normalizer):
    avg_categories = results['avg_categories']
    unique_categories = results['unique_categories']
    component_values = results['component_values']
    
    # colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

    print(f"Plotting for {analysis_type} with {normalizer} normalization")
    print(f"Number of rows (component values): {len(component_values)}")
    print(f"Number of columns (categories): {len(unique_categories)}")

    n_rows = len(component_values)
    n_cols = len(unique_categories)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    fig.suptitle(f'{analysis_type} Reconstructions', fontsize=30, fontweight='bold', y=1.05)
    
    # Ensure axes is 2D even if there's only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Add column titles (emotions)
    for j, cat in enumerate(unique_categories):
        axes[0, j].set_title(cat, color=emotion_colors[cat], fontsize=24, fontweight='bold', pad=10)

    for i, (n_components, avg_category) in enumerate(zip(component_values, avg_categories)):
        for j, (cat, avg_image) in enumerate(zip(unique_categories, avg_category)):
            print(f"Plotting row {i}, column {j}")
            print(f"Image shape: {avg_image.shape}")
            print(f"Image range: {avg_image.min():.4f} to {avg_image.max():.4f}")
            
            ax = axes[i, j]
            im = ax.imshow(avg_image.reshape(48, 48), cmap='gray')
            
            if j == 0:  # First column
                ax.set_ylabel(f'{component_values[i]}', fontsize=24, rotation=0, ha='right', va='center')
                ax.yaxis.set_label_coords(-0.1, 0.5)  # Adjust label position
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks
            
            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(emotion_colors[cat])
                spine.set_linewidth(8)
    
    # Add text box with details
    n_components = results['n_components']
    textstr = f'Analysis: {analysis_type}\nNormalization: {normalizer}\nComponents: {n_components}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.95, 1.02, textstr, transform=fig.transFigure, fontsize=14,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, top=0.9, bottom=0.05, right=0.95)
    return fig

def perform_unsupervised_analysis(df, img_path_column, label_column, config_path, analysis_types, emotion_colors, flatten=True, img_size=None):
    
    base_dir = 'models/unsupervised'
    os.makedirs(base_dir, exist_ok=True)

    # Load X and y only once
    X, y = create_X_y(df, img_path_column, label_column, flatten=flatten, img_size=img_size)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Load the configuration file using your custom function
    config = load_json(config_path)

    # Calculate total number of analyses
    total_analyses = sum(len(next(item for item in config['Analyses'] if item['type'] == analysis_type)['normalization']) 
                         for analysis_type in analysis_types)
    print(f"Total number of analyses to be run: {total_analyses}")

    current_analysis = 0

    for analysis_type in analysis_types:
        # Get the specific analysis configuration
        analyses_config = next(item for item in config['Analyses'] if item['type'] == analysis_type)

        for normalizer in analyses_config['normalization']:
            current_analysis += 1
            total_components = analyses_config['total_components']
            dir_name = f"{analysis_type.lower()}_{normalizer}_{total_components}"
            result_dir = os.path.join(base_dir, dir_name)
            metrics_file = os.path.join(result_dir, f"{dir_name}_metrics.csv")
            avg_reconstructions_file = os.path.join(result_dir, f"{analysis_type}_avg_reconstructions.npz")

            print(f"\nStarting analysis {current_analysis}/{total_analyses}")
            print(f"Analysis type: {analysis_type}")
            print(f"Normalization: {normalizer}")
            print(f"Total components: {total_components}")
            print(f"Parameters: {config[analysis_type]['params']}")

            # Skip if metrics file already exists
            if os.path.exists(metrics_file):
                print(f"Metrics file for {dir_name} already exists. Skipping...")
                continue
            
            os.makedirs(result_dir, exist_ok=True)

            # Run the analysis
            results, metrics_df = run_analysis(X, y, analysis_type, normalizer, analyses_config, config)

            print("Results keys:", results.keys())
            print("Unique categories:", results['unique_categories'])
            print("Component values:", results['component_values'])
            
            # Save analysis settings and results using your custom function
            analysis_info = {
                'analysis_type': analysis_type,
                'normalizer': normalizer,
                'total_components': results['n_components'],
                'component_values': results['component_values']
            }
            write_json(analysis_info, os.path.join(result_dir, f"{dir_name}_info.json"))
            
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Metrics saved to {metrics_file}")

            np.savez_compressed(avg_reconstructions_file, 
                                avg_categories=results['avg_categories'], 
                                unique_categories=results['unique_categories'],
                                component_values=results['component_values']) 
            
            print(f"Averaged reconstructions saved to {avg_reconstructions_file}")

            # Save plot
            fig = create_enhanced_matrix_plot(results, analysis_type, emotion_colors, normalizer)
            fig.savefig(os.path.join(result_dir, f"{dir_name}_plot.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Analysis results for {dir_name} saved successfully.")

    print("\nAll analyses completed.")

def load_avg_reconstructions(file_path):
    with np.load(file_path) as data:
        avg_categories = data['avg_categories']
        unique_categories = data['unique_categories']
        component_values = data['component_values']
    return avg_categories, unique_categories, component_values