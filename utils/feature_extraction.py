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

def get_n_components(model):
    """
    Get the number of components from a model, handling different attribute names.
    """
    if hasattr(model, 'n_components_'):
        return model.n_components_
    elif hasattr(model, 'n_components'):
        return model.n_components
    else:
        raise AttributeError(f"Model of type {type(model)} doesn't have n_components or n_components_ attribute")

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
    if analysis_type.lower() == 'nmf':
        print(f"NMF reconstruction error: {model.reconstruction_err_}")
        print(f"NMF n_iter_: {model.n_iter_}")
        print(f"NMF components shape: {model.components_.shape}")
        print(f"NMF components range: {model.components_.min()} to {model.components_.max()}")
        print(f"NMF features shape: {features.shape}")
        print(f"NMF features range: {features.min()} to {features.max()}")
                # Additional checks
        if model.n_iter_ >= model.max_iter:
            print("Warning: NMF reached max_iter before converging. Consider increasing max_iter.")
        
        # Check for any zero or very small components
        small_components = (model.components_ < 1e-10).sum()
        if small_components > 0:
            print(f"Warning: {small_components} components have very small values (< 1e-10).")
        
        # Check for any zero or very small feature values
        small_features = (features < 1e-10).sum()
        if small_features > 0:
            print(f"Warning: {small_features} feature values are very small (< 1e-10).")

        # Check for component similarity
        from sklearn.metrics.pairwise import cosine_similarity
        component_similarity = cosine_similarity(model.components_)
        np.fill_diagonal(component_similarity, 0)  # Zero out self-similarity
        print(f"Max component similarity: {component_similarity.max():.4f}")
        print(f"Mean component similarity: {component_similarity.mean():.4f}")
        
        # Check for feature similarity
        feature_similarity = cosine_similarity(features.T)
        np.fill_diagonal(feature_similarity, 0)
        print(f"Max feature similarity: {feature_similarity.max():.4f}")
        print(f"Mean feature similarity: {feature_similarity.mean():.4f}")
        check_component_uniqueness(model)

    return model, features

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

def average_reconstructions(reconstructions, y):
    unique_categories = np.unique(y)
    avg_categories = []
    
    for reconstruction in reconstructions:
        # Calculate overall average (using all of X)
        overall_avg = np.mean(reconstruction, axis=0)
        
        # Calculate average for each category
        category_avgs = [overall_avg]  # Start with the overall average
        for category in unique_categories:
            category_avg = np.mean(reconstruction[y == category], axis=0)
            category_avgs.append(category_avg)
        
        avg_categories.append(category_avgs)

    # Add 'Overall' to the beginning of unique_categories
    unique_categories = np.insert(unique_categories, 0, 'Overall')
    print("Shape of avg_categories:", np.array(avg_categories).shape)
    print("Unique categories:", unique_categories)  
    
    return avg_categories, unique_categories

def compare_reconstructions(reconstructions):
    n_reconstructions = len(reconstructions)
    for i in range(n_reconstructions):
        for j in range(i+1, n_reconstructions):
            if np.allclose(reconstructions[i], reconstructions[j], rtol=1e-5, atol=1e-8):
                print(f"Warning: Reconstructions {i} and {j} are numerically very close")

def check_component_uniqueness(model):
    from sklearn.metrics.pairwise import cosine_similarity
    
    components = model.components_
    similarity_matrix = cosine_similarity(components)
    np.fill_diagonal(similarity_matrix, 0)  # Exclude self-similarity
    
    high_similarity = (similarity_matrix > 0.95).sum() / 2  # Divide by 2 to avoid double counting
    
    print(f"Number of highly similar component pairs (>0.95 cosine similarity): {high_similarity}")
    print(f"Max non-self similarity between components: {similarity_matrix.max():.4f}")

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

# def run_analysis(X, y, analysis_type, normalizer, analysis_config, config):
#     # If X is not flattened, flatten it for the unsupervised analysis
#     if len(X.shape) > 2:
#         X_flattened = X.reshape(X.shape[0], -1)
#     else:
#         X_flattened = X  

#     normalized_X = normalize_data(X_flattened, normalizer)
#     model, features = extract_features(normalized_X, analysis_type, analysis_config['total_components'], config)
#     reconstructions = partial_reconstruction(model, features, analysis_config['components_for_reconstruction'])
#     avg_categories, unique_categories = average_reconstructions(reconstructions, y)
    
#     # Calculate metrics for each partial reconstruction
#     metrics = {
#         'mse': [],
#         'ssim': [],
#         'psnr': []
#     }
#     for reconstruction in reconstructions:
#         mse, ssim_value, psnr = calculate_metrics(normalized_X, reconstruction)
#         metrics['mse'].append(mse)
#         metrics['ssim'].append(ssim_value)
#         metrics['psnr'].append(psnr)
    
#     n_components = get_n_components(model)

#     results = {
#         'model': model,
#         'features': features,
#         'reconstructions': reconstructions,
#         # 'avg_all': avg_all,
#         'avg_categories': avg_categories,
#         'unique_categories': unique_categories,
#         'metrics': metrics,
#         'y': y,
#         'component_values': analysis_config['components_for_reconstruction'],
#         'n_components': n_components
#     }
    
#     print("Reconstructions shape:", [r.shape for r in reconstructions])
#     # print("Avg all shape:", avg_all.shape)
#     print("Avg categories shape:", np.array(avg_categories).shape)

#     return results

def run_analysis(X, y, analysis_type, normalizer, analyses_config, config):
    n_components = analyses_config['total_components']
    # params = analyses_config['params']
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

    # if analysis_type == 'NMF':
    #     model = NMF(n_components=n_components, **params)
    # elif analysis_type == 'PCA':
    #     model = PCA(n_components=n_components, **params)
    # elif analysis_type == 'FastICA':
    #     model = FastICA(n_components=n_components, **params)
    # else:
    #     raise ValueError(f"Unsupported analysis type: {analysis_type}")

    # Perform fit_transform for all X
    features_all = model.fit_transform(X_normalized)
    reconstructions_all, metrics_all = partial_reconstruction(model, features_all, component_values, X_normalized)

    # Initialize results
    avg_categories = [np.mean(reconstruction, axis=0) for reconstruction in reconstructions_all]
    metrics = [dict(m, Category='Overall') for m in metrics_all]

    # Perform fit_transform for each category
    unique_categories = np.unique(y)
    # features_categories = []
    # reconstructions_categories = []

    for category in unique_categories:
        X_category = X_normalized[y == category]
        model_category = clone(model)  # Create a new instance of the model for each category
        features_category = model_category.fit_transform(X_category)
        # features_categories.append(features_category)
        # reconstructions_category = partial_reconstruction(model_category, features_category, component_values)
        reconstructions_category, metrics_category = partial_reconstruction(model_category, features_category, component_values, X_category)
        # reconstructions_categories.append(reconstructions_category)
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
    
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

    print(f"Plotting for {analysis_type} with {normalizer} normalization")
    print(f"Number of rows (component values): {len(component_values)}")
    print(f"Number of columns (categories): {len(unique_categories)}")
    print(f"Shape of avg_categories: {np.array(avg_categories).shape}")

    # Add a color for the 'Overall' category
    overall_color = 'black'  # You can change this to any color you prefer
    emotion_colors = {**{'Overall': overall_color}, **emotion_colors}

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
        cmap = colormaps[i % len(colormaps)]
        for j, (cat, avg_image) in enumerate(zip(unique_categories, avg_category)):
            print(f"Plotting row {i}, column {j}")
            print(f"Image shape: {avg_image.shape}")
            print(f"Image range: {avg_image.min():.4f} to {avg_image.max():.4f}")
            
            ax = axes[i, j]
            im = ax.imshow(avg_image.reshape(48, 48), cmap=cmap)
            # ax.axis('off')
            
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
    fig.text(0.02, 0.98, textstr, transform=fig.transFigure, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # Add 'Components' label to the left of the plot
    fig.text(0.02, 0.5, 'Components', va='center', rotation='vertical', fontsize=24)

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
            
                        # Check component uniqueness
            check_component_uniqueness(results['model'])
            
            # Compare reconstructions
            compare_reconstructions(results['reconstructions'])

            print("Results keys:", results.keys())
            print("Avg categories shape:", np.array(results['avg_categories']).shape)
            print("Unique categories:", results['unique_categories'])
            print("Component values:", results['component_values'])

            # Save plot
            fig = create_enhanced_matrix_plot(results, analysis_type, emotion_colors, normalizer)
            fig.savefig(os.path.join(result_dir, f"{dir_name}_plot.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
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

            print(f"Analysis results for {dir_name} saved successfully.")

    print("\nAll analyses completed.")


def check_component_uniqueness(model):
    from sklearn.metrics.pairwise import cosine_similarity
    
    components = model.components_
    similarity_matrix = cosine_similarity(components)
    np.fill_diagonal(similarity_matrix, 0)  # Exclude self-similarity
    
    high_similarity = (similarity_matrix > 0.95).sum() / 2  # Divide by 2 to avoid double counting
    
    print(f"Number of highly similar component pairs (>0.95 cosine similarity): {high_similarity}")
    print(f"Max non-self similarity between components: {similarity_matrix.max():.4f}")

def compare_reconstructions(reconstructions):
    n_reconstructions = len(reconstructions)
    for i in range(n_reconstructions):
        for j in range(i+1, n_reconstructions):
            if np.allclose(reconstructions[i], reconstructions[j], rtol=1e-5, atol=1e-8):
                print(f"Warning: Reconstructions {i} and {j} are numerically very close")