import os
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# MLflow Experiment Tracking
mlflow.set_experiment("Image_Emotion_Analysis")

# Data Loading and Preprocessing
def load_images(df):
    """Load images from file paths and convert to 48x48 grayscale."""
    images = []
    for file_path in df['img_path']:
        img = Image.open(file_path).convert('L')  # 'L' mode ensures grayscale
        img = img.resize((48, 48))
        images.append(np.array(img).flatten())  # Flatten for PCA/NMF
    return np.array(images)

def preprocess_images(df, usage='Training'):
    """Filter and load images based on 'Training' or 'Testing' usage."""
    subset = df[df['usage'] == usage]
    X = load_images(subset)
    y = subset['emotion']
    return X, y

def reshape_to_image(arr):
    # Check if the array has 2304 elements (for 48x48 image)
    if arr.size == 2304:
        # Reshape to 48x48
        return arr.reshape(48, 48)
    else:
        raise ValueError("Array size is not 2304, cannot reshape to 48x48.")

def generate_sample_images(df, n=3, cat_col='emotion', path_col='img_path'):
    # Ensure the number of rows is within the allowed range
    if not (1 <= n <= 5):
        raise ValueError("Number rows must be between 1 and 5")
    
    categories = df[cat_col].unique()

    # Pre-sample the random images for each category and store them in a dictionary
    random_samples = {}
    for category in categories:
        category_df = df[df[cat_col] == category]
        random_samples[category] = category_df[path_col].sample(n=n).tolist()
    return random_samples

def generate_composite_faces(X, y, overall=True):
    """
    Generate composite faces (mean and median) for each emotion category and an overall category.
    
    Parameters:
        X (list of numpy arrays): Array of array images where each element is a 2D numpy array representing a face.
        y (list of str): Emotion labels corresponding to each image in X.
    
    Returns:
        tuple: A dictionary where keys are emotion categories and values are lists of composite face arrays,
               and a list of row labels ['Mean', 'Median'].
    """   
    # Initialize the dictionary to store composite faces
    composite_faces_dict = {}

    if overall:

        # Add "Overall" category that includes all images
        composite_faces_dict['Overall'] = [
            np.mean(X, axis=0).reshape((48, 48)),
            np.median(X, axis=0).reshape((48, 48))
        ]
    
    # Group images by emotion and calculate mean and median for each group
    unique_emotions = np.unique(y)
    for emotion in unique_emotions:
        # Create a boolean mask for the current emotion
        label_mask = (y == emotion)
        
        # Filter images using the mask
        emotion_images = X[label_mask]
        
        # Calculate mean and median composite faces
        mean_face = np.mean(emotion_images, axis=0).reshape((48, 48))
        median_face = np.median(emotion_images, axis=0).reshape((48, 48))

        # Store the results in the dictionary
        composite_faces_dict[emotion] = [mean_face, median_face]
    
    # Define the row labels for the matrix
    row_labels = ['Mean', 'Median']
    
    return composite_faces_dict, row_labels

def load_image(image):
    """
    Load an image from a file path or return the array if already a numpy array.
    """
    if isinstance(image, str):
        # Load the image from a file path
        return np.array(Image.open(image))
    elif isinstance(image, np.ndarray):
        # Return the array directly if it's already a numpy array
        return image
    else:
        raise ValueError("Image must be a file path or a numpy array.")

def plot_face_matrix(
        image_dict, 
        row_labels=None, 
        group_colors=None, 
        save_path=None,
        method=None,
        norm=None,
        total_components=None):
    """
    Plot a matrix of 2D faces based on the given image dictionary.
    
    Parameters:
        image_dict (dict): A dictionary with column labels as keys and a list of image file paths or numpy arrays as values.
        row_labels (list, optional): A list of labels for each row.
        group_colors (dict, optional): A dictionary mapping column labels to colors for outlining groups.
        save_path (str, optional): Path where the final plot will be saved. If set, the directories will be created if they don't exist.
    """
    # Determine the number of rows and columns
    columns = list(image_dict.keys())
    num_cols = len(columns)
    num_rows = max(len(images) for images in image_dict.values())

    # Add an extra column if row labels are provided
    extra_col = 1 if row_labels else 0

    # Create a figure and a set of subplots, adding an extra column for row labels if needed
    fig, axes = plt.subplots(num_rows, num_cols + extra_col, figsize=((num_cols + extra_col) * 3, num_rows * 3))
    axes = np.atleast_2d(axes)

    # Set the main title and subtitle
    fig.suptitle("Unsupervised Feature Reconstruction", fontsize=26, fontweight='bold')

    # Dynamically build the subtitle based on available parameters
    subtitle_parts = []
    if method:
        subtitle_parts.append(f"Method: {method.upper()}")
    if norm:
        subtitle_parts.append(f"Normalization: {norm}")
    if total_components:
        subtitle_parts.append(f"Total Components: {total_components}")
    
    if subtitle_parts:
        # Join subtitle parts into a single string
        subtitle = ", ".join(subtitle_parts)
        fig.text(0.5, 0.94, subtitle, ha='center', fontsize=16)
        plt.suptitle(subtitle, fontsize=24)
        # Set the header title above the first column
        axes[0, 0].set_title("Components Used", fontsize=24, fontweight='bold', pad=10)

    # Populate the first column with row labels if provided
    if row_labels:
        for row_idx, row_label in enumerate(row_labels):
            ax = axes[row_idx, 0]
            ax.text(0.5, 0.5, row_label, fontsize=24, fontweight='bold', ha='center', va='center')
            ax.axis('off')  # Turn off the axis for the label column

    # Plot each image in the remaining columns
    for col_idx, (col_label, images) in enumerate(image_dict.items()):
        for row_idx in range(num_rows):
            ax = axes[row_idx, col_idx + extra_col]

            if row_idx < len(images):
                image = load_image(images[row_idx])
                ax.imshow(image, cmap='gray')

                # Apply group colors based on column labels if provided
                if group_colors and col_label in group_colors:
                    color = group_colors[col_label]
                else:
                    color = 'black'
                rect = Rectangle((0, 0), image.shape[1] - 1, image.shape[0] - 1, 
                                linewidth=6, edgecolor=color, facecolor='none',
                                zorder=0)

                ax.add_patch(rect)
                # Set the title with emphasis (larger font size and bold) for the first row of each column
                if row_idx == 0:
                    ax.set_title(col_label, fontsize=24, fontweight='bold', pad=10, color=color)
            ax.axis('off')  # Hide axis ticks

    # Save the plot if a save_path is given
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.tight_layout()
    plt.show()

def run_dimensionality_reduction(X, y, max_components, components_list, method='pca', normalize='none'):
    # Step 1: Normalize data if needed
    if normalize == 'standard':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif normalize == 'minmax':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # Step 2: Initialize the results dictionary
    results = {}

    # Step 3: Define the mapping for the dimensionality reduction methods
    method_mapping = {
        'pca': PCA,
        'nmf': NMF,
        'fa': FactorAnalysis,
        'ica': FastICA
    }

    # Validate the method choice
    if method not in method_mapping:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")

    # Step 4: Apply dimensionality reduction on the overall dataset
    try:
        # Instantiate the reducer with max_components
        if method in ['fa', 'ica']:
            reducer = method_mapping[method](n_components=max_components, random_state=42, max_iter=500)
        else:
            reducer = method_mapping[method](n_components=max_components)

        # Fit the reducer to the overall data
        reducer.fit(X)

        # List to store the average reconstructed images for different components
        overall_reconstructed = []

        # Reconstruct the data using the specified components in components_list
        for n_components in components_list:
            if n_components > max_components:
                continue  # Skip if n_components exceeds max_components

            # Reconfigure the reducer for n_components and fit again
            if method in ['pca', 'fa']:
                reducer_n = method_mapping[method](n_components=n_components)
                reducer_n.fit(X)
                X_reconstructed = reducer_n.inverse_transform(reducer_n.transform(X))
            else:
                reducer_n = method_mapping[method](n_components=n_components, random_state=42, max_iter=1000)
                reducer_n.fit(X)
                X_reconstructed = np.dot(reducer_n.transform(X), reducer_n.components_)

            # Compute the average reconstructed image
            avg_image = np.mean(X_reconstructed, axis=0)

            # Optionally reshape to 48x48 if desired
            avg_image_reshaped = avg_image.reshape(48, 48)

            # Append the average image to the list
            overall_reconstructed.append(avg_image_reshaped)

        # Store the results for the overall dataset
        results['Overall'] = overall_reconstructed

    except Exception as e:
        print(f"An error occurred during the overall analysis with method '{method}': {e}")

    # Step 5: Apply dimensionality reduction for each emotion category
    unique_emotions = np.unique(y)
    for emotion in unique_emotions:
        # Filter X for the current emotion category
        X_emotion = X[y == emotion]

        # List to store the average reconstructed images for current emotion
        reconstructed_images = []

        # Apply dimensionality reduction using the specified method
        try:
            for n_components in components_list:
                if n_components > max_components:
                    continue  # Skip if n_components exceeds max_components

                # Reconfigure the reducer for n_components
                if method in ['pca', 'fa']:
                    reducer_n = method_mapping[method](n_components=n_components)
                    reducer_n.fit(X_emotion)
                    X_reconstructed = reducer_n.inverse_transform(reducer_n.transform(X_emotion))
                else:
                    reducer_n = method_mapping[method](n_components=n_components, random_state=42, max_iter=500)
                    reducer_n.fit(X_emotion)
                    X_reconstructed = np.dot(reducer_n.transform(X_emotion), reducer_n.components_)

                # Compute the average reconstructed image
                avg_image = np.mean(X_reconstructed, axis=0)

                # Optionally reshape to 48x48 if desired
                avg_image_reshaped = avg_image.reshape(48, 48)

                # Append the average image to the list
                reconstructed_images.append(avg_image_reshaped)

            # Store the average reconstructed images for the current emotion
            results[emotion] = reconstructed_images

        except Exception as e:
            print(f"An error occurred for emotion '{emotion}' with method '{method}': {e}")

    # Step 6: Return the results dictionary, components_list, and max_components
    return results, components_list, max_components

def generate_pixel_intensities(X, y, color_dict=None, save_path=None):
    pixel_dict = {}
    pixel_dict['Overall'] = np.concatenate([img.flatten() for img in X])

    unique_emotions = np.unique(y)
    for emotion in unique_emotions:
        # Filter X for the current emotion category
        X_emotion = X[y == emotion]
        emo_pixels = np.concatenate([img.flatten() for img in X_emotion])
        pixel_dict[emotion] = emo_pixels

    num_columns = len(pixel_dict)
    fig, axes = plt.subplots(1, num_columns, figsize=(num_columns * 4, 5), sharey=True)

    # Plot each column
    for idx, (key, values) in enumerate(pixel_dict.items()):
        if color_dict and key in color_dict:
            color = color_dict[key]
        else:
            color = 'black'
        # rect = Rectangle((0, 0), 1, 1, 
        #     transform=axes[idx].transAxes,  # Use axes coordinates
        #     color=color, 
        #     linewidth=6,
        #     fill=False,  # Only an outline, no fill
        #     zorder=0
        # )

        # axes[idx].add_patch(rect)

        # Plot the histogram
        axes[idx].hist(values, bins=64, range=(0, 256), color='gray', alpha=0.7, density=True)
        # ax.set_title(f"{category} - Histogram")
        axes[idx].set_xlabel(key)
        for spine in axes[idx].spines.values():
            spine.set_linewidth(5)
            spine.set_edgecolor(color)

    # Set shared y-axis label
    axes[0].set_ylabel("Pixel Density")
    fig.suptitle("Histogram of Pixel Intensity Density", fontsize=26, fontweight='bold')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")



