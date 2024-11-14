import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .analysis_tools import instantiate_model

def load_image(image, flatten=False):
    """
    Load an image from a file path or return the array if already a numpy array.
    """
    if isinstance(image, str):
        # Load the image from a file path
        img = np.array(Image.open(image).convert('L'))
    elif isinstance(image, np.ndarray):
        # Return the array directly if it's already a numpy array
        img = image
    else:
        raise ValueError("Image must be a file path or a numpy array.")
    if flatten:
        img = img.flatten()
    return img

def load_images(df, flatten=False):
    """Load images from file paths and convert to 48x48 grayscale."""
    images = []
    for file_path in df['img_path']:
        img = load_image(file_path, flatten=flatten)
        images.append(img)
    return np.array(images)

def preprocess_images(df, usage='Training', flatten=False):
    """Filter and load images based on 'Training' or 'Testing' usage."""
    subset = df[df['usage'] == usage]
    X = load_images(subset, flatten=flatten)
    y = subset['emotion']
    return X, y

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
    row_labels = {
        "Method" : ['Mean', 'Median']
    }
    
    return composite_faces_dict, row_labels

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
    fig.suptitle("Pixel Intensity Density Plot", fontsize=26, fontweight='bold', y=1.05)

    # Custom x-ticks and font size for readability
    x_ticks = [0, 128, 255]
    tick_fontsize = 16 

    # Plot each column
    for idx, (key, values) in enumerate(pixel_dict.items()):
        if color_dict and key in color_dict:
            color = color_dict[key]
        else:
            color = 'black'

        # Plot the histogram
        axes[idx].hist(values, bins=64, range=(0, 256), color=color, alpha=0.7, density=True)
        axes[idx].set_title(key, fontsize=24, fontweight='bold', pad=10, color=color)

        # Customize x-axis ticks to only have 0, 128, and 255
        axes[idx].set_xticks(x_ticks)
        # Set the font size for x and y ticks
        axes[idx].tick_params(axis='x', labelsize=tick_fontsize)

        axes[idx].spines['left'].set_linewidth(8)
        axes[idx].spines['left'].set_edgecolor(color)

        axes[idx].spines['bottom'].set_linewidth(8)
        axes[idx].spines['bottom'].set_edgecolor(color)

        # Remove the top and right spines
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)

    # Set shared y-axis label
    axes[0].set_ylim(0, 0.007)
    axes[0].set_yticks([0.001, 0.003, 0.005, 0.007])
    axes[0].set_yticklabels([f"{y * 100:.1f}%" for y in axes[0].get_yticks()])
    axes[0].set_ylabel("Pixel Density (%)", fontsize=24)
    axes[0].tick_params(axis='y', labelsize=tick_fontsize)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")


def plot_matrix(image_dict, row_labels=None):
    columns = list(image_dict.keys())
    num_cols = len(columns)
    num_rows = max(len(images) for images in image_dict.values())

    # Add an extra column if row labels are provided
    extra_col = 1 if row_labels else 0

    # Create a figure and a set of subplots, adding an extra column for row labels if needed
    fig, axes = plt.subplots(num_rows, num_cols + extra_col, figsize=((num_cols + extra_col) * 3, num_rows * 3))
    axes = np.atleast_2d(axes)

    # Populate the first column with row labels if provided
    if row_labels:
        for title, lables in row_labels.items():
            for row_idx, row_label in enumerate(lables):
                ax = axes[row_idx, 0]
                ax.text(0.5, 0.5, row_label, fontsize=24, fontweight='bold', ha='center', va='center')
                ax.axis('off')  # Turn off the axis for the label column
                if row_idx == 0:
                    ax.set_title(title, fontsize=24, fontweight='bold', pad=10)

    # Plot each image in the remaining columns
    for col_idx, (col_label, images) in enumerate(image_dict.items()):
        for row_idx in range(num_rows):
            print(f'col_idx: {col_idx}')
            print(f'col_label: {col_label}')
            print(f'row_idx: {row_idx}')
            print(f'extra_col: {extra_col}')


            ax = axes[row_idx, col_idx + extra_col]

            if row_idx < len(images):
                image = load_image(images[row_idx])
                ax.imshow(image, cmap='gray')

                # Set the title with emphasis (larger font size and bold) for the first row of each column
                if row_idx == 0:
                    ax.set_title(col_label, fontsize=24, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig, axes.flatten()


# def plot_face_matrix(
#         image_dict, 
#         row_labels=None, 
#         group_colors=None, 
#         save_path=None,
#         method=None,
#         norm=None,
#         total_components=None):
#     """
#     Plot a matrix of 2D faces based on the given image dictionary.
    
#     Parameters:
#         image_dict (dict): A dictionary with column labels as keys and a list of image file paths or numpy arrays as values.
#         row_labels (list, optional): A list of labels for each row.
#         group_colors (dict, optional): A dictionary mapping column labels to colors for outlining groups.
#         save_path (str, optional): Path where the final plot will be saved. If set, the directories will be created if they don't exist.
#     """
#     # Determine the number of rows and columns
#     columns = list(image_dict.keys())
#     num_cols = len(columns)
#     num_rows = max(len(images) for images in image_dict.values())

#     # Add an extra column if row labels are provided
#     extra_col = 1 if row_labels else 0

#     # Create a figure and a set of subplots, adding an extra column for row labels if needed
#     fig, axes = plt.subplots(num_rows, num_cols + extra_col, figsize=((num_cols + extra_col) * 3, num_rows * 3))
#     axes = np.atleast_2d(axes)

#     # Set the main title and subtitle
#     fig.suptitle("Unsupervised Feature Reconstruction", fontsize=26, fontweight='bold')

#     # Dynamically build the subtitle based on available parameters
#     subtitle_parts = []
#     if method:
#         subtitle_parts.append(f"Method: {method.upper()}")
#     if norm:
#         subtitle_parts.append(f"Normalization: {norm}")
#     if total_components:
#         subtitle_parts.append(f"Total Components: {total_components}")
    
#     if subtitle_parts:
#         # Join subtitle parts into a single string
#         subtitle = ", ".join(subtitle_parts)
#         # fig.text(0.5, 0.94, subtitle, ha='center', fontsize=16)
#         plt.suptitle(subtitle, fontsize=24)
#         # Set the header title above the first column
#         axes[0, 0].set_title("Components Used", fontsize=24, fontweight='bold', pad=10)

#     # Populate the first column with row labels if provided
#     if row_labels:
#         for row_idx, row_label in enumerate(row_labels):
#             ax = axes[row_idx, 0]
#             ax.text(0.5, 0.5, row_label, fontsize=24, fontweight='bold', ha='center', va='center')
#             ax.axis('off')  # Turn off the axis for the label column

#     # Plot each image in the remaining columns
#     for col_idx, (col_label, images) in enumerate(image_dict.items()):
#         for row_idx in range(num_rows):
#             ax = axes[row_idx, col_idx + extra_col]

#             if row_idx < len(images):
#                 image = load_image(images[row_idx])
#                 ax.imshow(image, cmap='gray')

#                 # Apply group colors based on column labels if provided
#                 if group_colors and col_label in group_colors:
#                     color = group_colors[col_label]
#                 else:
#                     color = 'black'

#                 for spine in ax.spines.values():
#                     spine.set_linewidth(7)
#                     spine.set_edgecolor(color)

#                 # Set the title with emphasis (larger font size and bold) for the first row of each column
#                 if row_idx == 0:
#                     ax.set_title(col_label, fontsize=24, fontweight='bold', pad=10, color=color)
#             # ax.axis('off')  # Hide axis ticks
#             ax.set_xticks([])
#             ax.set_yticks([])

#     # Save the plot if a save_path is given
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         fig.savefig(save_path, bbox_inches='tight')
#         print(f"Plot saved to {save_path}")

#     plt.tight_layout()
#     plt.show()


def apply_ticks(axes, plotting_params):
    """
    Applies x-axis and y-axis tick settings from `plotting_params` to each axis in `axes`.
    
    Parameters:
    - axes: list or array of matplotlib Axes objects to modify
    - plotting_params: dict containing tick settings for x and y axes (e.g., `xticks`, `yticks`, `xlim`, `ylim`)
    
    Returns:
    - axes: The modified axes after applying tick settings
    """
    
    # Extract x_params and y_params from plotting_params
    x_params = plotting_params.get("x_params", {})
    y_params = plotting_params.get("y_params", {})

    # Loop through each axis to apply x and y tick settings
    for ax in axes:
        # Apply x-axis settings
        if "xticks" in x_params and x_params["xticks"] is not None:
            ax.set_xticks(x_params["xticks"])
        if "xlim" in x_params and x_params["xlim"] is not None:
            ax.set_xlim(x_params["xlim"])
        if "labelsize" in x_params and x_params["labelsize"] is not None:
            ax.tick_params(axis='x', labelsize=x_params["labelsize"])
        if "labelrotation" in x_params and x_params["labelrotation"] is not None:
            ax.tick_params(axis='x', rotation=x_params["labelrotation"])

        # Apply y-axis settings
        if "yticks" in y_params and y_params["yticks"] is not None:
            ax.set_yticks(y_params["yticks"])
        if "ylim" in y_params and y_params["ylim"] is not None:
            ax.set_ylim(y_params["ylim"])
        if "labelsize" in y_params and y_params["labelsize"] is not None:
            ax.tick_params(axis='y', labelsize=y_params["labelsize"])
        if "labelcolor" in y_params and y_params["labelcolor"] is not None:
            ax.tick_params(axis='y', colors=y_params["labelcolor"])

def set_spines_and_titles_by_column(axes, title_colors):
    """
    Sets spine and title colors for each column based on the first row's titles,
    while retaining existing title properties.
    
    Parameters:
    - axes: Flattened list or array of matplotlib Axes objects.
    - title_colors: Dict mapping each column title to a color (e.g., {"Column A": "blue"}).
    """
    # Determine the number of columns by counting titles in the first row
    ncols = sum(1 for ax in axes if ax.get_title())
    nrows = len(axes) // ncols
    
    for col in range(ncols):
        title = axes[col].get_title()
        color = title_colors.get(title, "black")
        
        for row in range(nrows):
            ax = axes[row * ncols + col]
            
            # Set title color directly for the first row only
            if row == 0:
                ax.title.set_color(color)
            
            # Apply color to all spines in this subplot
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(6)  # Adjust line width as needed

def add_figure_title(fig, plotting_params, title_text):
    """
    Adds an overall title to the figure based on JSON configuration.
    
    Parameters:
    - fig: Matplotlib Figure object.
    - title_params: Dictionary containing title settings (e.g., title, fontsize, color, fontweight, y).
    """
    title_params = plotting_params.get('figure_title', {})
    # Extract parameters from the dictionary
    fontsize = title_params.get("fontsize", 24)
    color = title_params.get("color", "black")
    fontweight = title_params.get("fontweight", "bold")
    y = title_params.get("y", 1.05)  # Default y position to avoid overlap
    
    # Set the figure title with all specified properties
    fig.suptitle(title_text, fontsize=fontsize, color=color, fontweight=fontweight, y=y)

def add_text_box(fig, plotting_params, text):
    """
    Adds a text box to the figure based on JSON configuration.
    
    Parameters:
    - fig: Matplotlib Figure object.
    - text_params: Dictionary containing text box settings (e.g., text, position, fontsize, color, fontweight, ha).
    """
    text_params = plotting_params.get('text_box', {})

    # Extract parameters from the dictionary
    position = text_params.get("position", [0.95, 1.02])
    fontsize = text_params.get("fontsize", 14)
    color = text_params.get("color", "black")
    fontweight = text_params.get("fontweight", "normal")
    ha = text_params.get("ha", "center")  # Horizontal alignment
    
    # Add the text box with specified properties
    fig.text(position[0], position[1], text, fontsize=fontsize, color=color, fontweight=fontweight, ha=ha)

def save_figure(fig, save_path, dpi=150):
    """
    Saves the figure to the specified path, applying `tight_layout` before saving.
    
    Parameters:
    - fig: Matplotlib Figure object to save.
    - save_path: Path (including filename) to save the figure to.
    - dpi: Resolution in dots per inch for the saved figure (default 300).
    """
    # Ensure the directory exists
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Apply tight layout for final formatting
    fig.tight_layout()
    
    # Save the figure
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

def generate_samples_figure(
        image_dict, 
        row_labels, 
        plot_params, 
        color_dict, 
        title='Example Faces',
        text_box=None,
        save_path=None,
        dpi=150
):
    fig, axes = plot_matrix(image_dict=image_dict, row_labels=row_labels)
    apply_ticks(axes, plotting_params=plot_params)
    set_spines_and_titles_by_column(axes, title_colors=color_dict)
    if title:
        add_figure_title(fig, plotting_params=plot_params, title_text=title)
    if text_box:
        add_text_box(fig, plotting_params=plot_params, text=text_box)
    save_figure(fig, save_path=save_path, dpi=dpi)

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def run_dimensionality_reduction(
        X, 
        y, 
        model_dict, 
        normalization, 
        total_components, 
        components_for_reconstruction):
    # Dictionary to store the averaged reconstructions, starting with 'Overall'
    results = {"Overall": None}
    
    # Apply normalization if specified
    if normalization == 'minmax':
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    elif normalization == 'standard':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X  # No normalization

    # Instantiate the model (provided by your external model loading function)
    model_dict['n_components']=total_components
    model = instantiate_model(model_dict)

    # Fit the model with the maximum number of components
    X_transformed = model.fit_transform(X_scaled)

    # Track valid components actually used for reconstruction
    valid_components_used = []

    # Reconstruct data for 'Overall' using the specified components to illustrate additive contributions
    overall_reconstructions = []
    for n_components in components_for_reconstruction:
        if n_components > total_components:
            raise ValueError(f"Requested components ({n_components}) exceed total_components ({total_components}).")
        
        # Zero out components beyond `n_components`
        X_partial = np.zeros_like(X_transformed)
        X_partial[:, :n_components] = X_transformed[:, :n_components]

        # Create partial reconstruction using the first `n_components` components
        partial_reconstruction = model.inverse_transform(X_partial)
        overall_reconstructions.append(partial_reconstruction)
        
        # Track valid component number
        valid_components_used.append(n_components)
    
    # Average overall reconstructions
    averaged_overall_reconstruction = np.mean(overall_reconstructions, axis=0)
    results["Overall"] = averaged_overall_reconstruction

    # Process each category in `y` for category-wise reconstructions
    unique_categories = np.unique(y)
    for category in unique_categories:
        category_mask = (y == category)
        
        category_reconstructions = []
        for n_components in components_for_reconstruction:
            # Zero out components beyond `n_components` for the category subset
            X_partial_category = np.zeros_like(X_transformed[category_mask])
            X_partial_category[:, :n_components] = X_transformed[category_mask, :n_components]
  
            # Partial reconstruction for this category
            partial_reconstruction = model.inverse_transform(X_transformed[category_mask, :n_components])
            category_reconstructions.append(partial_reconstruction)

        # Average reconstructions across components for this category
        averaged_category_reconstruction = np.mean(category_reconstructions, axis=0)
        results[category] = averaged_category_reconstruction

    return results, valid_components_used



