import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def plot_facial_expressions(
        image_dict, 
        row_labels=None, 
        label_colors=None, 
        save_path=None,
        file_name=None 
    ):
    """
    Plot facial expression images in subplots. Supports coloring borders by category, row labels, and saving to disk.
    
    Parameters:
    - image_dict (dict): A dictionary where the keys are categories and values are lists of image file paths or arrays.
    - row_labels (list): Optional list of row labels, must match the number of rows in the dataset.
    - label_colors (dict): Optional dictionary to color-code borders of the images by category.
    - save_path (str): Optional path to save the resulting figure. If not provided, the plot is just shown.
    - columns (int): Number of columns in the subplot (including row labels if present).
    
    Returns:
    - fig (matplotlib.figure.Figure): The figure object for further modifications.
    """
    # Validate inputs
    if row_labels and len(row_labels) != len(next(iter(image_dict.values()))):
        raise ValueError("Row labels must match the number of rows in the image dataset.")
    
    # Check if image_dict contains paths or arrays and read image files if necessary
    for category, images in image_dict.items():
        for i, img in enumerate(images):
            if isinstance(img, str):  # Image is a file path, load it
                image_dict[category][i] = np.array(Image.open(img))
    
    # Determine the number of rows and columns
    num_categories = len(image_dict)
    num_images_per_category = len(next(iter(image_dict.values())))  # All categories should have same number of images
    
    if row_labels:
        # columns = min(columns, num_categories + 1)  # Include an extra column for row labels
        columns = num_categories + 1
    else:
        columns = num_categories
    
    # Create the subplot grid
    fig, axes = plt.subplots(num_images_per_category, columns, figsize=(columns * 3, num_images_per_category * 3))
    
    # Plot each image in the correct spot
    for row in range(num_images_per_category):
        if row_labels:  # Add row labels to the first column
            axes[row, 0].text(0.5, 0.5, row_labels[row], va='center', ha='center')
            axes[row, 0].axis('off')
        
        for col, (category, images) in enumerate(image_dict.items()):
            # Adjust column index if row_labels are used
            image_col = col + 1 if row_labels else col
            
            # Show the image
            axes[row, image_col].imshow(images[row])
            axes[row, image_col].axis('off')
            
            # Add a colored border if label_colors are provided
            if label_colors and category in label_colors:
                color = label_colors[category]
                rect = patches.Rectangle((0, 0), images[row].shape[1], images[row].shape[0], 
                                         linewidth=5, edgecolor=color, facecolor='none')
                axes[row, image_col].add_patch(rect)
    
    # Adjust subplot spacing
    plt.tight_layout()
    
    # Save the plot to disk if a path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, 'facial_expressions.png')
        fig.savefig(save_file)
        print(f"Plot saved to {save_file}")
    
    # Return the figure object for further modification
    return fig

def generate_sample_images(df, n=3, cat_col='emotion', path_col='img_path'):
    # Ensure the number of rows is within the allowed range
    if not (1 <= n <= 5):
        raise ValueError("Number rows must be between 1 and 5")
    
    categories = df[cat_col].unique()

    # Pre-sample the random images for each category and store them in a dictionary
    random_samples = {}
    for category in categories:
        category_df = df[df[cat_col] == category]
        random_samples[category] = df[path_col].sample(n=n).tolist()
    return random_samples