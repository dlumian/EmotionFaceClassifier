import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from matplotlib.patches import Rectangle
import numpy as np

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


def plot_facial_expressions(
        image_dict, 
        row_labels=None, 
        label_colors=None, 
        save_path='imgs/test_examples',
        file_name='sample_images.png'
    ):
    # Default save path if not provided
    base_save_path = save_path if save_path else '../imgs/'
        # Ensure the base path exists
    os.makedirs(base_save_path, exist_ok=True)
    
    # Check if image_dict contains paths or arrays and read image files if necessary
    for category, images in image_dict.items():
        for i, img in enumerate(images):
            if isinstance(img, str):  # Image is a file path, load it
                image_dict[category][i] = np.array(Image.open(img).convert('L'))  # Convert to grayscale

    # Determine the number of rows based on the largest category
    max_images_per_category = max(len(images) for images in image_dict.values())
        
    # Validate row_labels if provided
    if row_labels:
        if len(row_labels) != max_images_per_category:
            raise ValueError(f"Row labels must match the number of rows in the dataset: {max_images_per_category} expected.")
    
    # Adjust columns if row_labels are provided
    num_categories = len(image_dict)
    if row_labels:
        columns = num_categories + 1  # Include an extra column for row labels
    else:
        columns = num_categories

    # Create the subplot grid
    fig, axes = plt.subplots(max_images_per_category, columns, figsize=(columns * 3, max_images_per_category * 3))
    
    # Plot each image in the correct spot and save individual images
    for row in range(max_images_per_category):
        if row_labels:  # Add row labels to the first column
            if row < len(row_labels):
                axes[row, 0].text(0.5, 0.5, row_labels[row], fontsize=24, va='center', ha='center')
            axes[row, 0].axis('off')
        
        for col, (category, images) in enumerate(image_dict.items()):
            # Adjust column index if row_labels are used
            image_col = col + 1 if row_labels else col
            
            # Check if the current category has an image for this row
            if row < len(images):
                img = images[row]
                axes[row, image_col].imshow(img, cmap='gray')  # Display image in grayscale
                axes[row, image_col].axis('off')
                
                # Add a colored border if label_colors are provided
                if label_colors and category in label_colors:
                    color = label_colors[category]
                else:
                    color = 'black'
                rect = Rectangle((0, 0), img.shape[1] - 1, img.shape[0] - 1, 
                             linewidth=6, edgecolor=color, facecolor='none')
                axes[row, image_col].add_patch(rect)

                # Set the title with emphasis (larger font size and bold) for the first row of each column
                if row == 0:
                    axes[row, image_col].set_title(category, fontsize=24, fontweight='bold', pad=10, color=color)
        
            else:
                axes[row, image_col].axis('off')  # No image for this row in this category
    
    # Adjust subplot spacing
    plt.tight_layout()
    
    # Save the subplot comparison image
    if save_path:
        comparison_image_path = os.path.join(save_path, file_name or 'facial_expressions.png')
        fig.savefig(comparison_image_path)
        print(f"Comparison subplot saved to {comparison_image_path}")

    # Close the figure to prevent it from being displayed twice
    plt.close(fig)    
    # Return the figure object for further modification
    return fig