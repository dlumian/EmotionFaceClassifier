import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def convert_pixels_to_array(pixels):
    """
    Convert a space-separated string of pixels into a 2-D numpy array.

    Parameters:
        pixels (str): A string containing space-separated pixel values.

    Returns:
        numpy.ndarray: A 2-D numpy array with shape (48, 48) and dtype 'uint8'.
    """
    # Convert the space-separated string into a list of integers
    pixel_values = [int(x) for x in pixels.split(' ')]

    # Reshape the list into a 48x48 2-D array
    array = np.array(pixel_values).reshape(48, 48)

    return np.array(array, dtype='uint8').reshape(48, 48)

def save_image(row):
    # Construct directory path and ensure it exists
    """
    Save an image to a specified directory path based on its emotion and usage.

    Parameters:
        row (pandas.Series): A row containing image data and metadata, 
                             with keys 'emotion', 'usage', 'emo_count_id', and 'image'.

    Returns:
        str: The file path where the image was saved.
    """
    
    base_dir = 'data'
    emotion_label = row['emotion']
    directory_path = os.path.join(base_dir, row['usage'], emotion_label)
    os.makedirs(directory_path, exist_ok=True)
    
    # Construct the final image file name and path
    file_name = f"{emotion_label}-{row['emo_count_id']}.jpg"
    image_path = os.path.join(directory_path, file_name)
    
    # Convert the image array to grayscale and save
    image = Image.fromarray(row['image'].astype('uint8'), 'L')
    image.save(image_path)
    
    return image_path

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

def plot_face_matrix(
        image_dict, 
        row_labels=None,
        group_colors=None, 
        save_path=None,
        main_title=None,
        box_text=None
):
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
    if main_title:
        fig.suptitle(main_title, fontsize=26, fontweight='bold', y=1.05)
    
    if box_text:
        fig.text(0.95, 1.05, box_text, ha='right', va='top', fontsize=20, transform=fig.transFigure)

    if row_labels:
        for title, labels in row_labels.items():
            axes[0, 0].set_title(title, fontsize=24, fontweight='bold', pad=10)
            for row_idx, row_label in enumerate(labels):
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

                for spine in ax.spines.values():
                    spine.set_linewidth(7)
                    spine.set_edgecolor(color)

                # Set the title with emphasis (larger font size and bold) for the first row of each column
                if row_idx == 0:
                    ax.set_title(col_label, fontsize=24, fontweight='bold', pad=10, color=color)
            # ax.axis('off')  # Hide axis ticks
            ax.set_xticks([])
            ax.set_yticks([])

    # Save the plot if a save_path is given
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.tight_layout()
    plt.show()
    return None