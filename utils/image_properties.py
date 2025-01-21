import os
import numpy as np
import matplotlib.pyplot as plt

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

        axes[idx].spines['bottom'].set_linewidth(8)
        axes[idx].spines['bottom'].set_edgecolor('black')

        # Remove the left, top and right spines
        axes[idx].spines['left'].set_visible(False)
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









