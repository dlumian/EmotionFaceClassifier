import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import mlflow
import os


def display_random_images_from_df(df, n_rows=3):
    """
    Display randomly sampled images from each category in the DataFrame with colored outlines.
    
    Parameters:
    - df: pandas DataFrame containing 'Category', 'File Path', and 'Color' columns.
    - n_rows: Number of rows to display (default=3, must be between 1 and 5).
    """
    
    # Ensure the number of rows is within the allowed range
    if not (1 <= n_rows <= 5):
        raise ValueError("n_rows must be between 1 and 5")
    
    # Get the unique categories from the DataFrame
    categories = df['emotion'].unique()
    
    # Pre-sample the random images for each category and store them in a dictionary
    random_samples = {}
    for category in categories:
        category_df = df[df['emotion'] == category]
        random_samples[category] = category_df.sample(n=n_rows)  # Pre-sample all the rows
    
    # Create the plot
    fig, axes = plt.subplots(n_rows, len(categories), figsize=(len(categories) * 3, n_rows * 3))
    
    for i, category in enumerate(categories):
        # Retrieve the pre-sampled images for the current category
        category_sample = random_samples[category]
        
        for row in range(n_rows):
            # Get the image file path and color from the pre-sampled data
            img_path = category_sample.iloc[row]['img_path']
            img_color = category_sample.iloc[row]['color']
            
            # Load the image
            img = Image.open(img_path).convert('L')  # Convert to grayscale if needed
            
            # Plot the image
            ax = axes[row, i] if n_rows > 1 else axes[i]  # Handle case where n_rows = 1 (axes is 1D)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            # Add colored outline using the color from the DataFrame
            rect = Rectangle((0, 0), img.size[0] - 1, img.size[1] - 1, 
                             linewidth=4, edgecolor=img_color, facecolor='none')
            ax.add_patch(rect)
            
            # Set the title with emphasis (larger font size and bold) for the first row of each column
            if row == 0:
                ax.set_title(category, fontsize=24, fontweight='bold', pad=10, color=img_color)
    
    plt.tight_layout()
    plt.show()


def load_images_from_df(df):
    """
    Load images from the file paths in the DataFrame and return them as numpy arrays.
    
    Parameters:
    - df: pandas DataFrame containing 'File Path' and 'Category' columns.
    
    Returns:
    - A dictionary where keys are categories and values are lists of images as numpy arrays.
    """
    category_images = {}
    
    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        category = row['emotion']
        file_path = row['img_path']
        
        # Load image, convert to grayscale, and convert to numpy array
        img = Image.open(file_path).convert('L')
        img_array = np.array(img)
        
        # Store in dictionary
        if category not in category_images:
            category_images[category] = []
        category_images[category].append(img_array)
    
    return category_images

def compute_mean_image(images):
    """
    Compute the mean image from a list of images.
    
    Parameters:
    - images: List of images as numpy arrays.
    
    Returns:
    - The mean image as a numpy array.
    """
    return np.mean(images, axis=0)

def compute_median_image(images):
    """
    Compute the median image from a list of images.
    
    Parameters:
    - images: List of images as numpy arrays.
    
    Returns:
    - The median image as a numpy array.
    """
    return np.median(images, axis=0)

def plot_image(image, title="Average Image", cmap='gray'):
    """
    Plot a single image with a title.
    
    Parameters:
    - image: The image to plot as a numpy array.
    - title: Title for the image plot.
    - cmap: Colormap for the image plot (default is grayscale).
    """
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_image(image, path):
    """
    Save an image to the specified path.
    
    Parameters:
    - image: The image to save as a numpy array.
    - path: The file path to save the image to.
    """
    img = Image.fromarray(np.uint8(image))
    img.save(path)

def log_and_save_image(image, category, variation, output_dir):
    """
    Log an image and save it locally and to MLflow.
    
    Parameters:
    - image: The image to log and save.
    - category: The category (emotion) name.
    - variation: The variation of the average image process.
    - output_dir: Directory to save the image locally.
    """
    # Save the image locally
    file_path = os.path.join(output_dir, f"{category}_{variation}.png")
    save_image(image, file_path)
    
    # Log the image in MLflow
    mlflow.log_artifact(file_path)
    
    return file_path

def process_mean_images(df, output_dir="outputs"):
    """
    Process the mean images for each category and log results with MLflow.
    
    Parameters:
    - df: pandas DataFrame with 'Category', 'File Path', and 'Color' columns.
    - output_dir: Directory to save the images locally.
    """
    # Start an MLflow run
    with mlflow.start_run(run_name="Mean Image Processing"):
        # Load images by category
        category_images = load_images_from_df(df)
        
        # Process mean images
        for category, images in category_images.items():
            mean_image = compute_mean_image(images)
            mlflow.log_param(f"{category}_method", "mean")
            
            # Save and log the image
            file_path = log_and_save_image(mean_image, category, "mean", output_dir)
            print(f"Saved and logged mean image for {category}: {file_path}")

def process_median_images(df, output_dir="outputs"):
    """
    Process the median images for each category and log results with MLflow.
    
    Parameters:
    - df: pandas DataFrame with 'Category', 'File Path', and 'Color' columns.
    - output_dir: Directory to save the images locally.
    """
    with mlflow.start_run(run_name="Median Image Processing"):
        # Load images by category
        category_images = load_images_from_df(df)
        
        # Process median images
        for category, images in category_images.items():
            median_image = compute_median_image(images)
            mlflow.log_param(f"{category}_method", "median")
            
            # Save and log the image
            file_path = log_and_save_image(median_image, category, "median", output_dir)
            print(f"Saved and logged median image for {category}: {file_path}")

