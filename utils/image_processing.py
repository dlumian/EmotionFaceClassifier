import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import cv2
from sklearn.decomposition import PCA
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

def preprocess_image(image, image_size=(64, 64), normalize=True):
    """
    Preprocess a facial image: resize and normalize.
    Args:
        image: Input image (as a NumPy array).
        image_size: Size to resize the image to.
        normalize: Whether to normalize the pixel values.
    Returns:
        Preprocessed image.
    """
    image_resized = cv2.resize(image, image_size)
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    if normalize:
        image_gray = image_gray / 255.0
    return image_gray

# Utility function for dimensionality reduction using PCA
def apply_pca(images, n_components=50):
    """
    Apply PCA to reduce dimensionality of a list of images.
    Args:
        images: List of preprocessed images (as 2D arrays).
        n_components: Number of principal components to keep.
    Returns:
        Transformed images in the reduced dimensional space.
    """
    flat_images = [img.flatten() for img in images]
    pca = PCA(n_components=n_components)
    transformed_images = pca.fit_transform(flat_images)
    return transformed_images, pca

# Utility function to compute the average face for a category of images
def compute_average_face(images):
    """
    Compute the average face for a list of images.
    Args:
        images: List of preprocessed images (as 2D arrays).
    Returns:
        The average face (as a 2D array).
    """
    average_face = np.mean(images, axis=0)
    return average_face

# Utility function to track and log the PCA results and average face in MLflow
def log_pca_results(pca, average_face, emotion_category):
    """
    Log PCA results and average face in MLflow for tracking.
    Args:
        pca: The fitted PCA model.
        average_face: The computed average face.
        emotion_category: The category of emotion (for logging purposes).
    """
    pca_array_dir = os.path.join('data', 'pca_arrays')
    os.makedirs(pca_array_dir, exist_ok=True)

    facial_imgs = os.path.join('imgs', 'facial_features')
    os.makedirs(facial_imgs, exist_ok=True)

    with mlflow.start_run(run_name=f"PCA_{emotion_category}"):
        mlflow.log_param("n_components", pca.n_components_)
        mlflow.log_metric("explained_variance_ratio", np.sum(pca.explained_variance_ratio_))
        
        # Save and log the average face
        average_face_path = os.path.join(facial_imgs, f"average_face_{emotion_category}.png")
        plt.imsave(average_face_path, average_face, cmap='gray')
        mlflow.log_artifact(average_face_path)

        # Save PCA components and log
        components_path = os.path.join(pca_array_dir, f"pca_components_{emotion_category}.npy")
        np.save(components_path, pca.components_)
        mlflow.log_artifact(components_path)

        print(f"Logged PCA results for emotion category: {emotion_category}")
