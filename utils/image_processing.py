import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

# PCA and NMF
def apply_pca(X, n_components=100):
    """Apply PCA to the dataset."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)
    return X_pca, X_reconstructed, pca.explained_variance_ratio_.sum()

def apply_nmf(X, n_components=100):
    """Apply NMF to the dataset."""
    nmf = NMF(n_components=n_components, init='random', random_state=0)
    X_nmf = nmf.fit_transform(X)
    X_reconstructed = np.dot(X_nmf, nmf.components_)
    reconstruction_error = np.linalg.norm(X - X_reconstructed)
    return X_nmf, X_reconstructed, reconstruction_error

# MLflow Tracking
def track_with_mlflow(X, method='PCA', n_components_list=[50, 100, 150]):
    """Track analysis results using MLflow for multiple n_components values."""
    for n_components in n_components_list:
        with mlflow.start_run(run_name=f"{method}_n_components_{n_components}"):
            if method == 'PCA':
                _, X_reconstructed, variance_explained = apply_pca(X, n_components)
                mlflow.log_metric('variance_explained', variance_explained)
            elif method == 'NMF':
                _, X_reconstructed, reconstruction_error = apply_nmf(X, n_components)
                mlflow.log_metric('reconstruction_error', reconstruction_error)
            
            # Save reconstructed images as artifacts
            save_path = f"imgs/{method.lower()}_n_{n_components}"
            os.makedirs(save_path, exist_ok=True)
            for i in range(8):  # Save first 8 examples
                img = X_reconstructed[i].reshape(48, 48).astype(np.uint8)
                img_path = os.path.join(save_path, f"{method}_n_{n_components}_example_{i}.png")
                Image.fromarray(img).save(img_path)
                mlflow.log_artifact(img_path)

def compute_average_face(X, labels):
    """
    Compute average faces for each emotion category, and prepend the combined average face for all images.
    
    Args:
        X: Array of images.
        labels: Array of corresponding emotion labels.
    
    Returns:
        avg_faces: List of average face arrays, with the combined face at the start.
        emotion_labels: List of emotion labels, with 'Combined' at the start.
    """
    unique_labels = np.unique(labels)
    avg_faces = []
    
    # Compute the average face for each emotion category
    for label in unique_labels:
        label_mask = labels == label
        avg_face = np.mean(X[label_mask], axis=0)
        avg_faces.append(avg_face)
    
    # Compute combined average face for the whole dataset
    combined_face = np.mean(X, axis=0)
    
    # Prepend the combined face and 'Combined' label to the lists
    avg_faces.insert(0, combined_face)  # Add combined face at the start
    emotion_labels = ['Combined'] + list(unique_labels)  # Add 'Combined' label at the start
    
    return avg_faces, emotion_labels

def save_and_plot_images(image_list, labels, save_path, title="Images", label_colors=None, use_colors=False):
    """
    Save and plot images in a comparison grid, and save each image individually.
    
    Args:
        image_list: List of image arrays to display and save.
        labels: List of labels corresponding to the images.
        save_path: Path to save images and the comparison plot.
        title: Title for the entire plot.
        label_colors: Optional list of colors for the labels (same length as labels).
        use_colors: Whether to color-code the titles based on the label_colors list.
    """
    n_images = len(image_list)
    
    # Create subplots and add a larger main title
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 3, 3))
    fig.suptitle(title, fontsize=28)  # Larger font size for the main title
    
    # Plot and save each image individually
    for i, (image, label) in enumerate(zip(image_list, labels)):
        ax = axes[i]
        ax.imshow(image.reshape(48, 48), cmap='gray')
        
        # Use color for the title if specified, otherwise default to black
        title_color = label_colors[i] if use_colors and label_colors is not None else 'black'
        ax.set_title(label, fontsize=20, color=title_color)  # Increased font size for column titles
        ax.axis('off')
        
        # Save individual image
        img = Image.fromarray(image.reshape(48, 48).astype(np.uint8), 'L')
        img_path = os.path.join(save_path, f'{label.lower().replace(" ", "_")}.png')
        img.save(img_path)
    
    # Save the comparison plot
    os.makedirs(save_path, exist_ok=True)
    comparison_path = os.path.join(save_path, 'comparison_plot.png')
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()

def display_sample_images(df, n_rows=3, save_path=None):
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
                             linewidth=6, edgecolor=img_color, facecolor='none')
            ax.add_patch(rect)
            
            # Set the title with emphasis (larger font size and bold) for the first row of each column
            if row == 0:
                ax.set_title(category, fontsize=24, fontweight='bold', pad=10, color=img_color)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()