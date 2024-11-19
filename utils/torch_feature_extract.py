import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import time
import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Check for GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logging.info(f"Using device: {device}")

class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_
        U, S, V = torch.pca_lowrank(X_centered, q=self.n_components)
        self.components_ = V.T

    def transform(self, X):
        X_centered = X - self.mean_
        return torch.matmul(X_centered, self.components_.T)

    def inverse_transform(self, X_transformed):
        return torch.matmul(X_transformed, self.components_) + self.mean_

def calculate_metrics(X_true, X_pred):
    X_true_np = X_true.cpu().numpy()
    X_pred_np = X_pred.cpu().numpy()
    
    mse = mean_squared_error(X_true_np, X_pred_np)
    psnr = 10 * np.log10((255**2) / mse)  # Assuming pixel values are in [0, 255]
    
    # Reshape if necessary (assuming images are square)
    img_size = int(np.sqrt(X_true_np.shape[1]))
    X_true_2d = X_true_np.reshape(-1, img_size, img_size)
    X_pred_2d = X_pred_np.reshape(-1, img_size, img_size)
    
    ssim_value = ssim(X_true_2d, X_pred_2d, 
                      data_range=X_true_2d.max() - X_true_2d.min(), 
                      multichannel=True)
    
    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_value
    }

def run_single_analysis(X, y, analysis_config):
    start_time = time.time()
    logging.info("Starting analysis")

    n_components = analysis_config['total_components']
    model = PCA(n_components).to(device)

    logging.info("Fitting PCA model")
    model.fit(X)
    
    logging.info("Transforming data")
    features = model.transform(X)

    results = []
    for category in torch.unique(y):
        X_category = X[y == category]
        features_category = features[y == category]

        for recon_components in analysis_config['components_for_reconstruction']:
            logging.info(f"Processing category {category.item()} with {recon_components} components")
            
            partial_features = torch.zeros_like(features_category)
            partial_features[:, :recon_components] = features_category[:, :recon_components]
            
            recon_images = model.inverse_transform(partial_features)
            avg_image = torch.mean(recon_images, dim=0)

            metrics = calculate_metrics(X_category, recon_images)

            results.append({
                'category': category.item(),
                'components': recon_components,
                'avg_image': avg_image.cpu().numpy(),
                'metrics': metrics
            })

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Analysis completed in {total_time:.2f} seconds")

    return results, total_time

# # Assume X and y are loaded as numpy arrays
# X = torch.tensor(X, dtype=torch.float32).to(device)
# y = torch.tensor(y).to(device)

# analysis_config = {
#     'total_components': 100,
#     'components_for_reconstruction': [1, 10, 30, 50, 100]
# }

# results, total_time = run_single_analysis(X, y, analysis_config)

# # Saving results
# save_path = 'pca_results.pt'
# torch.save({
#     'results': results,
#     'total_time': total_time,
#     'config': analysis_config
# }, save_path)
# logging.info(f"Results saved to {save_path}")

# # If you prefer numpy compressed format:
# np_results = np.array(results, dtype=object)
# np.savez_compressed('pca_results.npz', results=np_results, total_time=total_time, config=analysis_config)
# logging.info("Results also saved in numpy compressed format")