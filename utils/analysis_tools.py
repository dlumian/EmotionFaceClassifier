import time
import logging
import warnings
import importlib
from functools import wraps
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # Log the timing
        logging.info(f'Function {func.__name__} Took {total_time:.4f} seconds')
        # Print the timing (optional)
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        
        return result
    return timeit_wrapper

def suppress_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return func(*args, **kwargs)
    return wrapper

def data_loader(all=True, mapping_json=False, df=False,
                unsupervised_json=False, vectorized_json=False):

def instantiate_model(model_config):
    'Function to create model instances from the configuration'
    module = importlib.import_module(model_config['module'])
    model_class = getattr(module, model_config['class'])
    model= model_class(**model_config['params'])
    return model

def normalize_data(data, normalizer='none'):
    """Normalize the data using the specified method."""
    if normalizer == 'none':
        return data, None
    elif normalizer == 'minmax':
        scaler = MinMaxScaler()
        return scaler.fit_transform(data), scaler
    elif normalizer == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(data), scaler
    else:
        raise ValueError(f"Unknown normalization method: {normalizer}")