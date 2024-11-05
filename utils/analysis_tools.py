import importlib
from copy import deepcopy
from collections.abc import Mapping

def instantiate_model(model_config):
    'Function to create model instances from the configuration'
    module = importlib.import_module(model_config['module'])
    model_class = getattr(module, model_config['class'])
    model= model_class(**model_config['params'])
    return model

def deep_update(base, updates):
    """
    Recursively updates nested dictionaries. Values in `updates` override those in `base`.
    Automatically creates a deepcopy of `base` to prevent modifications to the original dictionary.
    """
    base = deepcopy(base)  # Ensure base is not modified
    for key, value in updates.items():
        if isinstance(value, Mapping) and key in base:
            base[key] = deep_update(base.get(key, {}), value)
        else:
            base[key] = value
    return base
