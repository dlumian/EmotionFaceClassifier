import importlib


def instantiate_model(model_config):
    'Function to create model instances from the configuration'
    module = importlib.import_module(model_config['module'])
    model_class = getattr(module, model_config['class'])
    model= model_class(**model_config['params'])
    return model
