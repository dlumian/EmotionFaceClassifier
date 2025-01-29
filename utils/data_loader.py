import os
import pandas as pd
from datascifuncs import load_json

class DataLoader:
    def __init__(self, settings=True, unsupervised=True, vectorized=True, 
                 initial_data=True, processed_data=True): 
        self.file_paths = {
            'settings': './configs/settings.json',
            'unsupervised': './configs/unsupervised_models.json',
            'vectorized': './configs/vectorized_models.json',
            'initial_data': 'data/fer2013.csv',
            'processed_data': 'data/fer2013_processed.csv'
        }
        self.load_flags = {
            'settings': settings,
            'unsupervised': unsupervised,
            'vectorized': vectorized,
            'initial_data': initial_data,
            'processed_data': processed_data
        }
        self.data = {}

    def load_csv_data(self, file_path):
        return pd.read_csv(file_path)

    def load_json_data(self, file_path):
        return load_json(file_path)

    def load_all_data(self):
        for key, should_load in self.load_flags.items():
            if should_load and os.path.exists(self.file_paths[key]):
                if '_data' in key:
                    self.data[key] = self.load_csv_data(self.file_paths[key])
                else:
                    self.data[key] = self.load_json_data(self.file_paths[key])
            else:
                self.data[key] = None
        self.set_data_as_attributes()

    def set_data_as_attributes(self):
        for key, value in self.data.items():
            if key == 'settings' and value is not None:
                for sub_key, sub_value in value.items():
                    setattr(self, sub_key, sub_value)
            setattr(self, key, value)