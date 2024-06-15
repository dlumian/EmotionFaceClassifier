

class VectorClassifier():
    def __init__(self, df, mdl_dict) -> None:
        self.df = df
        self.mdl_dict = mdl_dict

    def train_test_split(self):
        