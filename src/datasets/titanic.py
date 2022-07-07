import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class Titanic(Dataset):
    def __init__(self, config_params, filename):
        self.input_loc = config_params.get("input_loc")
        self.filename = filename
    
    def setup(self):
        self.raw_data = self.read_in_data()
    
    def read_in_data(self):
        df = pd.read_csv(os.path.join(self.input_loc, self.filename))
        return df

    def __getitem__(self, index):
        y = self.raw_data[["Survived_1", "Survived_2"]]
        x = self.raw_data.drop(["Survived_1", "Survived_2", "Unnamed: 0"], axis=1)
        x = x.iloc[index].to_numpy().astype(np.float32)
        target = y.iloc[index].to_numpy().astype(np.float32)
        return x, target
    
    def __len__(self):
        return len(self.raw_data)