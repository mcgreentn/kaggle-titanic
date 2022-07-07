import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class Titanic(Dataset):
    def __init__(self, config_params, filename, data_type="train"):
        self.input_loc = config_params.get("input_loc")
        self.filename = filename
        self.data_type = data_type
    
    def setup(self):
        self.raw_data = self.read_in_data()
    
    def read_in_data(self):
        df = pd.read_csv(os.path.join(self.input_loc, self.filename))
        return df

    def __getitem__(self, index):
        if self.data_type == "train":
            y = self.raw_data[["Survived"]]
            x = self.raw_data.drop(["Survived", "Unnamed: 0"], axis=1)
            x = x.iloc[index].to_numpy().astype(np.float32)
            target = y.iloc[index].to_numpy().astype(np.float32)
            return x, target
        elif self.data_type == "test":
            pass_idx = self.raw_data["PassengerId"].iloc[index]
            x = self.raw_data.drop(["PassengerId","Survived", "Unnamed: 0"], axis=1)
            x = x.iloc[index].to_numpy().astype(np.float32)
            return x, pass_idx
    
    def __len__(self):
        return len(self.raw_data)