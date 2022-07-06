from pyparsing import Optional
import pytorch_lightning as pl
from src.datasets.titanic import Titanic
from torch.utils.data import random_split, DataLoader

class Titanic(pl.DataModule):
    def __init__(self, config_params, data_params):
        super().__init__()
        self.data_params = data_params

        train_params = data_params.get("train", {})
        self.train_batch_size = train_params.get("batch_size")
        self.train_filename = train_params.get("filename")
        test_params = data_params.get("test", {})
        self.test_batch_size = test_params.get("batch_size")
        self.test_filename = test_params.get("filename")

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            data_full = Titanic(self.config_params, self.train_filename)
            split = [round(data_full.length * 0.8), round(data_full.length * 0.2)]
            self.data_train, self.data_val = random_split(data_full, split)
            self.data_test = Titanic(self.config_params, self.test_filename)
        elif stage == "test":
            self.data_test = Titanic(self.config_params, self.test_filename)

    
    def train_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.train_batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.train_batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.test_batch_size)