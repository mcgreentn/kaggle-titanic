import pytorch_lightning as pl
from datasets.titanic import Titanic as TDataset
from torch.utils.data import random_split, DataLoader

class Titanic(pl.LightningDataModule):
    def __init__(self, config_params, data_params):
        super().__init__()
        self.config_params = config_params
        self.data_params = data_params

        train_params = self.data_params.get("train", {})
        self.train_batch_size = train_params.get("batch_size")
        self.train_filename = train_params.get("filename")
        test_params = self.data_params.get("test", {})
        self.test_batch_size = test_params.get("batch_size")
        self.test_filename = test_params.get("filename")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            data_full = TDataset(self.config_params, self.train_filename)
            data_full.setup()
            split = [round(len(data_full) * 0.8), round(len(data_full) * 0.2)]
            self.data_train, self.data_val = random_split(data_full, split)
            self.data_test = TDataset(self.config_params, self.test_filename, data_type="test")
            self.data_test.setup()
        elif stage == "test":
            self.data_train = TDataset(self.config_params, self.train_filename)
            self.data_train.setup()
            self.data_test = TDataset(self.config_params, self.test_filename, data_type="test")
            self.data_test.setup()


    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.train_batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.train_batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.test_batch_size)