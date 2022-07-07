from torch import optim, nn
from statistics import mean
import pytorch_lightning as pl
import torch
class Simple(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.input_size = params.get("input_size")
        self.output_size = params.get("output_size")
        self.hidden_size = params.get("hidden_size")
        self.hidden_count = params.get("hidden_count")
        self.lr = params.get("lr")


        self.input_layer = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            # nn.Sigmoid()
        )

        self.hidden_layers = nn.Sequential()
        for _ in range(self.hidden_count):
            hidden = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU()
            )
            self.hidden_layers.append(hidden)
        
    
    def forward(self, x):
        x = self.input_layer(x)
        if self.hidden_count > 0:
            x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)

        loss = self.loss(out, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)

        loss = self.loss(out, y)
        accuracy = self.accuracy(out, y)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)

        loss = self.loss(out, y)
        # Logging to TensorBoard by default
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, pass_idx = batch
        
        out = self.forward(x)

        prediction = self.make_prediction(out)
        prediction = torch.tensor(prediction)
        combo = torch.cat((pass_idx.unsqueeze(1), prediction), 1)

        return combo
        

    def loss(self, out, y):
        loss_func = nn.BCEWithLogitsLoss()
        loss = loss_func(out, y)
        return loss

    def accuracy(self, out, y):
        y = y.numpy()
        prediction =self.make_prediction(out)
        accuracy = abs(prediction - y)
        accuracy = 1 - accuracy.mean()
        return accuracy


    def make_prediction(self, out):
        sig = nn.Sigmoid()
        prediction = sig(out).numpy()
        prediction = torch.tensor(prediction.round())
        return prediction
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer