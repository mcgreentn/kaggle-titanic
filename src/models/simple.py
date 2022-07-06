from torch import optim, nn
import pytorch_lightning as pl

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
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)

        loss = self.loss(out, y)
        # Logging to TensorBoard by default
        self.log("test_loss", loss)
        return loss

    
    def loss(out, y):
        loss = nn.functional.cross_entropy(out, y)
        return loss

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer