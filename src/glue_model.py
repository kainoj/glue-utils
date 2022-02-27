from pytorch_lightning import LightningModule

class GlueModel(LightningModule):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        pass
 
    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass