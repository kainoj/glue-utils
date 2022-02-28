from pytorch_lightning import LightningModule
from transformers import AdamW
from datasets import load_metric

class GlueModel(LightningModule):

    def __init__(self, model, learning_rate, task_name) -> None:
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.metric = load_metric("glue", task_name)
        self.task_name = task_name
        self.is_regression = self.task_name == "stsb"

    def step(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        
        outputs = self.step(batch)

        loss = outputs.loss


        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        # preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        # preds = torch.squeeze(preds) if self.is_regression else torch.argmax(preds, axis=1)
    
        # result = self.metric.compute(predictions=preds, references=p.label_ids)
        # if len(result) > 1:
        #     result["combined_score"] = torch.mean(list(result.values())).item()
        
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        # todo sscheduler
        return AdamW(self.model.parameters(), lr=self.lr)