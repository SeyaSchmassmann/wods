import os
import torch
from torchmetrics import Accuracy, MulticlassF1Score
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb


class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.train_acc = Accuracy(task="multiclass", num_classes=model.num_classes)
        self.train_f1 = MulticlassF1Score(num_classes=model.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=model.num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=model.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=model.num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=model.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.train_acc.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.val_acc.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):
        self.test_acc.reset()
        self.test_f1.reset()


    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train_test_model(model, train_loader, val_loader, test_loader, epochs=30):
    lit_model = LitModel(model)

    try:
        wandb.login(key=os.getenv('API_KEY_WANDB'))

        wandb_logger = WandbLogger(entity="wods", project="wods")
        wandb_logger.experiment.config["batch_size"] = 64
        wandb_logger.experiment.config["epochs"] = epochs

        trainer = pl.Trainer(max_epochs=epochs,
                             accelerator="auto",
                             devices="auto",
                             logger=wandb_logger)

        trainer.fit(lit_model, train_loader, val_loader)
        trainer.test(lit_model, test_loader)

        wandb.finish()
    except Exception as e:
        wandb.finish()
        raise e