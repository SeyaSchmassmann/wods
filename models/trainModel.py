import os
import torch
from timm.scheduler import CosineLRScheduler
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime


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
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=True)
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

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.trainer.max_epochs,
            lr_min=1e-6,
            warmup_lr_init=1e-6,
            warmup_t=5,
            t_in_epochs=True
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)


def run_single_training(model, train_loader, val_loader, test_loader, epochs, run_name_suffix=""):
    model = model()
    lit_model = LitModel(model)

    try:
        wandb.login(key=os.getenv('API_KEY_WANDB'))

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{timestamp}_{model.__class__.__name__}{run_name_suffix}"

        wandb_logger = WandbLogger(entity="wods", project="wods")
        wandb_logger.experiment.name = run_name
        wandb_logger.log_hyperparams({
            "model_name": model.__class__.__name__,
            "batch_size": train_loader.batch_size,
            "epochs": epochs,
            "learning_rate": '1e-6 -> 1e-4',
            "cross_validation": bool(run_name_suffix),
            "k_folds": run_name_suffix.lstrip("_fold") if run_name_suffix else None
        })

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


def train_test_model(model_class, train_loader, val_loader, test_loader, epochs=30):
    if isinstance(train_loader, list) and isinstance(val_loader, list):
        assert len(train_loader) == len(val_loader), "Mismatch in folds between train and val loaders"

        for fold, (train, val) in enumerate(zip(train_loader, val_loader)):
            print(f"\n===== Fold {fold + 1}/{len(train_loader)} =====\n")
            run_single_training(model_class, train, val, test_loader, epochs, run_name_suffix=f"_fold{fold + 1}")

    else:
        run_single_training(model_class, train_loader, val_loader, test_loader, epochs)
