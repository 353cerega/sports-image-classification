import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics import Accuracy


class SportsClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        num_classes: int = 100,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        if not hasattr(models, model_name):
            raise ValueError(f"Модель {model_name} не найдена в torchvision.models")

        weights = "DEFAULT"
        self.backbone = getattr(models, model_name)(weights=weights)

        if hasattr(self.backbone, "fc"):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, "classifier"):
            if isinstance(self.backbone.classifier, nn.Sequential):
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Не удалось определить структуру модели {model_name}")

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
