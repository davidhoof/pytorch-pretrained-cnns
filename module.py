import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy
from scheduler import WarmupCosineLR
import numpy as np
import models


class TrainModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.model = models.get_model(self.hparams["classifier"])(in_channels=hparams["in_channels"],
                                                                  num_classes=hparams["num_classes"])
        self.acc_max = 0
        
    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = torch.nn.CrossEntropyLoss()(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        return loss

    def training_epoch_end(self, outs):            
        self.log("loss/train", np.mean([d["loss"].item() for d in outs]))
        self.log("acc/train", self.accuracy.compute() * 100)
        self.accuracy.reset()
    
    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        return loss

    def validation_epoch_end(self, outs):
        self.log("loss/val", np.mean([d.item() for d in outs]))
        
        acc = self.accuracy.compute() * 100
        if acc > self.acc_max:
            self.acc_max = acc
        
        self.log("acc_max/val", self.acc_max)
        self.log("acc/val", acc)

        self.accuracy.reset()

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
            momentum=self.hparams["momentum"],
            nesterov=True
        )

        total_steps = self.hparams["max_epochs"] * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps),
            "interval": "step",
            "name": "learning_rate",
        }

        return [optimizer], [scheduler]
