import sys
import json

import numpy as np

from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from torchmetrics import Accuracy

from torchvision.datasets import MNIST

from albumentations import Compose, RandomResizedCrop
from albumentations.pytorch.transforms import ToTensorV2

# import pytorch_lightning as pl
import lightning as L
import lightning.pytorch as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def get_args():
    """Loads program arguments"""
    assert (
        len(sys.argv) >= 2
    ), "Program needs a path to the configuration file in JSON format."

    fp = open(sys.argv[1], "r")
    args = json.load(fp)
    fp.close()
    print(args)

    return args


def alb_to_transform(aug):
    """Given an albumentation, returns the respective simple transform version"""
    return lambda x: aug(image=np.array(x))["image"].float()


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, root_dir="data", batch_size=32, val_batch_size=32):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.train_transform = alb_to_transform(
            Compose([RandomResizedCrop(28, 28, scale=(0.8, 1)), ToTensorV2()])
        )

        self.val_transform = alb_to_transform(ToTensorV2())

    def prepare_data(self):
        MNIST(self.root_dir, train=True, download=True)
        MNIST(self.root_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_set = MNIST(
                self.root_dir, train=True, download=True, transform=self.train_transform
            )
        
        if stage in ("fit", "validate"):
            self.val_set = MNIST(
                self.root_dir, train=False, download=True, transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None

    def teardown(self):
        pass


class MNISTClassificationTask(pl.LightningModule):
    """Classification task for the MNIST dataset"""

    def __init__(self, model=None, lr=0.001):
        super(MNISTClassificationTask, self).__init__()
        self.save_hyperparameters()

        # Attributes
        self.model = ConvModel() if model is None else model
        self.lr = lr

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optim = Adam(self.model.parameters(), self.lr)
        scheduler = LinearLR(optim)
        return [optim], [
            {
                "scheduler": scheduler,
                "interval": "epoch",  # LR Scheduler to update the learning rate each epoch
            }
        ]

    def _get_preds_loss_acc(self, batch):
        x, y = batch  # No need to worry about devices

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        return y_hat, loss, acc

    def training_step(self, batch):
        _, loss, acc = self._get_preds_loss_acc(batch)

        self.log_dict({"train loss": loss, "train accuracy": acc}, sync_dist=True)

        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_idx):
        pred, loss, acc = self._get_preds_loss_acc(batch)

        self.log_dict(
            {"validation loss": loss, "validation accuracy": acc}, sync_dist=True
        )

        return pred


class ConvModel(pl.LightningModule):
    """Convolutional model to be used for the MNIST classification task"""

    def __init__(
        self, conv1_size=3, conv2_size=3, hidden_channels=10, mlp_hidden=50, **kwargs
    ):
        super(ConvModel, self).__init__()

        assert conv1_size % 2 == 1
        assert conv2_size % 2 == 1

        # Arguments
        if kwargs:
            conv1_size = kwargs["conv1_size"]
            conv2_size = kwargs["conv2_size"]
            hidden_channels = kwargs["hidden_channels"]
            mlp_hidden = kwargs["mlp_hidden"]

        # Hyper-Parameters
        self.conv1_size = conv1_size
        self.conv2_size = conv2_size
        self.hidden_channels = hidden_channels
        self.mlp_hidden = mlp_hidden

        # Non parametric function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.flatten = nn.Flatten()
        self.max_pool = nn.MaxPool2d(2, 2)

        # Parameters
        self.conv1 = nn.Conv2d(1, hidden_channels, conv1_size, 1, conv1_size // 2)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, conv2_size, 1, conv2_size // 2
        )
        self.linear1 = nn.Linear(7 * 7 * hidden_channels, mlp_hidden)
        self.linear2 = nn.Linear(mlp_hidden, 10)

        # Saving hyper-parameters (logged by W&B)
        self.save_hyperparameters()

    def forward(self, x):
        out = self.max_pool(self.relu(self.conv1(x)))
        out = self.max_pool(self.relu(self.conv2(out)))
        out = self.relu(self.linear1(self.flatten(out)))
        out = self.softmax(self.linear2(out))

        return out


def main():
    # Getting program arguments
    args = get_args()

    # Setting reproducibility
    seed = 0 if "seed" not in args.keys() else args["seed"]
    pl.seed_everything(seed)

    # Getting data
    dataset = MNISTDataModule(
        root_dir=args["data"]["root_dir"],
        batch_size=args["optimization"]["batch_size"],
        val_batch_size=args["optimization"]["val_batch_size"],
    )

    # Building model
    model = MNISTClassificationTask(
        model=ConvModel(**args["model"]["params"]),
        optimizer_fn=getattr(torch.optim, args["optimization"]["optimizer_fn"]),
        lr=args["optimization"]["lr"],
        scheduler_fn=getattr(
            torch.optim.lr_scheduler, args["optimization"]["scheduler_fn"]
        ),
    )

    # Training
    logger = WandbLogger(
        project=args["logger"]["project"],
        name=args["logger"]["name"],
        save_dir=args["logger"]["save_dir"],
        log_model="all",
    )
    logger.watch(model)
    trainer = Trainer(
        logger=logger,
        max_epochs=args["optimization"]["epochs"],
        strategy=args["optimization"]["strategy"],
        num_nodes=args["optimization"]["num_nodes"],
        deterministic=True,
        profiler='simple',
        callbacks=[
            ModelCheckpoint(
                dirpath=args["optimization"]["callbacks"]["ModelCheckpoint"]["dirpath"],
                filename=args["optimization"]["callbacks"]["ModelCheckpoint"][
                    "filename"
                ],
            ),
            EarlyStopping(
                monitor=args["optimization"]["callbacks"]["EarlyStopping"]["monitor"],
                patience=args["optimization"]["callbacks"]["EarlyStopping"]["patience"],
            ),
        ],
    )

    trainer.fit(model, datamodule=dataset)


if __name__ == "__main__":
    main()
