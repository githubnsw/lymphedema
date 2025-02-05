import os
import torch

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from lym_model import LymModel
from util import SimpleLymDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import albumentations as albu
from pprint import pprint

# -------------------------------
# Configuration
# -------------------------------
CONFIG = {
    "model_name": "unetplusplus",
    "encoder": "efficientnet-b7",
    "in_channels": 3,
    "out_classes": 3,
    "batch_size": 8,
    "num_workers": 0,
    "max_epochs": 200,
    "patience": 5,
    "save_model_path": "./trained_model.pt",
    "data_dir": "data_path",
    "gpus": 1 if torch.cuda.is_available() else 0,
}

# -------------------------------
# Data Augmentation
# -------------------------------
def get_training_augmentation():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.OneOf([albu.CLAHE(p=1), albu.RandomBrightness(p=1), albu.RandomGamma(p=1)], p=0.9),
        albu.OneOf([albu.RandomContrast(p=1), albu.HueSaturationValue(p=1)], p=0.9),
    ])

def get_validation_augmentation():
    return albu.Compose([albu.PadIfNeeded(512, 512)])

# -------------------------------
# Dataset Preparation
# -------------------------------
CLASSES = ["background", "muscle", "fat"]

DATA_DIR = CONFIG["data_dir"]
x_train_dir, y_train_dir = os.path.join(DATA_DIR, 'train'), os.path.join(DATA_DIR, 'trainannot')
x_valid_dir, y_valid_dir = os.path.join(DATA_DIR, 'val'), os.path.join(DATA_DIR, 'valannot')

train_dataset = SimpleLymDataset(x_train_dir, y_train_dir, CLASSES, augmentation=get_training_augmentation())
valid_dataset = SimpleLymDataset(x_valid_dir, y_valid_dir, CLASSES, augmentation=get_validation_augmentation())

train_dataloader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
valid_dataloader = DataLoader(valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

# -------------------------------
# Model Training
# -------------------------------
if __name__ == "__main__":
    model = LymModel(CONFIG["model_name"], CONFIG["encoder"], in_channels=CONFIG["in_channels"], out_classes=CONFIG["out_classes"])

    early_stop_callback = EarlyStopping(monitor="valid_dataset_iou", patience=CONFIG["patience"], min_delta=0.00, verbose=False, mode="max")

    trainer = pl.Trainer(
        gpus=CONFIG["gpus"],
        max_epochs=CONFIG["max_epochs"],
        callbacks=[early_stop_callback],
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # Save model
    torch.save(model.state_dict(), CONFIG["save_model_path"])
    print(f"Model saved at: {CONFIG['save_model_path']}")

    # Validate model
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)
