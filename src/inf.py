import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import random
from contextlib import contextmanager
import functools
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from collections import defaultdict, Counter
import sys
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
# from pytorch_lightning.metrics.functional.classification import auroc
import cv2
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split, StratifiedKFold
import albumentations as A
from sklearn.metrics import roc_auc_score, accuracy_score
from pytorch_lightning.core.lightning import LightningModule

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, CLAHE, MultiplicativeNoise, IAASharpen
)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import matplotlib.pyplot as plt
from pathlib import Path
import timm
from google.cloud import storage
from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import pytorch_lightning as pl
from torch import optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

# os.chdir("/home/jupyter/src")
TRAIN_PATH = '../data/train_p'
rand = 98677

# model_path = f"../outputs/{rand}/fold-0.ckpt"
TEST_PATH = "../data/eval_p"
train = pd.read_csv('../data/Training_Set/RFMiD_Training_Labels.csv')
test = train.iloc[:640, :]
# test = pd.read_csv('../data/sample_submission.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = roc_auc_score(y_true[:, i].astype(int), y_pred[:, i])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score, scores


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ====================================================
# Transforms
# ====================================================
def get_transforms(img_size, data):
    if data == 'train':
        return Compose([
            Resize(img_size, img_size),
            # RandomResizedCrop(img_size, img_size, scale=(0.85, 1.0)),
            # CLAHE(clip_limit=(1, 4), p=1),
            # MultiplicativeNoise(p=1),
            # IAASharpen(p=1),
            HorizontalFlip(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            IAAAdditiveGaussianNoise(),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(img_size, img_size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df, target_cols, transform=None, inference=True):
        self.df = df
        self.file_names = df['ID'].values
        target_cols = df.drop(columns=["ID", "fold"]).columns if target_cols == "all" else target_cols
        self.labels = df[target_cols].values
        self.transform = transform
        self.inference = inference

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TRAIN_PATH}/{file_name}.png'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = circle_crop(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = torch.tensor(self.labels[idx]).float()
        if self.inference:
            return image
        else:
            return image, label


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['ID'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TEST_PATH}/{file_name}.png'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = circle_crop(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image


class RANZCRDataModule(LightningDataModule):
    def __init__(
            self,
            cfg,
            train_df,
            val_df,
            aug_p: float = 0.5,
            val_pct: float = 0.2,
            img_sz: int = 224,
            batch_size: int = 64,
            num_workers: int = 4,
            fold_id: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.aug_p = aug_p
        self.val_pct = val_pct
        self.img_sz = img_sz
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold_id = fold_id

        self.train_df = train_df
        self.val_df = val_df

    def train_dataloader(self):
        train_dataset = TrainDataset(self.train_df, self.cfg.base.target_cols,
                                     transform=get_transforms(self.img_sz, data="train"))

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        valid_dataset = TrainDataset(self.val_df, self.cfg.base.target_cols,
                                     transform=get_transforms(self.img_sz, data="valid"))
        return DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class RANZCRModel(LightningModule):
    def __init__(self, cfg, model_name="resnext50_32x4d", pretrained=False):
        super().__init__()

        self.cfg = cfg
        self.wd = 1e-6
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained)

        if "efficient" not in self.model_name:
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, cfg.base.target_size)
        else:
            "efficient"
            self.model.classifier = nn.Linear(self.model.num_features, cfg.base.target_size)

        self.optimizer = Adam(self.model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay,
                              amsgrad=False)
        self.model.avg_pool = GeM()

        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = CB_loss

        self.scheduler = self.get_scheduler()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # self.log("valid_loss", loss, prog_bar=True)
        return loss, y_hat.flatten(), y.flatten()

    def validation_epoch_end(self, input_):
        loss_list, y_hat_list, y_list = np.array([]), np.array([]), np.array([])
        for i, (loss, y_hat, y) in enumerate(input_):
            loss_list = np.append(loss_list, float(loss.cpu()))
            y_hat_list = np.append(y_hat_list, y_hat.cpu().numpy().flatten())
            y_list = np.append(y_list, y.cpu().numpy().flatten())

        auc = roc_auc_score(y_list, y_hat_list)
        acc = accuracy_score(y_list, np.round(y_hat_list))
        self.log("valid_loss", loss_list.mean(), prog_bar=True)
        self.log("valid auc", auc, prog_bar=True)
        self.log("valid Acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.cfg.model.lr, weight_decay=self.wd
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, 0
        )

        return [optimizer], [scheduler]

    def get_scheduler(self):
        if self.cfg.model.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.cfg.model.factor,
                                          patience=self.cfg.model.patience,
                                          verbose=True,
                                          eps=self.cfg.model.eps)
        elif self.cfg.model.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.model.T_max, eta_min=self.cfg.model.min_lr,
                                          last_epoch=-1)
        elif self.cfg.model.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=self.cfg.model.T_0, T_mult=1,
                                                    eta_min=self.cfg.model.min_lr,
                                                    last_epoch=-1)
        return scheduler


def test_inf(dataset, model, model_path):
    model = model.load_from_checkpoint(model_path, cfg=cfg, model_name=cfg.model.model_name).to(device)
    model.freeze()
    model.eval()
    test_loader = DataLoader(
        dataset,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.base.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    for i, img in tqdm(enumerate(test_loader)):
        y_hat = model(img.to(device))
        if i == 0:
            pred = y_hat.cpu().numpy()
        else:
            pred = np.append(pred, y_hat.cpu().numpy(), axis=0)

    pred = sigmoid(pred)

    return pred


def train_loop(cfg, folds, fold):
    model_path = f"../outputs/{rand}/fold-{fold}.ckpt"
    val_idx = folds[folds['fold'] == fold].index
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    model = RANZCRModel(
        cfg,
        model_name=cfg.model.model_name,
    )

    pr_model = model.load_from_checkpoint(model_path, cfg=cfg, model_name=cfg.model.model_name).to(device)
    pr_model.freeze()
    pr_model.eval()

    test_set = TestDataset(
        test,
        get_transforms(
            cfg.model.size,
            "valid"
        )
    )
    pred = test_inf(test_set, model, model_path)
    oof = None

    if cfg.base.oof:
        val_set = TrainDataset(
            valid_folds,
            cfg.base.target_cols,
            get_transforms(
                cfg.model.size,
                "valid"
            ),
            inference=True
        )
        oof = test_inf(val_set, model, model_path)
    return pred, oof


def main(cfg):
    seed_torch(seed=cfg.base.seed)

    folds = train.copy()
    Fold = KFold(n_splits=cfg.base.n_fold, shuffle=True, random_state=cfg.base.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds)):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)

    oof_df = train.copy()
    test_pred = test.copy()
    test_pred.iloc[:, 1:] = 0

    for fold in range(cfg.base.n_fold):
        if fold in cfg.base.trn_fold:
            fold_pred, fold_oof = train_loop(cfg, folds, fold)
            if cfg.base.oof:
                oof_df.iloc[folds["fold"] == fold, 1:] = fold_oof

            test_pred.iloc[:, 1:] += fold_pred / len(cfg.base.trn_fold)

    test_pred.to_csv(f"../outputs/{rand}/{rand}_{cfg.base.n_fold}_{len(cfg.base.trn_fold)}_inf_only3.csv", index=False)
    oof_df.to_csv(f"../outputs/{rand}/{rand}_oof_inf.csv", index=False)


cfg = OmegaConf.load("../yaml/1.yaml")
main(cfg)
