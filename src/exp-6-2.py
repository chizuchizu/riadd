import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import random
from contextlib import contextmanager
from adabelief_pytorch import AdaBelief
import functools
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from collections import defaultdict, Counter
import sys
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
# from pytorch_lightning.metrics.functional.classification import auroc
import cv2
from pytorch_lightning import LightningDataModule
from sklearn import model_selection
import albumentations as A
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_score
from pytorch_lightning.core.lightning import LightningModule

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, CLAHE, MultiplicativeNoise, IAASharpen
)
from torch.autograd import Variable
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import matplotlib.pyplot as plt
from pathlib import Path
# import timm
from google.cloud import storage
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
# from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import pytorch_lightning as pl
import torch.optim as optim
from src.optimizer import get_optimizer
from src.loss import get_criterion
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

sys.path.append("../pytorch-image-models")
import timm

train = pd.read_csv('../data/Training_Set/RFMiD_Training_Labels.csv')
extra = pd.read_csv("../extra/use_df.csv").iloc[:0, :]
extra["fold"] = -1
test = train.iloc[:640, :]
# test = pd.read_csv('../data/sample_submission.csv')
rand = random.randint(0, 100000)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def multi_disease_avg_score(y_true: np.ndarray, y_pred: np.ndarray):
    map_score = average_precision_score(y_true=y_true, y_score=y_pred, average=None)
    map_score = np.nan_to_num(map_score, nan=0.0).mean()

    scores = []
    for i in range(len(y_true[0])):
        if y_true[:, i].mean() > 0.0:
            auc = roc_auc_score(y_true=y_true[:, i], y_score=y_pred[:, i])
            scores.append(auc)
        else:
            scores.append(0.0)
    auc_score = np.mean(scores)
    return 0.5 * map_score + 0.5 * auc_score


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
    def __init__(self, cfg, df, transform=None, inference=False):
        self.cfg = cfg
        self.df = df
        self.file_names = df['ID'].values
        # target_cols = # df.drop(columns=["ID", "fold"]).columns if target_cols == "all" else target_cols
        self.labels = df[cfg.base.target_cols].values
        self.transform = transform
        self.inference = inference

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        if "right" in str(file_name) or "left" in str(file_name):
            file_path = f'{self.cfg.base.extra_path}/{file_name}.png'
        else:
            file_path = f'{self.cfg.base.train_path}/{file_name}.png'
        image = cv2.imread(file_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(file_path)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = torch.tensor(self.labels[idx]).float()

        if self.inference:
            return image
        else:
            return image, label


class TestDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df
        self.file_names = df['ID'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{self.cfg.base.test_path}/{file_name}.png'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image


class CHIZUDataModule(LightningDataModule):
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
        train_dataset = TrainDataset(self.cfg, self.train_df,
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
        valid_dataset = TrainDataset(self.cfg, self.val_df,
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


__SPLITS__ = {
    "MultilabelStratifiedKFold": MultilabelStratifiedKFold
}


def get_scheduler(cfg, optimizer):
    scheduler_name = cfg.scheduler.name

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(optimizer, **cfg.scheduler.param)


def get_split(cfg):
    if hasattr(model_selection, cfg.split.name):
        return model_selection.__getattribute__(cfg.split.name)(**cfg.split.param)
    elif __SPLITS__.get(cfg.split.name) is not None:
        return __SPLITS__[cfg.split.name](**cfg.split.param)
    else:
        raise NotImplementedError


class MAP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        targets = torch.sigmoid(targets)
        loss = torch.from_numpy(
            np.array(1 - average_precision_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())))
        return loss


class CHIZUModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.wd = 1e-6

        self.model_name = cfg.model.model_name
        self.model = timm.create_model(cfg.model.model_name, pretrained=cfg.model.pretrained)

        if "nfnet" in self.model_name:
            # n_features = self.model.num_classes
            n_features = self.model.num_features
            self.model.head.fc = nn.Linear(n_features, cfg.base.target_size)
        elif "efficient" not in self.model_name:
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, cfg.base.target_size)
        else:
            self.model.classifier = nn.Linear(self.model.num_features, cfg.base.target_size)

        if self.cfg.model.gem:
            self.model.avg_pool = GeM()

        self.optimizer = get_optimizer(cfg, self.model)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        self.criterion = get_criterion(cfg)
        self.sub_criterion = MAP()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        # x = self.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        loss2 = self.sub_criterion(y_hat, y)
        loss = loss * 0.7 + loss2 * 0.3
        # self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # self.log("valid_loss", loss, prog_bar=True)
        return loss, y_hat.cpu().numpy(), y.cpu().numpy()

    def validation_epoch_end(self, input_):
        auc_l = 0
        acc_l = 0
        acc_f = 0
        # map_l = 0
        for j in range(self.cfg.base.target_size):
            loss_list, y_hat_list, y_list = np.array([]), np.array([]), np.array([])
            for i, (loss, y_hat, y) in enumerate(input_):
                # y_hat_list = np.append(y_hat_list, y_hat.argmax(1))
                y_hat_list = np.append(y_hat_list, y_hat[:, j])
                y_list = np.append(y_list, y[:, j])

            y_hat_list = sigmoid(y_hat_list)
            try:
                auc = roc_auc_score(y_list, y_hat_list)
            except ValueError:
                auc = 0
            acc = accuracy_score(y_list, np.round(y_hat_list))
            # ap = precision_score(y_list, np.round(y_hat_list), average="macro")
            auc_l += auc / self.cfg.base.target_size
            acc_l += acc / self.cfg.base.target_size
            # map_l += ap / self.cfg.base.target_size

            num = "{0:02d}".format(j + 1)
            if j == 0:
                self.log(f"{self.cfg.base.target_cols[j]}-auc", auc, prog_bar=True)
            else:
                self.log(f"{self.cfg.base.target_cols[j]}-auc", auc, prog_bar=False)

        for i, (loss, y_hat, y) in enumerate(input_):
            loss_list = np.append(loss_list, float(loss.cpu()))
        self.log("valid_loss", loss_list.mean(), prog_bar=True)
        self.log("valid auc", auc_l, prog_bar=True)
        # self.log("1st auc", auc_f, prog_bar=True)

        # self.log("score", ((map_l + sub2_auc) / 4) + auc_f / 2, prog_bar=True)
        # self.log("valid Acc", acc_l, prog_bar=True)
        # self.log("auc-1st", auc_f, prog_bar=True)

        all_pred, all_true = np.empty((0, self.cfg.base.target_size), dtype=int), np.empty(
            (0, self.cfg.base.target_size), dtype=int)
        for i, (loss, y_hat, y) in enumerate(input_):
            # y_hat_list = np.append(y_hat, y_hat.argmax(1))
            all_pred = np.append(all_pred, y_hat, axis=0)
            all_true = np.append(all_true, y, axis=0)

        map = average_precision_score(all_true, all_pred, average=None)
        map = np.nan_to_num(map, nan=0.0)
        for j in range(self.cfg.base.target_size):
            self.log(f"{self.cfg.base.target_cols[j]}-map", map, prog_bar=False)

    """
    loss_list, all_pred, all_true = np.empty((0, self.cfg.base.target_size), dtype=int),np.empty((0, self.cfg.base.target_size), dtype=int),np.empty((0, self.cfg.base.target_size), dtype=int)
    for i, (loss, y_hat, y) in enumerate(input_):
        # y_hat_list = np.append(y_hat, y_hat.argmax(1))
        all_pred = np.append(all_pred, y_hat, axis=0)
        all_true = np.append(all_true, y, axis=0)

    """

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler

        return [optimizer], [scheduler]


class SAMModel(CHIZUModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        optimizer.first_step(closure=optimizer_closure, zero_grad=True)
        optimizer.second_step(closure=optimizer_closure, zero_grad=True)


def test_inf(cfg, dataset, model, model_path):
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
    if cfg.wandb.use:
        wandb.init(
            name=cfg.wandb.name + f"-fold-{fold}-{rand}",
            project=cfg.wandb.project,
            tags=cfg.wandb.tags + [str(rand)],
            reinit=True
        )
        wandb_logger = WandbLogger(
            name=cfg.wandb.name + f"-fold-{fold}-{rand}",
            project=cfg.wandb.project,
            tags=cfg.wandb.tags + [str(rand)]
        )
        wandb_logger.log_hyperparams(dict(cfg))
        wandb_logger.log_hyperparams(dict({"rand": rand, "fold": fold, }))

    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_folds = pd.concat(
        [
            train_folds,
            extra
        ],
        # axis=1
    )

    data_module = CHIZUDataModule(
        cfg,
        train_folds,
        valid_folds,
        aug_p=0.5,
        img_sz=cfg.model.size,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.base.num_workers,
        # fold_id=fold,
    )
    if cfg.optimizer.name == "SAM":
        model = SAMModel(
            cfg
        )
    else:
        model = CHIZUModel(
            cfg
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'../exp6/{rand}',
        filename=f"fold-{fold}",
        # save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=cfg.model.epochs,
        gradient_clip_val=0.1,
        precision=16,
        logger=wandb_logger if "wandb_logger" in locals() else None,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model=model, datamodule=data_module)
    model_path = checkpoint_callback.best_model_path

    test_set = TestDataset(
        cfg,
        test,
        get_transforms(
            cfg.model.size,
            "valid"
        )
    )
    pred = test_inf(cfg, test_set, model, model_path)
    oof = None

    if cfg.base.oof:
        val_set = TrainDataset(
            cfg,
            valid_folds,
            get_transforms(
                cfg.model.size,
                "valid"
            ),
            inference=True
        )
        oof = test_inf(cfg, val_set, model, model_path)
    return pred, oof


def main(cfg):
    seed_torch(seed=cfg.base.seed)

    folds = train.copy()
    Fold = get_split(cfg)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[cfg.base.target_cols])):
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

            test_pred[list(cfg.base.target_cols)] += fold_pred / len(cfg.base.trn_fold)

    test_pred[["ID"] + list(cfg.base.target_cols)].to_csv(
        f"../exp6/{rand}/{rand}_{cfg.base.n_fold}_{len(cfg.base.trn_fold)}.csv", index=False)
    # oof_df.to_csv(f"../exp2/{rand}/{rand}_oof.csv", index=False)


if __name__ == "__main__":
    main(OmegaConf.load("../yaml/6-2.yaml"))
