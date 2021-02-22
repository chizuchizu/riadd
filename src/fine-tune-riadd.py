import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRFinetuneTransform
# from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner

import random

from contextlib import contextmanager
# from adabelief_pytorch import AdaBelief
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
from sklearn.metrics import roc_auc_score, accuracy_score
from pytorch_lightning.core.lightning import LightningModule
from src.loss import BCEFocalLoss
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, CLAHE, MultiplicativeNoise, IAASharpen
)
from torch.autograd import Variable
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import matplotlib.pyplot as plt
from pathlib import Path
import timm
from google.cloud import storage
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
# from torch.optim import Adam, SGD
from typing import List, Optional, Any
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pl_bolts.models.self_supervised import SSLEvaluator

from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pytorch_lightning.loggers import WandbLogger
import wandb
from PIL import Image
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

train = pd.read_csv("../data/Training_Set/RFMiD_Training_Labels.csv")  # .iloc[:, 2:]
rand = random.randint(0, 100000)
test = train.iloc[:640, :]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

conf = """
base:
  train_path: '../data/train_p_1'
  test_path: "../data/eval_p_1"
  print_freq: 100
  num_workers: 4
  seed: 42
  # target_size: 2
  # target_cols: [
  #     "TV"
  # ]
  target_size: 1
  target_cols: [
     "TV"
  ]
  n_fold: 4
  trn_fold: [0]
  train: True
  debug: False
  oof: False

split:
  # name: "MultilabelStratifiedKFold"
  name: "StratifiedKFold"
  param: {
           "n_splits": 4,
           "shuffle": True,
           "random_state": 1212
  }

model:
  model_name: "tf_efficientnet_b0_ns"
  size: 224  # 480
  batch_size: 128
  pretrained: true
  epochs: 10
  in_features: 2048

loss:
  name: "BCEFocalLoss"
  param: {}

optimizer:
  name: "AdamW"
  param: {
           "lr": 5e-2,
           "weight_decay": 1e-6,
           "amsgrad": False
  }

scheduler:
  name: "CosineAnnealingLR"
  param: {
            "T_max": 10,
            "eta_min": 0,
            "last_epoch": -1
  }
wandb:
  use: true
  project: "fine-tuning-1"
  name: "TV"
  tags: [
          # "tf_efficientnet_b0_ns",
          "resnet50",
          "MYA"
          # "focal_loss"
          # "aug_7"
  ]
"""


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, cfg, df, transform=None, inference=False):
        self.df = df
        self.cfg = cfg
        self.file_names = df['ID'].values
        # target_cols = # df.drop(columns=["ID", "fold"]).columns if target_cols == "all" else target_cols
        self.labels = df[cfg.base.target_cols].values
        self.transform = transform
        self.inference = inference

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{self.cfg.base.train_path}/{file_name}.png'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(np.uint8(image)).convert("RGB")
        if self.transform:
            # print(image.shape)
            # image = image.transpose(2, 0, 1)
            image = self.transform(image=image)["image"]
            # print(image)
            # print(augmented)
            # image = image['image']
        label = torch.tensor(self.labels[idx]).long()
        # label = int(self.labels[idx])

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
        # image = Image.fromarray(np.uint8(image)).convert("RGB")
        if self.transform:
            image = self.transform(image=image)["image"]
            # image = augmented['image']
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
        train_dataset = TrainDataset(self.cfg, self.train_df, transform=get_transforms(self.img_sz, mode="train"))
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        valid_dataset = TrainDataset(self.cfg, self.val_df, transform=get_transforms(self.img_sz, mode="valid"))
        return DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )


__CRITERIONS__ = {
    "BCEFocalLoss": BCEFocalLoss
}

__SPLITS__ = {
    "MultilabelStratifiedKFold": MultilabelStratifiedKFold,
    "KFold": KFold,
}

__OPTIMIZERS__ = {
    # "AdaBelief": AdaBelief,
    # "RAdam": torch_optimizer.RAdam
}


def get_criterion(cfg):
    if hasattr(nn, cfg.loss.name):
        return nn.__getattribute__(cfg.loss.name)(**cfg.loss.param)
    elif __CRITERIONS__.get(cfg.loss.name) is not None:
        return __CRITERIONS__[cfg.loss.name](**cfg.loss.param)
    else:
        raise NotImplementedError


def get_optimizer(cfg, model):
    optimizer_name = cfg.optimizer.name

    if __OPTIMIZERS__.get(optimizer_name) is not None:
        return __OPTIMIZERS__[optimizer_name](model.parameters(), **cfg.optimizer.param)
    else:
        return optim.__getattribute__(optimizer_name)(model.parameters(), **cfg.optimizer.param)


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


def main(cfg):
    global train
    seed_torch(seed=cfg.base.seed)

    folds = train.copy()
    if cfg.base.debug:
        folds = folds.sample(n=1000, random_state=cfg.base.seed).reset_index(drop=True)
        cfg.model.epochs = 1
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
                oof_df.iloc[folds["fold"] == fold, 1:cfg.base.target_size] = fold_oof

            test_pred[list(cfg.base.target_cols)] += fold_pred / len(cfg.base.trn_fold)

    test_pred[["ID"] + list(cfg.base.target_cols)].to_csv(
        f"../ft/{rand}_{cfg.base.n_fold}_{len(cfg.base.trn_fold)}.csv", index=False)


#
# main(OmegaConf.create(conf))
def get_transforms(img_size: int, mode="train"):
    if mode == "train":
        return A.Compose([
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=(0.9, 1.1),
                ratio=(0.9, 1.1),
                p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=180,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=5,
                val_shift_limit=5,
                p=0.5),
            A.CoarseDropout(max_holes=10, max_height=40, max_width=40, p=0.5),
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.4406],
                std=[0.229, 0.224, 0.225],
                always_apply=True),
            ToTensorV2()
        ])
    elif mode == "valid":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.4406],
                std=[0.229, 0.224, 0.225],
                always_apply=True),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.4406],
                std=[0.229, 0.224, 0.225],
                always_apply=True),
            ToTensorV2()
        ])


# def get_transforms(img_size, data):
#     if data == 'train':
#         return SimCLRFinetuneTransform(input_height=224, eval_transform=False)
#
#     elif data == 'valid':
#         return SimCLRFinetuneTransform(input_height=224, eval_transform=True)
#

class SSLEvaluator(nn.Module):

    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(Flatten(), nn.Dropout(p=p), nn.Linear(n_input, n_classes, bias=True))
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )

    def forward(self, x):
        logits = self.block_forward(x)
        # logits = torch.sigmoid(logits)
        return logits


def test_inf(cfg, dataset, model, model_path, backbone):
    model = model.load_from_checkpoint(model_path, backbone=backbone, cfg=cfg, model_name=cfg.model.model_name).to(
        device)
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


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class SSLFineTuner(LightningModule):
    """
    Finetunes a self-supervised learning backbone using the standard evaluation protocol of a singler layer MLP
    with 1024 units
    Example::
        from pl_bolts.utils.self_supervised import SSLFineTuner
        from pl_bolts.models.self_supervised import CPCV2
        from pl_bolts.datamodules import CIFAR10DataModule
        from pl_bolts.models.self_supervised.cpc.transforms import CPCEvalTransformsCIFAR10,
                                                                    CPCTrainTransformsCIFAR10
        # pretrained model
        backbone = CPCV2.load_from_checkpoint(PATH, strict=False)
        # dataset + transforms
        dm = CIFAR10DataModule(data_dir='.')
        dm.train_transforms = CPCTrainTransformsCIFAR10()
        dm.val_transforms = CPCEvalTransformsCIFAR10()
        # finetuner
        finetuner = SSLFineTuner(backbone, in_features=backbone.z_dim, num_classes=backbone.num_classes)
        # train
        trainer = pl.Trainer()
        trainer.fit(finetuner, dm)
        # test
        trainer.test(datamodule=dm)
    """

    def __init__(
            self,
            backbone: torch.nn.Module,
            cfg: Any,  # OmegaConf
            hidden_dim: Optional[int] = None,
            dropout: float = 0.,
            weight_decay: float = 1e-6,
            nesterov: bool = False,
            final_lr: float = 0.,
    ):
        """
        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
            num_classes: classes of the dataset
            hidden_dim: dim of the MLP (1024 default used in self-supervised literature)
        """
        super().__init__()

        self.cfg = cfg

        self.learning_rate = self.cfg.model.lr
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        self.epochs = self.cfg.model.epochs
        self.final_lr = final_lr

        self.backbone = backbone
        self.linear_layer = SSLEvaluator(n_input=self.cfg.model.in_features, n_classes=self.cfg.base.target_size,
                                         p=dropout,
                                         n_hidden=hidden_dim)

        self.criterion = get_criterion(cfg)

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        # self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        return loss, logits.cpu().numpy(), y.cpu().numpy()

    def validation_epoch_end(self, input_):
        auc_l = 0
        acc_l = 0
        acc_f = 0
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
            auc_l += auc / self.cfg.base.target_size
            acc_l += acc / self.cfg.base.target_size

            num = "{0:02d}".format(j + 1)
            self.log(f"{self.cfg.base.target_cols[j]}-auc", auc, prog_bar=False)
            self.log(f"{self.cfg.base.target_cols[j]}-acc", acc, prog_bar=False)

            if j == 0:
                auc_f = auc

        for i, (loss, y_hat, y) in enumerate(input_):
            loss_list = np.append(loss_list, float(loss.cpu()))
        # self.log("valid_loss", loss_list.mean(), prog_bar=True)
        self.log("valid auc", auc_l, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.test_acc(logits, y)

        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', self.test_acc)

        return loss

    def shared_step(self, batch):
        x, y = batch

        with torch.no_grad():
            feats = self.backbone(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        # logits = torch.sigmoid(logits)
        # loss = F.cross_entropy(logits, y)

        loss = self.criterion(logits.float(), y.float())

        return loss, logits, y

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        return logits

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.cfg,
            self.linear_layer,
        )

        scheduler = get_scheduler(
            self.cfg,
            optimizer
        )

        return [optimizer], [scheduler]


def train_loop(cfg, folds, fold):
    global rand
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

    data_module = CHIZUDataModule(
        cfg,
        train_folds,
        valid_folds,
        aug_p=0.5,
        # img_sz=cfg.model.size,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.base.num_workers,
        # dataset="cifer10",
        # fold_id=fold,
    )

    backbone = SimCLR(
        num_samples=1,
        batch_size=cfg.model.batch_size,
        dataset="imagenet",
        # maxpool1=False,
        # first_conv=False,
        gpus=1
    ).load_from_checkpoint(
        "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt",
        strict=False)  # .load_from_checkpoint("exp2/fold-0.ckpt")
    # backbone = backbone.load_from_checkpoint("exp2/fold-0.ckpt")
    tuner = SSLFineTuner(
        backbone,
        cfg,
        hidden_dim=None,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'exp2/',
        filename=f"fold-{fold}",
        # save_top_k=3,
        mode='min',
    )

    # trainer
    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=cfg.model.epochs,
        gradient_clip_val=0.1,
        precision=16,
        distributed_backend="ddp",
        logger=wandb_logger if "wandb_logger" in locals() else None,
        callbacks=[checkpoint_callback]
    )
    # print(tuner)
    trainer.fit(tuner, data_module)

    model_path = checkpoint_callback.best_model_path

    test_set = TestDataset(
        cfg,
        test,
        get_transforms(
            cfg.model.size,
            "valid"
        )
    )
    pred = test_inf(cfg, test_set, tuner, model_path, backbone)
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
        oof = test_inf(cfg, val_set, tuner, model_path, backbone)
    return pred, oof


main(OmegaConf.create(conf))
