# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from PIL import Image
from matplotlib import pyplot as plt

def torch_seed_everything(seed_value=777):
    """pytorch用ランダムシード初期化"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

"""# CONFIG"""

TRAIN_IMAGE_DIR = '/content/train_images'
TRAIN_CSV='/content/train_images.csv'
TEST_IMAGE_DIR = '/content/test_images'
TEST_CSV='/content/test_images.csv'
OUTPUT_DIR = './output'

class Config:
    DEBUG=False
    epochs=40
    fold_num=3
    target_col='class_num'
    seed=17
    model_arch = 'tf_efficientnet_b5_ns'
    img_size=456
    train_bs=12
    valid_bs=16
    T_0=10
    lr=1e-4
    min_lr=1e-7
    weight_decay=1e-6
    num_workers=8
    
    # suppoprt to do batch accumulation for backprop with effectively larger batch size
    # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
    accum_iter = 2 
    
    verbose_step=8
    device='cuda:0'
    num_class=4
    use_folds=[0, 1, 2]
    
config=Config()
torch_seed_everything(config.seed)

"""# Utils

"""

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(config.seed)

"""# Preprocessing"""

from albumentations.core.transforms_interface import ImageOnlyTransform

def resize_square(img):
    """長辺のサイズで正方形の画像に"""
    l=max(img.shape[:2])
    
    h,w = img.shape[:2]
    hm = (l-h)//2
    wm = (l-w)//2
    return cv2.copyMakeBorder(img,
                            hm,
                            hm+(l-h)%2,
                            wm,
                            wm+(l-w)%2,
                            cv2.BORDER_CONSTANT,
                            value=0)


class CropLemon(ImageOnlyTransform):
    """レモンが写っている部分をcrop"""

    def __init__(self, margin=10, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.margin = margin

    def get_box(self, img):
        """ 中央に近い黄色い領域を見つける """
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

        # h,v のしきい値で crop
        _, img_hcrop = cv2.threshold(h, 0, 40, cv2.THRESH_BINARY)
        _, img_vcrop = cv2.threshold(v, v.mean(), 255, cv2.THRESH_BINARY)  #平均より明るければ1, そうでなければ0
        # 平均より明るい かつ 40以上なら白色
        th_img = (img_hcrop * (img_vcrop / 255)).astype(np.uint8)

        contours, hierarchy = \
            cv2.findContours(th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # サイズの大きいものだけ選択
        contours = [c for c in contours if cv2.contourArea(c) > 10000]
        if not contours: return None

        # 中央に近いものを選択
        center = np.array([img.shape[1] / 2, img.shape[0] / 2])  # w, h
        min_contour = None
        min_dist = 1e10

        for c in contours:
            tmp = np.array(c).reshape(-1, 2)
            m = tmp.mean(axis=0)
            dist = sum((center - m) ** 2)
            if dist < min_dist:
                min_contour = tmp
                min_dist = dist

        box = [
            *(min_contour.min(axis=0) - self.margin).astype(np.int).tolist(),
            *(min_contour.max(axis=0) + self.margin).astype(np.int).tolist()]
        for i in range(4):
            if box[i] < 0: box[i] = 0
            if i % 2 == 0:
                if box[i] > img.shape[1]: box[i] = img.shape[1]
            else:
                if box[i] > img.shape[0]: box[i] = img.shape[0]

        return box  # left, top, right, bottom

    def apply(self, image, **params):
        image = image.copy()
        box = self.get_box(image)
        crop_img = None
        if not box or (box[3] - box[1] < 50 or box[2] - box[0] < 50):
            pass
        else:
            try:
                crop_img = image[box[1]:box[3], box[0]:box[2]]
            except:
                pass
        if crop_img is None:
            crop_img = image[40:, 10:-20]
        return resize_square(crop_img)

    def get_transform_init_args_names(self):
        return ("margin",)

img = cv2.imread(f'{TRAIN_IMAGE_DIR}/train_0001.jpg')
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
crop_img = CropLemon()(image=img)['image']
plt.imshow(crop_img)

"""# Dataset, Transforms"""

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
def get_img(path):
    im_bgr = cv2.imread(path)
    return cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)


class LemonDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root

        self.output_label = output_label

        if output_label:
            self.labels = self.df['class_num'].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels
        if self.output_label:
            target = self.labels[index]

        img_path = os.path.join(self.data_root, self.df.loc[index]['id'])
        img = get_img(img_path)

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.output_label:
            return img, target
        else:
            return img

def get_train_transforms(config, force_light=False):
    aug_list=[
        CropLemon(p=1),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, val_shift_limit=5, p=0.3),
        A.Resize(config.img_size, config.img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0),
        ToTensorV2(p=1.0),
    ]
    return A.Compose(aug_list, p=1.0)


def get_valid_transforms(config):
    aug_list=[
        CropLemon(p=1),
        A.Resize(config.img_size, config.img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0),
        ToTensorV2(p=1.0),
    ]
    return A.Compose(aug_list, p=1.0)


        
def prepare_dataloader(df, trn_idx, val_idx,
                       data_root=TRAIN_IMAGE_DIR):
    train_ = df.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df.loc[val_idx, :].reset_index(drop=True)

    train_ds = LemonDataset(train_, data_root, transforms=get_train_transforms(config),
                              output_label=True)
    valid_ds = LemonDataset(valid_, data_root, transforms=get_valid_transforms(config),
                              output_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.train_bs,
        drop_last=True,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=config.valid_bs,
        num_workers=config.num_workers,
        shuffle=False,
    )
    return train_loader, val_loader

"""# Model"""

import timm
import torch
from torch import nn

class LemonModel(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

        self.softmax = nn.Softmax(dim=1)
        self.label_vals = torch.arange(n_class)

    #  やりたいのは期待値返したいだけ
    def forward(self, x):
        x = self.model(x)
        return (self.softmax(x) * self.label_vals).sum(axis=1)

    def to(self, device, *args, **kwargs):
        self.label_vals = self.label_vals.to(device)
        return super().to(device, *args, **kwargs)

def fetch_loss_fn(device):
    return nn.MSELoss().to(device)

"""# train"""

from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(epoch, model, loss_fn, optimizer, scaler,
                    train_loader, device, scheduler=None, schd_batch_update=False):
    """
    Args:
        schd_batch_update: 
            バッチごとに scheduler.step() するか？
            False の場合は 1 epoch の終わりで step
    """
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    running_loss = 0.0
    data_cnt = 0
    for step, (imgs, image_labels) in pbar:

        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).float()
        #print(image_labels)
        data_cnt += imgs.shape[0]

        with autocast():
            outputs = model(imgs)
            
            # TODO: Gradient accumulation ちゃんとわかってない
            # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
            # 概要: https://qiita.com/cfiken/items/1de519e741cbbc09818c#gradient-accumulation-%E3%81%A8%E3%81%AF

            loss = loss_fn(outputs, image_labels)
            running_loss+=loss
            
            loss = loss / config.accum_iter
            scaler.scale(loss).backward()
            
            
            if ((step + 1) % config.accum_iter == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % config.verbose_step == 0) or ((step + 1) == len(train_loader)):
                description = f'train epoch {epoch} loss: {running_loss / data_cnt:.4f}'

                pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:
        scheduler.step()
    running_loss = running_loss/data_cnt
    print('train loss = {:.6f}'.format(running_loss))
    return running_loss


def valid_one_epoch(epoch, model, loss_fn, val_loader, device):
    model.eval()

    tot_loss = 0.0
    data_cnt = 0

    preds = []
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).float()
        data_cnt += imgs.shape[0]

        outputs = model(imgs)
        loss = loss_fn(outputs, image_labels)
        tot_loss += loss
        
        preds.append(outputs.detach().cpu().numpy())

        if ((step + 1) % config.verbose_step == 0) or ((step + 1) == len(val_loader)):
            description = f'val epoch {epoch} loss: {tot_loss / data_cnt:.4f}'
            pbar.set_description(description)
    
    y_pred = np.concatenate(preds)
    # y_pred=np.clip(y_pred.round(),0, config.num_class-1).astype(np.int)

    test_loss = tot_loss / data_cnt
    print('val loss = {:.4f}'.format(test_loss))
    return test_loss, y_pred

"""## train loop"""

from sklearn.metrics import cohen_kappa_score
def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def get_result(res_df):
    y_pred=res_df['preds'].values
    y_true=res_df[config.target_col].values
    return qwk(y_true,y_pred)

device = torch.device(config.device)

def get_dataset_with_folds():
    train_df = pd.read_csv(TRAIN_CSV)
    skf=StratifiedKFold(n_splits=config.fold_num,
                            shuffle=True, random_state=config.seed
                            )
    folds = skf.split(np.arange(train_df.shape[0]), train_df[config.target_col].values)
    return train_df, folds


def run_train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df, folds = get_dataset_with_folds()
    oof_df = pd.DataFrame()

    for fold, (trn_idx, val_idx) in enumerate(folds):
        if config.DEBUG:
            trn_idx = trn_idx[:len(trn_idx) // 20]
            val_idx = val_idx[:len(val_idx) // 10]
        
        valid_folds=train_df.loc[val_idx].reset_index(drop=True)
            
        if fold not in config.use_folds:
            continue

        print('Training with {} started'.format(fold))


        train_loader, val_loader = prepare_dataloader(
            train_df, trn_idx, val_idx, data_root=TRAIN_IMAGE_DIR)

        model = LemonModel(config.model_arch, config.num_class, pretrained=True).to(device)
        scaler = GradScaler()
        optimizer=torch.optim.Adam(model.parameters(), 
                         lr=config.lr, 
                         weight_decay=config.weight_decay, 
                         amsgrad=False)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=config.T_0, 
            T_mult=1,
            eta_min=config.min_lr, 
            last_epoch=-1)
        
        loss_fn = fetch_loss_fn(device)
        best_loss = 1e10
        for epoch in range(config.epochs):
            train_one_epoch(epoch, model, loss_fn, optimizer,
                            scaler, train_loader, device, scheduler=scheduler,
                            schd_batch_update=False)

            with torch.no_grad():
                val_loss, preds = valid_one_epoch(epoch, model, loss_fn, val_loader, device)

            # TODO:max(qwk) でとるほうが良い?
            if best_loss > val_loss:
                best_loss = val_loss
                best_path =  os.path.join(OUTPUT_DIR,f'fold_{fold}_best.bin')
                torch.save({'model':model.state_dict(), 'preds': preds},
                           best_path)
        
        best_path =  os.path.join(OUTPUT_DIR,f'fold_{fold}_best.bin')
        preds = torch.load(best_path)['preds']
        preds=np.clip(preds.round(),0, config.num_class-1).astype(np.int)
        valid_folds['preds']=preds
        
        print(f'----- fold: {fold} result ------')
        print(f'qwk score: {get_result(valid_folds)}')
        
        oof_df = pd.concat([oof_df, valid_folds])
        
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()

    
    print('----- cv ------')
    print(f'qwk score: {get_result(oof_df)}')

"""## inference"""

def inference(model, data_loader, device):
    model.eval()
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    pbar.set_description("inference")
    
    preds = []
    for step, imgs in pbar:
        imgs = imgs.to(device).float()
        outputs = model(imgs).detach().cpu().numpy()
        preds.append(outputs)

    y_pred = np.concatenate(preds)
    return np.clip(y_pred.round(),0,3).astype(np.int)

def run_inference():
        model = LemonModel(config.model_arch,config.num_class).to(device)
        test_ds = LemonDataset(pd.read_csv(TEST_CSV), 
                               TEST_IMAGE_DIR, 
                               transforms=get_valid_transforms(config),
                               output_label=False)

        data_loader = torch.utils.data.DataLoader(test_ds,
                                                  batch_size=config.valid_bs,
                                                  num_workers=config.num_workers,
                                                  shuffle=False,)
        v_pred_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for fold in config.use_folds:
            model_path = os.path.join(OUTPUT_DIR,f'fold_{fold}_best.bin')
            model.load_state_dict(torch.load(model_path)['model'])
            with torch.no_grad():
                pred = inference(model, data_loader, device)

            test_df = pd.concat([test_df, pd.DataFrame(pred)], axis=1)

        # 最頻値とる
        test_pred = test_df.mode(axis=1).loc[:, 0]
        sub_df = pd.read_csv(TEST_CSV)
        sub_df['num_class'] = test_pred.astype(np.int)
        sub_df.to_csv('submission.csv', header=False, index=False)
        print(sub_df.head())

def main():
    seed_everything(config.seed)
    run_train()
    run_inference()

main()

