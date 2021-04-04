import albumentations as A
import os
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook as tqdm
from glob import glob
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import albumentations as A
output_notebook()
def apply_transforms(transform, df, n_transforms=3):
    """
    transformを適応して表示
    transform = A.Compose([A.Resize(height=256, width=256, p=1)])
    """
    idx = np.random.randint(len(df), size=1)[0]
    image_id = df.iloc[idx].image_id
    
    image = Image.open(TRAIN_DIR + image_id)
    fig, axs = plt.subplots(1, n_transforms+1, figsize=(15,7))
    image = np.array(image)
    # 元画像を出力
    axs[0].imshow(image)
    axs[0].set_title('original')
    
    # 画像変換適応
    for i in range(n_transforms):
        image_aug = transform(image=image)['image']
        print(image_aug.shape)
        axs[i+1].imshow(image_aug)

transform = A.Compose([
#    A.Resize(height=256, width=256, p=1) #リサイズ
#    A.Rotate(limit=45, p=1) #-45~45 回転,タプル指定やはみ出た部分補間形式も指定可能
#    A.ShiftScaleRotate(p=1) #ランダムにAffine変換(せん断、スケール、回転)
#    A.RGBShift(p=1) #RGB値をランダムにシフト
#     A.RandomCrop(width=256, height=256),
#     A.HorizontalFlip(p=0.5), # 輝度とコントラストをランダムに変える
#     A.RandomBrightnessContrast(p=0.2),
#    A.RandomBrightness(p=1) #画像の輝度を変化させる
#    A.ChannelDropout(channel_drop_range=(1,1,), p=1) #チャネルをランダムに落とす
#    A.RandomContrast(p=1.0) #
#    A.RandomCrop(height=100,width=100,p=1) #指定したサイズでランダムにクロップ 舌との違いは？
#    A.RandomResizedCrop(height=256, width=256, p=1) #指定したサイズでランダムな領域をクロップ
#    A.RandomSizedCrop(min_max_height=(256,256), height=100, width=100) #min_max_hでクロップしたあと、 height,widthにリサイズ
#    A.RandomCropNearBBox(p=1) #バウンディングボックスのクロップ
#    A.RandomSizedBBoxSafeCrop(height=, width=) #
#    A.RandomFog(p=1) #霧状のぼかしをかける
#    A.RandomRain(p=1) #湿ったような効果をかけ
#    A.RandomSnow(p=1) #雪
#    A.RandomSunFlare(p=1) #ランダム白飛び領域
#    A.RandomShadow(p=1) #ランダムな領域に影のような効果
#    A.RandomGamma(gamma_limit=(80,120), p=1) # トーンカーブをガンマ値でそうさ
#    A.RandomGridShuffle(grid=(3,3), p=1) # 画像をgridの個数に分割してシャッフル
#    A.RandomRotate90(p=1) #ランダムに90ど回転 回転で十分
#    A.RandomScale(p=1) #画像をランダムにリサイズ 必要ない
#        A.ChannelShuffle(p=1)
#    A.CLAHE(p=1), #ヒストグラム平坦化
#    A.CoarseDropout(max_height=16, max_width=16, p=1) #ランダムにピクセル上のドロップアウト発生
#    A.ColorJitter() #画像の輝度、彩度、コントラストをランダムに変える
#    A.Crop(x_max=800, y_max=200, p=1) 画像をクロップ。使えない。
#    A.Cutout(p=1) #CoarseDropout を正方形で行う。
#    A.Downscale(p=1) #ダウンサンプル。 画像は補間される
#    A.ElasticTransform(p=1) ???
#    A.Equalize() #ヒストグラム平坦化。CHAHEとはちがう方法
#    A.FancyPCA(p=1) ???
#    A.Flip(p=1) #水平垂直 ランダムな方向のフリップ
#    A.VerticalFlip(p=1)
#    A.GaussianBlur(p=1) ガウシアンぼかし
#    A.GaussNoise(var_limit=(10, 40), p=1) #ガウシアンノイズを加えて画素値を変化させる
#    A.GridDropout(p=1) #グリッド状のドロップアウト
#    A.HueSaturationValue(p=1) #hsvの値をランダムに変える
#    A.InvertImg(p=1) #各画素値 v を 255-v に変換
#    A.ISONoise(p=1) #ISOノイズを加える
#    A.Lambda() #カスタム処理を加えられる
#    A.LongestMaxSize(p=1) #画像のアスペクト比を維持しながら最大長に合わせたサイズにする
#    A.MedianBlur(blur_limit=11, p=1) #ぼかし
#    A.MotionBlur(p=1)
#    A.Normalize() #標準化 順番は？？
#    A.OpticalDistortion() #?
#    A.Posterize(p=1)?
#    A.Solarize(threshold=128) #ソラリゼーション
#    A.ToFloat() 
#    A.ToGray(p=1) #グレイスケール化
#    A.ToSepia(p=1) #セピア
#    A.Transpose(p=1) # row, column を入れ替える(転置)
    
    # ソースイメージへターゲットイメージの雰囲気をコピーする系
#    A.augmentations.FDA() #StyleTransfer用 input画像の低周波領域をターゲットイメージのものに置き換え
#    A.augmentations.HistogramMatching() #ヒストグラムの形状を移す
])