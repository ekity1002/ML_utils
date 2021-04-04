# 画像EDA用関数など
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

from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel
from bokeh.models.widgets import Tabs

import albumentations as A
output_notebook()

def hist_hover(dataframe, column, colors=["#94c8d8", "#ea5e51"], bins=30, title=''):
    """
    bokeh でhistgram を表示
    # https://towardsdatascience.com/interactive-histograms-with-bokeh-202b522265f3
    ex1:
    hist_hover(train, 'brightness', title='輝度のヒストグラム')

    ex2:
    for title in sorted(train['disease_name'].unique()):
        hist_hover(train[train['disease_name']==title], 'brightness', title=f'{title} brightness histgram')
    """
    hist, edges = np.histogram(dataframe[column], bins = bins)
    
    hist_df = pd.DataFrame({column: hist,
                             "left": edges[:-1],
                             "right": edges[1:]})
    hist_df["interval"] = ["%d to %d" % (left, right) for left, 
                           right in zip(hist_df["left"], hist_df["right"])]

    src = ColumnDataSource(hist_df)
    plot = figure(plot_height = 400, plot_width = 600,
          title = title,
          x_axis_label = column,
          y_axis_label = "Count")    
    plot.quad(bottom = 0, top = column,left = "left", 
        right = "right", source = src, fill_color = colors[0], 
        line_color = "#35838d", fill_alpha = 0.7,
        hover_fill_alpha = 0.7, hover_fill_color = colors[1])
        
    hover = HoverTool(tooltips = [('Interval', '@interval'),
                              ('Count', str("@" + column))])
    plot.add_tools(hover)
    
    output_notebook()
    show(plot)

    # 画像の緑成分、黄色成分
def get_percentage_of_green_pixels(image):
    # to HSV
    # 色相(0~180)、彩度(0-255)、輝度(0-255)の値をとる
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #print(hsv.shape)

    # 緑マスク取得
    hsv_lower = (40,40,40)
    hsv_higher = (70,255,255)
    green_mask = cv2.inRange(hsv, hsv_lower, hsv_higher) #0 or 255に変換
    #print(green_mask)

    # 全画素数で平均を取り、255で割って正規化
    return float(np.sum(green_mask)) / 255 / (600*800)

def get_percentage_of_yellow_pixels(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # get yellow mask
    hsv_lower = (25, 40, 40)
    hsv_higher = (35, 255, 255)
    yellow_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)
    
    return float(np.sum(yellow_mask)) / 255 / (600*800)

def add_green_pixels_percentage(df):
    green = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row.path
        image = cv2.imread(path)
        green.append(get_percentage_of_green_pixels(image))
 
    green_df = pd.DataFrame(green)
    green_df.columns = ['green_pixels']
    df = pd.concat([df, green_df], axis=1)    
    return df
    
def add_yellow_pixels_percentage(df):
    """
    グリーンのピクセルパーセンテージを取得
    """
    yellow = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row.path
        image = cv2.imread(path)
        yellow.append(get_percentage_of_yellow_pixels(image))
 
    yellow_df = pd.DataFrame(yellow)
    yellow_df.columns = ['yellow_pixels']
    df = pd.concat([df, yellow_df], axis=1)    
    return df


def get_image_brightness(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # get average brightness
    return np.array(gray).mean()

def add_brightness(df):
    """
    画像の明度を取得してDFにcolmunとして追記
    """
    brightness = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row.image_id  
        image = cv2.imread(TRAIN_DIR + img_id)
        brightness.append(get_image_brightness(image))
        
    brightness_df = pd.DataFrame(brightness)
    brightness_df.columns = ['brightness']
    df = pd.concat([df, brightness_df], axis=1)
    #df.columns = ['image_id', 'brightness']
    
    return df

def plot_image_examples(df, rows=3, cols=3, title='Image examples'):
    """
    データセットを row, col数ランダムに表示
    """
    fig, axs = plt.subplots(rows, cols, figsize=(15,10))
    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(len(train), size=1)[0]
            img_id = train.iloc[idx].image_id
            img = Image.open(TRAIN_DIR+img_id)

            axs[row, col].imshow(img)
            axs[row, col].axis('off')
    plt.suptitle(title)    
    plt.show()

for name in sorted(train['disease_name'].unique()):
    plot_image_examples(train[train['disease_name']==name], title=name)