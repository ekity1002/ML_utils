import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

from collections import defaultdict
from matplotlib_venn import venn2  # venn図を作成する用
import seaborn as sns
sns.set()
japanize_matplotlib.japanize()
%matplotlib inline

import pandas_profiling as pdp
from pandas_profiling import ProfileReport  # profile report を作る用
from ptitprince import RainCloud

# 警告が鬱陶しい時はこれを記述
import warnings
warnings.filterwarnings('ignore')


# other libs


def rainCloud(df, x, y, ax=None, width_viol=2.):
    """
    raincloud:https://github.com/pog87/PtitPrince
    TODO:使い方
    """
    fig, ax = plt.subplots(figsize=(20, 6))
    RainCloud(data=train, y='log_likes',
              x='dating_period', ax=ax, width_viol=2.)
    ax.grid()
    RainCloud(data=df, y='log_likes', x='dating_period', ax=ax, width_viol=2.)
    ax.grid()


def histplot():
    """seaborn histplot"""
    fig, ax = plt.subplots(figsize=(8, 8))
    # train と testの比較もできる
    sns.histplot(np.log1p(pred), label='Test Predict', ax=ax, color='black')
    sns.histplot(oof, label='Out of fold', ax=ax, color='C1')
    ax.legend()
    ax.grid()


def venn_fugure():
    """ベン図を書いて2つのカテゴリの重複を見る"""
    def filter_names(master_df: pd.DataFrame, input_df: pd.DataFrame) -> set:
        """
        master_df: train,testを結合した図
        input_df: train or test
        """
        idx = master_df['object_id'].isin(input_df['object_id'])
        return set(master_df[idx]['name'].unique())

    # 一般化
    def plot_one2many_relation_table_intersection(train, test, name: str, ax: plt.Axes) -> plt.Axes:
        """ベン図をかく"""
        master_df = read_csv(name)
        venn2(subsets=(filter_names(master_df, train), filter_names(master_df, test)), 
            set_labels=('Train', 'Test'), 
            ax=ax, 
            set_colors=('C0', 'C1'))
        ax.set_title(f'{name}の分布')
        return ax

    table_names = [
    'production_place',
    'material',
    'object_collection',
    'historical_person',
    'technique'
    ]

    n_figs = len(table_names)
    n_cols = 3
    n_rows = - (- n_figs // n_cols)

    fig, axes = plt.subplots(figsize=(4 * n_cols, 3 * n_rows), ncols=n_cols, nrows=n_rows)
    axes = np.ravel(axes)

    for name, ax in zip(table_names, axes):
        plot_one2many_relation_table_intersection(name=name, ax=ax)
    fig.tight_layout()


###アソシエーション分析
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
def association_analisys():
    """データ全体に対して あるカテゴリを持つデータが占める割合をみる
    !　pip install mlxtend 必要
    """
    df = pd.crosstab(target_df['object_id'], target_df['name'])
     #material ごとの 全オブジェクト中の割合. 複数持つobjectもあるので supportの和は1より大きい
    freq_item_df = apriori(df, min_support=.005, use_colnames=True)
    freq_item_df.sort_values('support', ascending=False)