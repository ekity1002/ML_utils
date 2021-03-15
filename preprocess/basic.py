import japanize_matplotlib
import warnings
from collections import defaultdict
from matplotlib_venn import venn2  # venn図を作成する用
from pandas_profiling import ProfileReport  # profile report を作る用
import pandas_profiling as pdp
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
japanize_matplotlib.japanize()
%matplotlib inline


# 警告が鬱陶しい時はこれを記述
warnings.filterwarnings('ignore')
