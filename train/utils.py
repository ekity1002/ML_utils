
import os
import random
import numpy as np
import torch
import transformers
from transformers import BertTokenizer
from tqdm import tqdm
import umap
import warnings
warnings.filterwarnings('ignore')


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
