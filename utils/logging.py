from time import time
import os
import random
import numpy as np
import pandas as pd


def decorate(s: str, decoration=None):
    """目立つようにプリント
    from https://github.com/nyk510/vivid/blob/master/vivid/utils.py
    ex:
    print(decorate('start run blocks...'))
    -> ★★★★★★★★★★★★★★★★★★★★ start run blocks... ★★★★★★★★★★★★★★★★★★★★
    """
    if decoration is None:
        decoration = '★' * 20

    return ' '.join([decoration, str(s), decoration])


class Timer:
    """タイマークラス
    from https://github.com/nyk510/vivid/blob/master/vivid/utils.py
    ex:
    with Timer(prefix='run test={}'.format(test)):
        #処理

    ->run test=True 0.376[s]
    """

    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' ', verbose=0):

        if prefix:
            format_str = str(prefix) + sep + format_str
        if suffix:
            format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None
        self.verbose = verbose

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        if self.verbose is None:
            return
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)
