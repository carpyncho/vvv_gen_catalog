#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob

import numpy as np
import pandas as pd

PATH = "/home/data/carpyncho/stored/light_curves/"

GLOB_FILES = os.path.join(PATH, "*", "features_*.npy")

FILES = dict(
    (os.path.basename(os.path.dirname(fname)), fname)
    for fname in glob.glob(GLOB_FILES))


def read_data(tile, random_seed=None):
    return np.load(FILES[tile])


read_data("b278")
