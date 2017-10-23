
import glob
import os

import numpy as np

import pandas as pd

class Container(dict):
    
    def __init__(self, path):
        files = []
        for fpath in glob.glob(os.path.join(path, "*.npy")):
            fname = os.path.basename(fpath).split("_", 1)[0]
            print "Loading '{}'...".format(fpath)
            self[fname] = pd.DataFrame(np.load(fpath))
            files.append(fname)
        self._repr = "{" + ", ".join(files) + "}"
    
    def __dir__(self):
        return list(self.keys())
    
    def __repr__(self):
        return "<Container({})>".format(self._repr)
    
    def __getattr__(self, an):
        return self[an]
        