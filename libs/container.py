
import glob
import os

import numpy as np

import pandas as pd

class Container(dict):
    
    def __init__(self, data):
        super(Container, self).__init__(data)
        self._repr = "{" + ", ".join(data.keys()) + "}"
    
    def __dir__(self):
        return list(self.keys())
    
    def __repr__(self):
        return "<Container({})>".format(self._repr)
    
    def __getattr__(self, an):
        return self[an]
        
        
def read(path):
    data = {}
    for fpath in glob.glob(os.path.join(path, "*.npy")):
        fname = os.path.basename(fpath).split("_", 1)[0]
        print "Loading '{}'...".format(fpath)
        data[fname] = pd.DataFrame(np.load(fpath))
    return Container(data)
        
    