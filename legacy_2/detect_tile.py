
# coding: utf-8

# # Features Selection MonoVariate

# In[1]:


#~ get_ipython().magic(u'matplotlib inline')

import pickle
import time

import pandas as pd

from IPython import display as d
# from IPython import

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()

import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn import feature_selection as fs
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import (
    KFold, StratifiedKFold, train_test_split)
from sklearn import preprocessing as prp
import sklearn

from joblib import Parallel, cpu_count, delayed

from libs import container
from libs.experiment import (
    WithAnotherExperiment, KFoldExperiment, roc,
    discretize_classes, clean_features)



# ## 1. Load Data

# In[2]:


#~ start = time.time()

df = pd.read_pickle("data/normalized2500.pkl")

scaler = {k: (v0, v1) for k,v0,v1 in pickle.load(open("data/scaler.pkl"))}


# ##  2. Preprocess
#
# ### 2.1. Discretize the classes

# In[3]:


def corrector(row):
    o3t = (row.ogle3_type or "").split("-", 1)[0]
    scls = row.scls
    return o3t, scls

classes = dict(set(map(tuple, df[["ogle3_type", "cls"]].values)))
sclasses = dict(set(map(tuple, df[["ogle3_type", "scls"]].apply(corrector, axis=1).values)))
tclasses = {k: v for v, k in enumerate(df.tcls.unique())}

def to_tcls(row):
    tcls = str(row.tcls)
    return tclasses[tcls]

df.tcls = df.apply(to_tcls, axis=1)

d.display(d.Markdown("**Classes**"))
d.display(classes)

d.display(d.Markdown("----"))
d.display(d.Markdown("**Simplified Classes**"))
d.display(sclasses)

d.display(d.Markdown("----"))
d.display(d.Markdown("**Tile Classes**"))
d.display(tclasses)


# ### 2.2. Removes all low-variance and "bad" features

# In[4]:


X_columns = clean_features({"df": df}, "df")
# X_columns = X_columns.drop("AndersonDarling")
X_columns = X_columns[~(X_columns.str.startswith("Freq2_") | X_columns.str.startswith("Freq3_"))]
print("Total features:", X_columns.size)



tile_unk = df.groupby("tcls").apply(lambda x: x[x.scls==0].sample(500))

min_size = df[df.scls == 3].groupby("tcls").size().min()
tile_balanced = df.groupby("tcls").apply(lambda x: x[x.scls==3].sample(min_size))

tile_mix = pd.concat([tile_balanced, tile_unk])


def experiment(clf, x_train, y_train, x_test, y_test, pos_label):
    clf = sklearn.clone(clf)
    clf.fit(x_train, y_train)

    predictions = clf.predict(x_test)
    probabilities = clf.predict_proba(x_test)

    fpr, tpr, thresholds = metrics.roc_curve(
        y_test, 1.-probabilities[:,0], pos_label=pos_label)
    prec_rec_curve = metrics.precision_recall_curve(
        y_test, 1.- probabilities[:,0], pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)

    return container.Container({
            'fpr': fpr,
            'tpr': tpr,
            'thresh': thresholds,
            'roc_auc': roc_auc,
            'prec_rec_curve': prec_rec_curve,
            'y_test': y_test,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': metrics.confusion_matrix(y_test, predictions)})


# kfold
def kfolds(data, X_columns, y_column, clf, nfolds, verbose=True):
    skf = StratifiedKFold(n_splits=nfolds)

    subject_df = data

    x = subject_df[X_columns].values
    y = subject_df[y_column].values
    y_real = subject_df[y_column].values

    probabilities = None
    predictions = np.array([])
    y_testing = np.array([])
    y_testing_real = np.array([])

    for train, test in skf.split(x, y):
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        y_test_real = y_real[test]

        rst = experiment(clf, x_train, y_train, x_test, y_test, pos_label=0)

        probabilities = (
            rst.probabilities if probabilities is None else
            np.vstack([probabilities, rst.probabilities]))
        predictions = np.hstack([predictions, rst.predictions])
        y_testing = np.hstack([y_testing, y_test])
        y_testing_real = np.hstack([y_testing_real, y_test_real])
        del rst

    fpr, tpr, thresholds = metrics.roc_curve(
        y_testing, 1.-probabilities[:,0], pos_label=0)
    prec_rec_curve = metrics.precision_recall_curve(
        y_testing, 1.- probabilities[:,0], pos_label=0)
    roc_auc = metrics.auc(fpr, tpr)

    if verbose:
        print metrics.classification_report(y_testing, predictions)
        print "-" * 80

    return container.Container({
        'fpr': fpr,
        'tpr': tpr,
        'thresh': thresholds,
        'roc_auc': roc_auc,
        'prec_rec_curve': prec_rec_curve,
        'y_test': y_testing,
        'y_test_real': y_testing_real,
        'predictions': predictions,
        'probabilities': probabilities,
        'confusion_matrix': metrics.confusion_matrix(y_testing, predictions)})



def _experiment(job_n, df, X_columns):
    print "Starting job ", job_n
    clf = SVC(kernel='poly', probability=True)
    return kfolds(
        df, X_columns, y_column="tcls",
        clf=clf, nfolds=10, verbose=False)

# ----
# ## Gausian noise

# In[29]:


def gn_columns(df, to_shuffle):
    size = len(df)
    columns = {c: np.copy(df[c].values) for c in df.columns}
    for c in to_shuffle:
        columns[c] = np.random.normal(size)
    new_df = pd.DataFrame(columns)
    return new_df

import multiprocessing as mp
class HandWritingParallel(mp.Process):

    def __init__(self, idx, df, X_columns):
        mp.Process.__init__(self)
        self.idx = idx
        self.noises = noises
        self.X_columns = X_columns
        self.queue = mp.Queue()

    def run(self):
        r = []
        for noise in self.noises:
            r.append(_experiment(self.idx, noise, self.X_columns))
#         self.queue.put(map(dict, r))

    def result(self):
        return self.queue.get()



# In[30]:


noise = gn_columns(tile_mix, X_columns)


# In[31]:


noise.tcls.unique()


# In[ ]:


# %%cache data/gn_mix_shuffle.pkl gn_mix_shuffle
#~ %%debug

procs = []
for idx in range(3):
    noises = [gn_columns(tile_mix, X_columns), gn_columns(tile_mix, X_columns)]
    proc = HandWritingParallel(idx, noises, X_columns)
    proc.run()
#     procs.append(proc)
for proc in procs:
    proc.join()

# with Parallel(n_jobs=cpu_count()-1) as jobs:
#     gn_mix_shuffle = jobs(delayed(_experiment)(idx, noise, X_columns)
#                    for idx, noise in gn_columns(tile_mix, X_columns, 100))
# gn_mix_shuffle = map(dict, gn_mix_shuffle)
# pickle.dump({"gn_mix_shuffle": gn_mix_shuffle}, open("data/gn_mix_shuffle.pkl", "wb"))
