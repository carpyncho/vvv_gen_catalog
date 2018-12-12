
# coding: utf-8

# # Entreno en 5000 pruebo en TODO

# In[1]:


import pickle

import numpy as np

import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from libs.container import Container
from libs.display import d
from libs.experiment import KFoldExperiment, WithAnotherExperiment, roc


# ## Funcion de ayuda

# In[2]:


def run_test(clf, tile, X_columns, y_column):
    X_test = tile[X_columns].values
    y_test = tile[y_column].values
    
    print np.all(np.isfinite(X_test))
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)
    
    fpr, tpr, thresholds = metrics.roc_curve(
        y_test, 1.-probabilities[:,0], pos_label=1)

    prec_rec_curve = metrics.precision_recall_curve(
        y_test, 1.- probabilities[:,0], pos_label=1)

    roc_auc = metrics.auc(fpr, tpr)
    
    print metrics.classification_report(y_test, predictions)
    
    return Container({
        'fpr': fpr,
        'tpr': tpr,
        'thresh': thresholds,
        'roc_auc': roc_auc,
        'prec_rec_curve': prec_rec_curve,
        'y_test': y_test,
        'predictions': predictions,
        'probabilities': probabilities,
        'confusion_matrix': metrics.confusion_matrix(y_test, predictions)})


# In[3]:


sample = pd.read_pickle("data/o3o4vZ/scaled/s5k.pkl.bz2")
sample["tile"] = sample["id"].apply(lambda i: "b" + str(i)[1:4])
sample["cls"] = sample.vs_type.apply(lambda x: 0 if x == "" else 1)

no_features = ["id", "vs_catalog", "vs_type", "ra_k", "dec_k", "tile", "cls"] 
X_columns = [c for c in sample.columns if c not in no_features]

grouped = sample.groupby("tile")
train = Container({k: grouped.get_group(k).copy() for k in grouped.groups.keys()})

del grouped, sample


# Reescalamos la muestra de TOTAL con media y desvio del de 5k

# In[4]:


skl = pickle.load(open("data/o3o4vZ/scalers/scaler_5k.pkl"))


# In[5]:


sample = pd.read_pickle("data/o3o4vZ/nonull/sALL.pkl.bz2")

sample[X_columns] = skl.transform(sample[X_columns])

sample["tile"] = sample["id"].apply(lambda i: "b" + str(i)[1:4])
sample["cls"] = sample.vs_type.apply(lambda x: 0 if x == "" else 1)

grouped = sample.groupby("tile")
test = Container({k: grouped.get_group(k).copy() for k in grouped.groups.keys()})

del grouped, sample
test


# In[6]:


cpu = joblib.cpu_count()
clf_med = RandomForestClassifier(n_estimators=500, criterion="entropy", n_jobs=cpu)
clf_med.fit(train.b278[X_columns].values, train.b278.cls.values)


# In[7]:


result_med = run_test(clf_med, test.b278, X_columns, "cls")


# In[ ]:


roc({"5K vs. 20K": result_med, "2.5K vs. 20K": result_small})


# In[9]:


df = test.b278[X_columns]


# In[11]:


df.values.dtype.char


# In[15]:


np.isfinite(df.values).all()

