
import warnings

import numpy as np

import pandas as pd

from matplotlib import cm
from matplotlib import pyplot as plt

import sklearn
from sklearn import feature_selection as fs
from sklearn import preprocessing as prp
from sklearn.model_selection import (
    KFold, StratifiedKFold, train_test_split)
from sklearn import metrics

from . import container


class Experiment(object):
    
    def __init__(self, data, clf, pcls, X_columns, y_column, clsnum, 
                 ncls="", sampler=None, verbose=True):
        self.data = data
        self.clf = clf
        self.pcls = pcls
        self.ncls = ncls
        self.sampler = sampler
        self.verbose = verbose
        self.X_columns = X_columns
        self.y_column = y_column
        self.clsnum = clsnum
        self.cfilter = [clsnum[pcls], clsnum[ncls]]
    
    def experiment(self, x_train, y_train, x_test, y_test):
        x_train = prp.StandardScaler().fit_transform(x_train)
        x_test = prp.StandardScaler().fit_transform(x_test)
        
        if self.sampler:
            x_train, y_train = self.sampler.fit_sample(x_train, y_train)
        
        clf = sklearn.clone(self.clf)
        clf.fit(x_train, y_train)

        predictions = clf.predict(x_test)
        probabilities = clf.predict_proba(x_test)

        fpr, tpr, thresholds = metrics.roc_curve(
            y_test, 1.-probabilities[:,0], pos_label=self.cfilter[0])
        prec_rec_curve = metrics.precision_recall_curve(
            y_test, 1.- probabilities[:,0], pos_label=self.cfilter[0])
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
    
    def __call__(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.run(*args, **kwargs)

    
class WithAnotherExperiment(Experiment):
    
    def run(self, train_name):
        if isinstance(train_name, str):
            train_df = self.data[train_name]
            train_name = [train_name]
        else:
            train_df = pd.concat(
                [self.data[n] for n in train_name], 
                ignore_index=True)

        results = []
        for test_name in self.data.keys():

            if test_name not in train_name:
                 # retrieve the train data and test dataframe

                test_df = self.data[test_name]

                # filter only the important lines
                train_df = train_df[train_df.scls.isin(self.cfilter)]
                test_df = test_df[test_df.scls.isin(self.cfilter)]

                # split in np arrays
                x_train = train_df[self.X_columns].values
                y_train = train_df[self.y_column].values
                x_test = test_df[self.X_columns].values
                y_test = test_df[self.y_column].values

                rst = self.experiment(x_train, y_train, x_test, y_test)
                rst.update({
                    'test_name': test_name,
                    'train_name': " + ".join(train_name)})
                
                if self.verbose:
                    print "{} (TRAIN) Vs. {} (TEST)".format(rst.train_name, rst.test_name)
                    print metrics.classification_report(rst.y_test, rst.predictions)
                    print "-" * 80
                results.append(rst)
        return tuple(results)
    

class KFoldExperiment(Experiment):
    
    def run(self, subject, nfolds=10):
        # kfold
        skf = StratifiedKFold(n_splits=nfolds)
        
        subject_df = self.data[subject]
        subject_df = subject_df[subject_df.scls.isin(self.cfilter)]
        
        x = subject_df[self.X_columns].values
        y = subject_df[self.y_column].values

        probabilities = None
        predictions = np.array([])
        y_testing = np.array([])

        for train, test in skf.split(x, y):
            x_train = x[train]
            y_train = y[train]
            x_test = x[test]
            y_test = y[test]
            
            rst = self.experiment(x_train, y_train, x_test, y_test)            
            probabilities = (
                rst.probabilities if probabilities is None else
                np.vstack([probabilities, rst.probabilities]))
            predictions = np.hstack([predictions, rst.predictions])
            y_testing = np.hstack([y_testing, y_test])
            del rst
            
        fpr, tpr, thresholds = metrics.roc_curve(
            y_testing, 1.-probabilities[:,0], 
            pos_label=self.cfilter[0])
        prec_rec_curve = metrics.precision_recall_curve(
            y_testing, 1.- probabilities[:,0], 
            pos_label=self.cfilter[0])
        roc_auc = metrics.auc(fpr, tpr)
        
        if self.verbose: 
            print metrics.classification_report(y_testing, predictions)
            print "-" * 80
        
        return container.Container({
            'fpr': fpr, 
            'tpr': tpr, 
            'thresh': thresholds, 
            'roc_auc': roc_auc, 
            'prec_rec_curve': prec_rec_curve,
            'y_test': y_testing, 
            'predictions': predictions,
            'probabilities': probabilities, 
            'confusion_matrix': metrics.confusion_matrix(y_testing, predictions)})
    

def roc(results, cmap="plasma"):
    cmap = cm.get_cmap(cmap)
    colors = iter(cmap(np.linspace(0, 1, len(results))))
    
    if isinstance(results, dict):
        for cname, res  in results.items():
            color = next(colors)
            label = '%s (area = %0.2f)' % (cname, res["roc_auc"])
            plt.plot(res["fpr"], res["tpr"], color=color, label=label)
    else:
        for res in results:
            cname = "Vs.{}".format(res.test_name)
            color = next(colors)
            label = '%s (area = %0.2f)' % (cname, res["roc_auc"])
            plt.plot(res["fpr"], res["tpr"], color=color, label=label)

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    
def resume(label, results, rfunc):
    slice = np.min([r.fpr.size for r in results])
    mfpr = rfunc(np.vstack([r.fpr[:slice] for r in results]), axis=0)
    mtpr = rfunc(np.vstack([r.tpr[:slice] for r in results]), axis=0)
    roc_auc = rfunc([r.roc_auc for r in results])
    return container.Container(test_name=label, fpr=mfpr, tpr=mtpr, roc_auc=roc_auc)


def discretize_classes(data):
    classes = set()
    for df in data.values():
        classes.update(df.ogle3_type)
    sclasses_names = set(c.split("-", 1)[0] for c in classes)
    classes = dict(zip(sorted(classes), range(len(classes))))
    sclasses = dict(zip(sorted(sclasses_names), range(len(sclasses_names))))

    for df in data.values():
        df["cls"] = df.ogle3_type.apply(classes.get)
        df["scls"] = df.ogle3_type.apply(lambda v: sclasses.get(v.split("-", 1)[0]))
    
    return data, classes, sclasses


def clean_features(data, name):
    df = data[name]
    X_columns = df.columns[~df.columns.isin(["id", "cls", "scls", "ogle3_type"])]

    # remove signatures
    X_columns = X_columns[~X_columns.str.startswith("Signature_")]
    X_columns

    # columns with nan and null
    with_nulls = set()
    for df in data.values():
        for c in X_columns:
            if df[c].isnull().any():
                with_nulls.add(c)
    X_columns = X_columns[~X_columns.isin(with_nulls)]

    # low variance
    df = pd.concat(data.values())
    y = df["cls"].values

    vt = fs.VarianceThreshold()
    vt.fit(df[X_columns].values, y)
    X_columns = X_columns[vt.get_support()]
    return X_columns