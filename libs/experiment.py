
import warnings

import numpy as np

import pandas as pd

from matplotlib import cm
from matplotlib import pyplot as plt

import seaborn as sns

import sklearn
from sklearn import feature_selection as fs
from sklearn import preprocessing as prp
from sklearn.model_selection import (
    KFold, StratifiedKFold, train_test_split)
from sklearn import metrics

from . import container


class Experiment(object):
<<<<<<< HEAD
    
    def __init__(self, data, clf, pcls, X_columns, y_column, clsnum, 
                 ncls="", sampler=None, verbose=True, real_y_column=None):
=======

    def __init__(self, data, clf, pcls, X_columns, y_column, clsnum,
                 ncls, sampler=None, verbose=True, real_y_column=None):
>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd
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
<<<<<<< HEAD
        
=======

>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd
        if real_y_column is None:
            real_y_column = "{}_orig".format(y_column)
            columns = list(data.values())[0].columns
            if real_y_column not in columns:
                real_y_column = y_column
        self.real_y_column = real_y_column
<<<<<<< HEAD
        
    
    def experiment(self, x_train, y_train, x_test, y_test):
        x_train = prp.StandardScaler().fit_transform(x_train)
        x_test = prp.StandardScaler().fit_transform(x_test)
        
        if self.sampler:
            sampler = sklearn.clone(self.sampler)
            x_train, y_train = sampler.fit_sample(x_train, y_train)
        
=======


    def experiment(self, x_train, y_train, x_test, y_test):
        if self.sampler:
            sampler = sklearn.clone(self.sampler)
            x_train, y_train = sampler.fit_sample(x_train, y_train)

>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd
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
<<<<<<< HEAD
                'fpr': fpr, 
                'tpr': tpr, 
                'thresh': thresholds, 
                'roc_auc': roc_auc, 
                'prec_rec_curve': prec_rec_curve,
                'y_test': y_test, 
                'predictions': predictions,
                'probabilities': probabilities, 
                'confusion_matrix': metrics.confusion_matrix(y_test, predictions)})
    
=======
                'fpr': fpr,
                'tpr': tpr,
                'thresh': thresholds,
                'roc_auc': roc_auc,
                'prec_rec_curve': prec_rec_curve,
                'y_test': y_test,
                'predictions': predictions,
                'probabilities': probabilities,
                'confusion_matrix': metrics.confusion_matrix(y_test, predictions)})

>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd
    def __call__(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.run(*args, **kwargs)

<<<<<<< HEAD
    
class WithAnotherExperiment(Experiment):
    
=======

class WithAnotherExperiment(Experiment):

>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd
    def run(self, train_name):
        if isinstance(train_name, str):
            train_df = self.data[train_name]
            train_name = [train_name]
        else:
            train_df = pd.concat(
<<<<<<< HEAD
                [self.data[n] for n in train_name], 
                ignore_index=True)

        results = []
        for test_name in self.data.keys():
=======
                [self.data[n] for n in train_name],
                ignore_index=True)

        results = []
        for test_name in sorted(self.data.keys()):
>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd

            if test_name not in train_name:
                 # retrieve the train data and test dataframe

                test_df = self.data[test_name]

                # filter only the important lines
<<<<<<< HEAD
                train_df = train_df[train_df.scls.isin(self.cfilter)]
                test_df = test_df[test_df.scls.isin(self.cfilter)]
=======
                train_df = train_df[train_df.cls.isin(self.cfilter)]
                test_df = test_df[test_df.cls.isin(self.cfilter)]
>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd

                # split in np arrays
                x_train = train_df[self.X_columns].values
                y_train = train_df[self.y_column].values
                x_test = test_df[self.X_columns].values
                y_test = test_df[self.y_column].values
                y_test_real = test_df[self.real_y_column].values

                rst = self.experiment(x_train, y_train, x_test, y_test)
                rst.update({
<<<<<<< HEAD
                    'test_name': test_name,
                    'y_test_real': y_test_real,
                    'train_name': " + ".join(train_name)})
                
=======
                    'ids': test_df.id.values,
                    'test_name': test_name,
                    'y_test_real': y_test_real,
                    'train_name': " + ".join(train_name)})

>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd
                if self.verbose:
                    print "{} (TRAIN) Vs. {} (TEST)".format(rst.train_name, rst.test_name)
                    print metrics.classification_report(rst.y_test, rst.predictions)
                    print "-" * 80
                results.append(rst)
        return tuple(results)
<<<<<<< HEAD
    

class KFoldExperiment(Experiment):
    
    def run(self, subject, nfolds=10):
        # kfold
        skf = StratifiedKFold(n_splits=nfolds)
        
        subject_df = self.data[subject]
        subject_df = subject_df[subject_df.scls.isin(self.cfilter)]
        
        x = subject_df[self.X_columns].values
        y = subject_df[self.y_column].values
=======


class KFoldExperiment(Experiment):

    def run(self, subject, nfolds=10):
        # kfold
        skf = StratifiedKFold(n_splits=nfolds)

        subject_df = self.data[subject]
        subject_df = subject_df[subject_df.cls.isin(self.cfilter)]

        x = subject_df[self.X_columns].values
        y = subject_df[self.y_column].values
        ids = subject_df.id.values
>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd
        y_real = subject_df[self.real_y_column].values

        probabilities = None
        predictions = np.array([])
        y_testing = np.array([])
        y_testing_real = np.array([])
<<<<<<< HEAD
=======
        u_ids = np.array([])
>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd

        for train, test in skf.split(x, y):
            x_train = x[train]
            y_train = y[train]
            x_test = x[test]
            y_test = y[test]
            y_test_real = y_real[test]
<<<<<<< HEAD
            
            rst = self.experiment(x_train, y_train, x_test, y_test)            
=======

            rst = self.experiment(x_train, y_train, x_test, y_test)
>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd
            probabilities = (
                rst.probabilities if probabilities is None else
                np.vstack([probabilities, rst.probabilities]))
            predictions = np.hstack([predictions, rst.predictions])
            y_testing = np.hstack([y_testing, y_test])
            y_testing_real = np.hstack([y_testing_real, y_test_real])
<<<<<<< HEAD
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
            'y_test_real': y_testing_real,
            'predictions': predictions,
            'probabilities': probabilities, 
            'confusion_matrix': metrics.confusion_matrix(y_testing, predictions)})
        
    

def roc(results, cmap="plasma"):
    cmap = cm.get_cmap(cmap)
    colors = iter(cmap(np.linspace(0, 1, len(results))))
    
=======
            u_ids = np.hstack([u_ids, ids[test]])
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
            "test_name": "kfold",
            'ids': u_ids,
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



def roc(results, cmap="plasma", save_to=None):
    cmap = cm.get_cmap(cmap)
    colors = iter(cmap(np.linspace(0, 1, len(results))))

>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd
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
<<<<<<< HEAD
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    
def discretize_classes(data):
    classes = set()
    for df in data.values():
        classes.update(df.ogle3_type)
    sclasses_names = set(c.split("-", 1)[0] for c in classes)
    classes = dict(zip(sorted(classes), range(len(classes))))
    sclasses = dict(zip(sorted(sclasses_names), range(len(sclasses_names))))

    for df in data.values():
        df["cls"] = df.ogle3_type.apply(classes.get)
        df["real_cls"] = df.ogle3_type.apply(classes.get)
        df["scls"] = df.ogle3_type.apply(lambda v: sclasses.get(v.split("-", 1)[0]))
        df["real_scls"] = df.ogle3_type.apply(lambda v: sclasses.get(v.split("-", 1)[0]))
    
    return data, classes, sclasses


def clean_features(data, name):
    df = data[name]
    X_columns = df.columns[~df.columns.isin([
        "id", "cls", "scls", "ogle3_type", "AMP", "real_cls", "real_scls", "tcls"])]
    
    # remove stellar classes
    X_columns = X_columns[~X_columns.str.startswith("scls_")]
    X_columns
    
    # remove signatures
    X_columns = X_columns[~X_columns.str.startswith("Signature_")]
    X_columns

    # columns with nan and null
    with_nulls = set()
    for df in data.values():
        for c in X_columns:
            if df[c].isnull().any():
                with_nulls.add(c)
    print("Removing {} because null".format(list(with_nulls)))
    X_columns = X_columns[~X_columns.isin(with_nulls)]

    # low variance
    df = pd.concat(data.values())
    y = df["cls"].values

    vt = fs.VarianceThreshold()
    vt.fit(df[X_columns].values, y)
    print("Removing {} because lowvariance".format(list(X_columns[~vt.get_support()])))
    X_columns = X_columns[vt.get_support()]
    
    return X_columns


def union(data, ycolumn, classes, preserve, unionvalue):
    if isinstance(preserve, str):
        preserve = [preserve]
    mapper = {v:(unionvalue if k not in preserve else v)
              for k, v in classes.items()}
    newcls = {k:(unionvalue if k not in preserve else v)
              for k, v in classes.items()}
    orig_name = "{}_orig".format(ycolumn)
    for df in data.values():
        df[orig_name] = df[ycolumn].copy()
        df[ycolumn] = df[ycolumn].apply(mapper.get)
    return data, newcls


def real_vs_predicted_data(result, classes):
    if not isinstance(result, dict):
        real_all = np.concatenate([r.y_test_real for r in result])
        condensed_all = np.concatenate([r.predictions for r in result])
    else:
        real_all = result.y_test_real
        condensed_all = result.predictions
        
    cls, totals, cls0, cls1, cls_name = [], [], [], [], []
    for real_cls in sorted(set(real_all)):
        condensed = condensed_all[real_all == real_cls]
        total = float(len(condensed))
        cls.append(real_cls)
        cls_name.append("{} ({})".format(
            classes[real_cls] or "Unknow", int(real_cls)))
        totals.append(total)
        cls0.append(len(condensed[condensed == 0]) / total) 
        cls1.append(len(condensed[condensed == 3]) / total)
    df = pd.DataFrame({"Total": totals, "Predicted0": 
                       cls0, "Predicted3": cls1, "Name": cls_name}, index=cls)
    return df[["Name", "Total", "Predicted0", "Predicted3"]]


def real_vs_predicted(result, classes):
    df = real_vs_predicted_data(result, classes)
    f, ax = plt.subplots(figsize=(12, 6))

    sns.set_color_codes("pastel")
    ax = sns.barplot(x="Total", y="Name", 
                     data=df, estimator=lambda x: 1, color="b")

    sns.set_color_codes("muted")
    sns.barplot(x="Predicted0", y="Name", data=df, 
                color="b", label="Predicted as Unknow (0)")

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 1.29), ylabel="",
           xlabel="Predicted class")
    sns.despine(left=True, bottom=True)
=======
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)
    
    plt.show()
>>>>>>> 355f3f69c644942bd910a5af461ce51c9b5f1bcd
