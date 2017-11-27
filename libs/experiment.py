import pandas as pd

from sklearn import preprocessing as prp
from sklearn import metrics

from . import container


class Experiment(object):
    
    def __init__(self, data, clf, pcsl, sampler=None, verbose=True):
        self.data = data
        self.clf = clf
        self.pcls = pcls
        self.sampler = sampler
        self.verbose = verbose

    def experiment(clf, train_name, test_name):
        clf = sklearn.clone(self.clf)
        
        # retrieve the train data
        if isinstance(train_name, str):
            train_df = data[train_name]
        else:
            train_df = pd.concat([data[n] for n in train_name], ignore_index=True)
            train_name = " ".join(train_name)
        
        test_df = data[test_name]
    
        cfilter = [sclasses[pcls], sclasses[""]]
        train_df = train_df[train_df.scls.isin(cfilter)]
        test_df = test_df[test_df.scls.isin(cfilter)]
    
        x_train = prp.StandardScaler().fit_transform(
            train_df[X_columns].values)
        y_train = train_df["scls"]
        
        if self.sampler:
            x_train, y_train = ros.fit_sample(x_train, y_train)        

        x_test = prp.StandardScaler().fit_transform(
            test_df[X_columns].values)
        y_testing = test_df["scls"]
    
        clf.fit(x_train, y_train)

        predictions = clf.predict(x_test)
        probabilities = clf.predict_proba(x_test)

        fpr, tpr, thresholds = metrics.roc_curve(
            y_testing, 1.-probabilities[:,0], pos_label=cfilter[0])
        prec_rec_curve = metrics.precision_recall_curve(
            y_testing, 1.- probabilities[:,0], pos_label=cfilter[0])
        roc_auc = metrics.auc(fpr, tpr)

        return container.Container({
                'test_name': test_name,
                'train_name': train_name,
                'fpr': fpr, 
                'tpr': tpr, 
                'thresh': thresholds, 
                'roc_auc': roc_auc, 
                'prec_rec_curve': prec_rec_curve,
                'y_test': y_testing, 
                'predictions': predictions,
                'probabilities': probabilities, 
                'confusion_matrix': metrics.confusion_matrix(y_testing, predictions)})


    def __call__(self, train_name):
        results = []
        for test_name in self.data.keys():
            if isinstance(train_name, str):
                compare = test_name != train_name
            else:
                compare = test_name not in train_name
            if compare:
                rst = self.experiment(train_name, test_name)
                results.append(rst)
                if verbose:
                    print "{} (TRAIN) Vs. {} (TEST)".format(rst.train_name, rst.test_name)
                    print metrics.classification_report(rst.y_test, rst.predictions)
                    print "-" * 80
        return tuple(results)


def roc(results):
    cmap = cm.get_cmap("plasma")
    colors = iter(cmap(np.linspace(0, 1, len(results))))

    for res  in results:
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