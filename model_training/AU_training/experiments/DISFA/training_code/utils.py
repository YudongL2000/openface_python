import numpy as np
from sklearn import svm
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def compute_F1(ground_truth, prediction):
    tp = np.sum(ground_truth == 1 & prediction == 1)
    fp = np.sum(ground_truth == 0 & prediction == 1)
    fn = np.sum(ground_truth == 1 & prediction == 0)
    tn = np.sum(ground_truth == 0 & prediction == 0)

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    if (precision + recall == 0):
        return 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        return f1


def svm_train_linear(train_labels, train_samples, hyper):
    clf = make_pipeline(StandardScaler(), SVC(C=hyper.C, gamma='auto'))
    clf.fit(train_samples, train_labels)
    return clf

def svm_train_linear(train_labels, train_samples, hyper):
    clf = make_pipeline(StandardScaler(), SVR(C=hyper.C, epsilon=hyper.e))
    clf.fit(train_samples, train_labels)
    return clf


    
