import numpy as np
from sklearn import svm


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


def evaluate_regression_results(labels, ground_truth):
    classes = np.sort(np.unique(ground_truth))
    accuracies = np.zeros(len(classes))
    F1s = np.zeros(len(classes))
    corrs = np.correlate(labels, ground_truth)
    rms = np.sqrt(np.mean((labels-ground_truth)**2))
    std_g = np.std(ground_truth)
    std_p = np.std(labels)
    ccc = 2 * corrs * std_g * std_p / (np.square(std_g) + np.square(std_p) + np.square(np.mean(labels) - np.mean(ground_truth)))
    label_dists = np.zeros((len(labels), len(classes)))
    for i in range(len(classes)):
        label_dists[:,i] = np.abs(labels - classes[i])
    labels = np.min(label_dists.T)
    labels = labels.T
    
    for i in (len(classes)):
        labels[labels==i] = classes[i]
    for i in range(len(classes)):
        pos_samples = (ground_truth==classes[i])
        neg_samples = (ground_truth!= classes[i])
        
        pos_labels = (labels==classes[i])
        neg_labels = (labels!=classes[i])
        
        TPs = np.sum(np.logical_and(pos_samples, pos_labels))
        TNs = np.sum(np.logical_and(neg_samples, neg_labels))
        
        FPs = np.sum(np.logical_and(pos_samples, neg_labels))
        FNs = np.sum(np.logical_and(neg_samples, pos_labels))
        
        accuracies[i] = (TPs + TNs) / len(pos_samples)
        F1s[i] = 2 * TPs / (2*TPs + FNs + FPs)
    return accuracies, F1s, corrs, ccc, rms, classes


def svm_test_linear(test_labels, test_samples, model, Labels=[1, 0]):
    prediction = model.predict(test_samples)
    l1_inds = prediction > 0
    l2_inds = prediction <= 0
    prediction[l1_inds] = Labels[0]
    prediction[l2_inds] = Labels[1]
    tp = np.sum(np.logical_and(test_labels == 1.0, prediction == 1.0))
    fp = np.sum(np.logical_and(test_labels == 0.0, prediction == 1.0))
    fn = np.sum(np.logical_and(test_labels == 1.0, prediction == 0.0))
    tn = np.sum(np.logical_and(test_labels == 0.0, prediction == 0.0))
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)
    result = f1
    return result, prediction


def svr_test_linear(test_labels, test_samples, model, Labels=[1,0]):
    prediction = model.predict(test_samples)
    prediction[prediction<0]=0
    prediction[prediction>5]=5
    result = np.correlate(test_labels, prediction)
    accuracies, F1s, corrs, ccc, rms, classes = evaluate_regression_results(prediction, test_labels)
    result = ccc
    return result, prediction


    

