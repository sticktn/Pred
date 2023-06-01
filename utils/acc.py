import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def accuracy(pred, test_label):
    """
    Computational model accuracy
    :param pred: predictive classification
    :param test_label: true classification
    :return:Computational model accuracy
    """
    length = len(test_label)
    accuracy = (pred == test_label).sum() / length
    return accuracy


def evaluation(pred, test_label):
    """
    Evaluation of classification model
    :param pred: predictive classification
    :param test_label: true classification
    :return: Evaluation of classification model
    """
    TP = ((test_label == 1) & (pred == 1)).sum()
    TN = ((test_label == 0) & (pred == 0)).sum()
    FP = ((test_label == 0) & (pred == 1)).sum()
    FN = ((test_label == 1) & (pred == 0)).sum()
    df = pd.DataFrame([[TP, FP], [TN, FN]], columns=['T', 'F'], index=['P', 'N'])
    print("confusion matrix:")
    print(df)
    # print(confusion_matrix(test_label, pred)) # 直接调用sklearn的混淆矩阵api
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, F1


def print_evaluation(pred, label):
    Accuracy, Precision, Recall, F1 = evaluation(pred, label)
    print("Accuracy:", Accuracy, "\nPrecision:", Precision, "\nRecall:", Recall, "\nF1:", F1)

def draw_confusion_matrix(pred, label):
    """
    draw confusion matrix
    :param pred: predictive classification
    :param label: true classification
    :return: None
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(label, pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.show()

def draw_roc(pred_pro, label):
    """
    draw roc curve
    :param pred_pro: predictive classification probability
    :param label: true classification
    :return: None
    """
    from sklearn.metrics import roc_curve
    from sklearn.metrics import RocCurveDisplay
    fpr, tpr, thresholds = roc_curve(label, pred_pro, pos_label=1)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()

def draw_recall(pred_pro,label):
    """
    draw recall curve
    :param pred_pro: predictive classification probability
    :param label: true classification
    :return: None
    """
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import PrecisionRecallDisplay
    precision, recall, thresholds = precision_recall_curve(label, pred_pro, pos_label=1)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.show()
