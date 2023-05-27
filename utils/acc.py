import pandas as pd
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
    print(df)
    # print(confusion_matrix(test_label, pred)) # 直接调用sklearn的混淆矩阵api
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, F1


def print_evaluation(pred, label):
    Accuracy, Precision, Recall, F1 = evaluation(pred, label)
    print("Accuracy:", Accuracy, "Precision:", Precision, "Recall:", Recall, "F1:", F1)
