import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


class Evaluation:
    def __init__(self,pred,test_label):
        self.pred = pred
        self.test_label = test_label
        TP = ((test_label == 1) & (pred == 1)).sum()
        TN = ((test_label == 0) & (pred == 0)).sum()
        FP = ((test_label == 0) & (pred == 1)).sum()
        FN = ((test_label == 1) & (pred == 0)).sum()
        self.confusion_matrix = pd.DataFrame([[TP, FP], [TN, FN]], columns=['T', 'F'], index=['P', 'N'])
        self.precision = TP / (TP + FP)
        self.recall = TP / (TP + FN)
        self.accuracy = (TP + TN) / (TP + TN + FP + FN)
        self.F1 = 2 * self.precision * self.recall / (self.precision + self.recall)

    def print_evaluation(self):
        print("Accuracy:", self.accuracy, "\nPrecision:", self.precision, "\nRecall:", self.recall, "\nF1:", self.F1)

    def draw_confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        cm = confusion_matrix(self.test_label, self.pred)
        cm_display = ConfusionMatrixDisplay(cm).plot()
        plt.show()

    # def draw_roc(pred_pro, label):
    #     """
    #     draw roc curve
    #     :param pred_pro: predictive classification probability
    #     :param label: true classification
    #     :return: None
    #     """
    #     from sklearn.metrics import roc_curve
    #     from sklearn.metrics import RocCurveDisplay
    #     fpr, tpr, thresholds = roc_curve(label, pred_pro, pos_label=1)
    #     roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    #     plt.show()

    def draw_recall(self,pred_pro):
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import PrecisionRecallDisplay
        precision, recall, thresholds = precision_recall_curve(self.test_label, pred_pro, pos_label=1)
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
        plt.show()
