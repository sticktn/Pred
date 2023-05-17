# -*- coding: UTF-8 -*-
"""
@Project ：Pred \n 
@File    ：MLP.py \n
@Author  ：guo \n
@Date    ：2023/5/17 下午4:37 \n
"""
from sklearn.neural_network import MLPClassifier


def MLPCLassify(train_data, train_label):
    """
    use MLP to fit data
    :param train_data:  train_Data
    :param train_label: train_Label
    :return: model
    """
    clf = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(100, 50), random_state=1)
    clf.fit(train_data, train_label)
    return clf

def MLP_Pred(model,test_data):
    """
    predictive classification
    :param model: MLP model
    :param test_data: test_Date
    :return: Predictive value
    """
    pred = model.predict(test_data)
    return pred
