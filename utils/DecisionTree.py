# -*- coding: UTF-8 -*-
"""
@Project ：Pred \n 
@File    ：DecisionTree.py \n
@Author  ：guo \n
@Date    ：2023/5/17 下午6:03 \n
"""
from sklearn.tree import DecisionTreeClassifier


def DecisionTreeClassify(train_data, train_label, criterion="gini"):
    """
    use DecisionTree to fit data
    :param criterion: criterion of DecisionTree, {“gini”, “entropy”, “log_loss”}, default=”gini”
    :param train_data:  train_Data
    :param train_label: train_Label
    :return: model
    """
    clf = DecisionTreeClassifier(criterion=criterion)
    clf.fit(train_data, train_label)
    return clf

def DecisionTree_Pred(model,test_data):
    """
    predictive classification
    :param model: DecisionTree model
    :param test_data: test_Date
    :return: Predictive value
    """
    pred = model.predict(test_data)
    return pred
