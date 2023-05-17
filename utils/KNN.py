# -*- coding: UTF-8 -*-
"""
@Project ：Pred \n 
@File    ：KNN.py \n
@Author  ：guo \n
@Date    ：2023/5/17 下午5:08 \n
"""
from sklearn.neighbors import KNeighborsClassifier


def KNNClassify(train_data, train_label, n=3):
    """
    use KNN to fit data
    :param n: k in knn
    :param train_data:  train_Data
    :param train_label: train_Label
    :return: model
    """
    neigh = KNeighborsClassifier(n_neighbors=n)
    neigh.fit(train_data, train_label)
    return neigh


def KNN_Pred(model, test_data):
    """
    predictive classification
    :param model: KNN model
    :param test_data: test_Date
    :return: Predictive value
    """
    pred = model.predict(test_data)
    return pred
