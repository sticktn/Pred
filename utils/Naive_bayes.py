# -*- coding: UTF-8 -*-
"""
@Project ：Pred \n 
@File    ：Naive_bayes.py \n
@Author  ：guo \n
@Date    ：2023/5/17 下午5:34 \n
"""
"""
在scikit-learn中，一共有3个朴素贝叶斯的分类算法类。分别是GaussianNB，MultinomialNB和BernoulliNB。
其中GaussianNB就是先验为高斯分布的朴素贝叶斯，MultinomialNB就是先验为多项式分布的朴素贝叶斯，而BernoulliNB就是先验为伯努利分布的朴素贝叶斯。
这三个类适用的分类场景各不相同，一般来说，如果样本特征的分布大部分是连续值，使用GaussianNB会比较好。
如果如果样本特征的分大部分是多元离散值，使用MultinomialNB比较合适。而如果样本特征是二元离散值或者很稀疏的多元离散值，应该使用BernoulliNB
"""
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

def BayesClassify(train_data, train_label, type="Gaussian"):
    """
    use Bayes to fit data
    :param type: type of Bayes
    :param train_data:  train_Data
    :param train_label: train_Label
    :return: model
    """
    if type == "Gaussian":
        clf = GaussianNB()
    elif type == "Multinomial":
        clf = MultinomialNB()
    elif type == "Bernoulli":
        clf = BernoulliNB()
    else:
        print("type error")
        return
    clf.fit(train_data, train_label)
    return clf

def Beyes_Pred(model,test_data):
    """
    predictive classification
    :param model: Bayes model
    :param test_data: test_Date
    :return: Predictive value
    """
    pred = model.predict(test_data)
    return pred
