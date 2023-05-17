# -*- coding: UTF-8 -*-
"""
@Project ：Pred \n 
@File    ：Bayes_main.py \n
@Author  ：guo \n
@Date    ：2023/5/17 下午5:40 \n
"""
from utils.acc import accuracy
from utils.Naive_bayes import BayesClassify, Beyes_Pred
from utils.read_data import read

train_data_std, train_label, val_data_std, val_label, test_data_std, test_label = read()



# GaussianNB 建议在连续值的时候使用，此处正确率很低
#  MultinomialNB不支持负数


print("type:", type)
model = BayesClassify(train_data_std, train_label, type="Bernoulli")

train_pred = Beyes_Pred(model, train_data_std)
train_acc = accuracy(train_pred, train_label)
print("train dataset accuracy:", train_acc)

val_pred = Beyes_Pred(model, val_data_std)
val_acc = accuracy(val_pred, val_label)
print("val dataset accuracy:", val_acc)

test_pred = Beyes_Pred(model, test_data_std)
test_acc = accuracy(test_pred, test_label)
print("test dataset accuracy:", test_acc)
