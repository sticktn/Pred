# -*- coding: UTF-8 -*-
"""
@Project ：Pred \n 
@File    ：DecisionTree_main.py \n
@Author  ：guo \n
@Date    ：2023/5/17 下午6:08 \n
"""
from utils.read_data import read
from utils.DecisionTree import DecisionTreeClassify,DecisionTree_Pred
from utils.acc import accuracy


train_data_std, train_label, val_data_std, val_label, test_data_std, test_label = read()
model = DecisionTreeClassify(train_data_std, train_label, criterion="gini")

train_pred = DecisionTree_Pred(model, train_data_std)
train_acc = accuracy(train_pred, train_label)
print("train dataset accuracy:", train_acc)

val_pred = DecisionTree_Pred(model, val_data_std)
val_acc = accuracy(val_pred, val_label)
print("val dataset accuracy:", val_acc)

test_pred = DecisionTree_Pred(model, test_data_std)
test_acc = accuracy(test_pred, test_label)
print("test dataset accuracy:", test_acc)