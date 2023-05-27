# -*- coding: UTF-8 -*-
"""
@Project ：Pred \n 
@File    ：DecisionTree_main.py \n
@Author  ：guo \n
@Date    ：2023/5/17 下午6:08 \n
"""
from utils.read_data import read
from utils.DecisionTree import DecisionTreeClassify, DecisionTree_Pred
from utils.acc import print_evaluation
from sklearn.tree import export_text
import joblib

train_data_std, train_label, val_data_std, val_label, test_data_std, test_label = read()
model = DecisionTreeClassify(train_data_std, train_label, criterion="gini")

train_pred = DecisionTree_Pred(model, train_data_std)
print("train dataset evaluation:")
print_evaluation(train_pred, train_label)

val_pred = DecisionTree_Pred(model, val_data_std)
print("val dataset evaluation:")
print_evaluation(val_pred, val_label)

test_pred = DecisionTree_Pred(model, test_data_std)
print("test dataset evaluation:")
print_evaluation(test_pred, test_label)

print(export_text(model))
joblib.dump(model, "DecisionTree.pkl")