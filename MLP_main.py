# -*- coding: UTF-8 -*-
"""
@Project ：Pred \n 
@File    ：MLP_main.py \n
@Author  ：guo \n
@Date    ：2023/5/17 下午4:45 \n
"""
from utils.read_data import read
from utils.MLP import MLPCLassify, MLP_Pred
from utils.acc import print_evaluation
import joblib

train_data_std, train_label, val_data_std, val_label, test_data_std, test_label = read()
model = MLPCLassify(train_data_std, train_label)

train_pred = MLP_Pred(model, train_data_std)
print("train dataset evaluation:")
print_evaluation(train_pred, train_label)

val_pred = MLP_Pred(model, val_data_std)
print("val dataset evaluation:")
print_evaluation(val_pred, val_label)

test_pred = MLP_Pred(model, test_data_std)
print("test dataset evaluation:")
print_evaluation(test_pred, test_label)

joblib.dump(model, "MLP.pkl")