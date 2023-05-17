# -*- coding: UTF-8 -*-
"""
@Project ：Pred \n 
@File    ：MLP_main.py \n
@Author  ：guo \n
@Date    ：2023/5/17 下午4:45 \n
"""
from utils.read_data import read
from utils.MLP import MLPCLassify, MLP_Pred
from utils.acc import accuracy

train_data_std, train_label, val_data_std, val_label, test_data_std, test_label = read()
model = MLPCLassify(train_data_std, train_label)

train_pred = MLP_Pred(model, train_data_std)
train_acc = accuracy(train_pred, train_label)
print("train dataset accuracy:", train_acc)

val_pred = MLP_Pred(model, val_data_std)
val_acc = accuracy(val_pred, val_label)
print("val dataset accuracy:", val_acc)

test_pred = MLP_Pred(model, test_data_std)
test_acc = accuracy(test_pred, test_label)
print("test dataset accuracy:", test_acc)
