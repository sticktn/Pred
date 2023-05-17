# -*- coding: UTF-8 -*-
"""
@Project ：Pred \n 
@File    ：KNN_main.py \n
@Author  ：guo \n
@Date    ：2023/5/17 下午5:12 \n
"""
from utils.read_data import read
from utils.KNN import KNNClassify, KNN_Pred
from utils.acc import accuracy
import matplotlib.pyplot as plt

train_data_std, train_label, val_data_std, val_label, test_data_std, test_label = read()
model = KNNClassify(train_data_std, train_label,n=8)

train_pred = KNN_Pred(model, train_data_std)
train_acc = accuracy(train_pred, train_label)
print("train dataset accuracy:", train_acc)

val_pred = KNN_Pred(model, val_data_std)
val_acc = accuracy(val_pred, val_label)
print("val dataset accuracy:", val_acc)

test_pred = KNN_Pred(model, test_data_std)
test_acc = accuracy(test_pred, test_label)
print("test dataset accuracy:", test_acc)

# 判断knn中的k值
# acc = []
# for i in range(1, 21):
#     model = KNNClassify(train_data_std, train_label, n=i)
#     pred = KNN_Pred(model, test_data_std)
#     acc.append(accuracy(pred, test_label))
#
# plt.xlabel("K")
# plt.ylabel("Accuracy")
# plt.plot(range(1, 21), acc)
#
# ax = plt.gca()
# ax.grid()
# plt.show()

