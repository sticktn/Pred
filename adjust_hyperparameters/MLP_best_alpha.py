"""
choose the best lr in MLP
"""
from utils.model import Model
from utils.acc import Evaluation
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

data_all = pd.read_excel('../data.xlsx')  # original data
test_all = pd.read_excel('../eval.xlsx')  # test data
data = data_all.iloc[:, :85]
label = data_all.iloc[:, 85]
test_data = test_all.iloc[:, :85]
test_label = test_all.iloc[:, 85]

# from sklearn.utils import resample, shuffle
# train_up = data_all[data_all['移动房车险数量'] == 1]
# train_down = data_all[data_all['移动房车险数量'] == 0]
# train_up = resample(train_up, n_samples=696, random_state=0)
# train_down = resample(train_down, n_samples=1095, random_state=0)
# train = shuffle(pd.concat([train_up, train_down]))
# data = train.iloc[:, :85]
# label = train.iloc[:, 85]

train_data = data.values
train_label = label.values
test_data = test_data.values
test_label = test_label.values


scaler = StandardScaler()
scaler.fit(train_data)

train_data_std = scaler.transform(train_data)
test_data_std = scaler.transform(test_data)

accuracy = []
precision = []
recall = []
F1 = []
alpha = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
for i in alpha:
    decision_tree = Model(train_data_std, train_label, model_name='MLP',alpha=i)
    test_pred = decision_tree.pred(test_data_std)
    eva = Evaluation(test_pred, test_label)
    accuracy.append(eva.accuracy)
    precision.append(eva.precision)
    recall.append(eva.recall)
    F1.append(eva.F1)

plt.plot(alpha,accuracy,label="accuracy",marker='o')
plt.plot(alpha,precision,label="precision")
plt.plot(alpha,recall,label="recall")
plt.plot(alpha,F1,label="F1")
plt.legend()
plt.show()

