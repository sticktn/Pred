"""
choose the best deepth of decision tree
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
deepth = range(5,31,5)
for i in deepth:
    decision_tree = Model(train_data_std, train_label, model_name="DecisionTree",tree_deepth=i)
    test_pred = decision_tree.pred(test_data_std)
    eva = Evaluation(test_pred, test_label)
    accuracy.append(eva.accuracy)
    precision.append(eva.precision)
    recall.append(eva.recall)
    F1.append(eva.F1)

plt.plot(deepth,accuracy,label="accuracy")
plt.plot(deepth,precision,label="precision")
plt.plot(deepth,recall,label="recall")
plt.plot(deepth,F1,label="F1")
plt.legend()
plt.show()

