from utils.read_data import read
from utils.LogisticRegression import LogisticRegression, LogisticRegression_pred
from utils.acc import accuracy

train_data_std, train_label, val_data_std, val_label, test_data_std, test_label = read()
model = LogisticRegression(train_data_std, train_label)
pred = LogisticRegression_pred(model, test_data_std)
acc = accuracy(pred, test_label)
print(acc)
