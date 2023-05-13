from utils.read_data import read
from utils.LogisticRegression import LogisticRegression, LogisticRegression_pred
from utils.acc import accuracy

train_data_std, train_label, val_data_std, val_label, test_data_std, test_label = read()
model = LogisticRegression(train_data_std, train_label)  # logistic regression fit

train_pred = LogisticRegression_pred(model, train_data_std)
train_acc = accuracy(train_pred, train_label)
print("train dataset accuracy:", train_acc)

val_pred = LogisticRegression_pred(model, val_data_std)
val_acc = accuracy(val_pred, val_label)
print("val dataset accuracy:", val_acc)

test_pred = LogisticRegression_pred(model, test_data_std)
test_acc = accuracy(test_pred, test_label)
print("test dataset accuracy:", test_acc)
