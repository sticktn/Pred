import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read():
    """
    read data
    :return train_data_std, train_label, val_data_std, val_label, test_data_std, test_label
    """
    data_all = pd.read_excel('./data.xlsx')  # original data
    test_all = pd.read_excel('./eval.xlsx')  # test data
    data = data_all.iloc[:, :85]
    label = data_all.iloc[:, 85]
    test_data = test_all.iloc[:, :85]
    test_label = test_all.iloc[:, 85]

    train_data, val_data, train_label, val_label = train_test_split(data, label, test_size=0.15, random_state=0)
    # Divide the data set into a training set and a validation set 85:15

    sc = StandardScaler()  # Normalize all datasets
    sc.fit(train_data)
    train_data_std = sc.transform(train_data)
    val_data_std = sc.transform(val_data)
    test_data_std = sc.transform(test_data)

    return train_data_std, train_label, val_data_std, val_label, test_data_std, test_label
