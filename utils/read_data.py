import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read(is_resample):
    """
    read data
    :return train_data_std, train_label, val_data_std, val_label, test_data_std, test_label
    """
    data_all = pd.read_excel('./data.xlsx')  # original data
    test_all = pd.read_excel('./eval.xlsx')  # test data
    data = data_all.iloc[:, :85]
    label = data_all.iloc[:, 85]


    if is_resample:
        from sklearn.utils import resample, shuffle
        train_up = data_all[data_all['移动房车险数量'] == 1]
        train_down = data_all[data_all['移动房车险数量'] == 0]
        train_up = resample(train_up, n_samples=696, random_state=0)
        train_down = resample(train_down, n_samples=1095, random_state=0)
        train = shuffle(pd.concat([train_up, train_down]))
        train_data = train.iloc[:, :85]
        train_label = train.iloc[:, 85]
    else:
        train_data = data_all.iloc[:, :85]
        train_label = data_all.iloc[:, 85]

    test_data = test_all.iloc[:, :85]
    test_label = test_all.iloc[:, 85]

    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.15, random_state=0)
    # Divide the data set into a training set and a validation set 85:15

    sc = StandardScaler()  # Normalize all datasets
    sc.fit(train_data)
    train_data_std = sc.transform(train_data)
    val_data_std = sc.transform(val_data)
    test_data_std = sc.transform(test_data)

    return train_data_std, train_label, val_data_std, val_label, test_data_std, test_label
