import numpy as np
from sklearn import linear_model


def LogisticRegression(train_data, train_label):
    """
    use logistic regression to fit data
    :param train_data:  train_Data
    :param train_label: train_Label
    :return: model
    """
    logreg = linear_model.LogisticRegression(max_iter=500)
    logreg.fit(train_data, train_label)
    return logreg


def LogisticRegression_pred(model, test_data):
    """
    predictive classification
    :param model: LogisticRegression model
    :param test_data: test_Date
    :return: Predictive value
    """
    pred = model.predict_proba(test_data)
    return np.argmax(pred, axis=1)


