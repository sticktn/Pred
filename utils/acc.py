def accuracy(pred, test_label):
    """
    Computational model accuracy
    :param pred: predictive classification
    :param test_label: true classification
    :return:Computational model accuracy
    """
    length = len(test_label)
    accuracy = (pred == test_label).sum() / length
    return accuracy
