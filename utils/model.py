class Model:
    def __init__(self, train_data, train_label, model_name=None,tree_deepth=None,alpha=None,n_neighbors=None):
        """
        :param train_data: Train data
        :param train_label: Train label
        """
        self.train_data = train_data
        self.train_label = train_label
        self.model = None
        if model_name == "DecisionTree":
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier(criterion='gini',class_weight='balanced', max_depth=tree_deepth)
        elif model_name == "MLP":
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(100, 50), random_state=1)
        elif model_name == "KNN":
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(train_data, train_label)

    def pred(self, test_data):
        return self.model.predict(test_data)

    def pred_proba(self, test_data):
        return self.model.predict_proba(test_data).max(axis=1)

    def save_model(self, model_path):
        """
        save model
        :param model_path:
        :return: None
        """
        import joblib
        joblib.dump(self.model, model_path)