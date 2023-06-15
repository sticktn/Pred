import sys
import joblib
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QGridLayout, QComboBox, QLabel, QMessageBox


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)

        self.input = QLineEdit()
        grid.addWidget(self.input, 1, 0)

        button = QPushButton('开始预测')
        button.clicked.connect(self.submit)
        grid.addWidget(button, 1, 2)

        # 添加标签
        label = QLabel('预测所用的模型:')
        grid.addWidget(label, 0, 0)

        # 创建QComboBox下拉选择框
        self.model_select = QComboBox()
        self.model_select.addItems(['MLP', 'DecisionTree', 'KNN'])
        grid.addWidget(self.model_select, 0, 2)

        label1 = QLabel("（八十五个特征由'/'分开）")
        grid.addWidget(label1, 2, 0)

        self.setWindowTitle('预测是否买保险')
        self.setGeometry(300, 300, 500, 150)
        self.show()

    def submit(self):

        input_test = self.input.text()
        model_choice = self.model_select.currentText()
        model=None
        try:
            if model_choice == 'MLP':
                model = joblib.load('./model/MLP.pkl')
            elif model_choice == 'DecisionTree':
                model = joblib.load('./model/DecisionTree.pkl')
            elif model_choice == 'KNN':
                model = joblib.load('./model/KNN.pkl')
            input = input_test.split('/')
            test = np.array(input,dtype=float).reshape(1, -1)
            result = model.predict(test,)
            message = None
            if result == 0:
                message = '此客户不买保险。'
            elif result == 1:
                message = '此客户买保险。'
            QMessageBox.information(self,'预测结果', message)
        except Exception as e:
            print(e)
        # print(model_choice)
        # print(input_test)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
