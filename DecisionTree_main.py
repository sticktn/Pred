from utils.read_data import read
from utils.model import Model
from utils.acc import Evaluation

train_data_std, train_label, val_data_std, val_label, test_data_std, test_label = read(is_resample=False)
Model_de = Model(train_data_std,train_label,model_name='DecisionTree',tree_deepth=6)
pred = Model_de.pred(test_data_std)
de_evaluation = Evaluation(pred,test_label)
de_evaluation.print_evaluation()
de_evaluation.draw_confusion_matrix()
de_evaluation.draw_recall(Model_de.pred_proba(test_data_std))
Model_de.save_model('DecisionTree.pkl')
