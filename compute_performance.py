import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from config import root_path
from train import pred_to_category


class ConfusionMatrix(object):

    def __init__(self, y_true, y_pred):
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0
        for y, p in zip(y_true, y_pred):
            if y == 1 and p == 1:
                self.tp = self.tp + 1
            elif y == 1 and p == 0:
                self.fn = self.fn + 1
            elif y == 0 and p == 1:
                self.fp = self.fp + 1
            else:
                self.tn = self.tn + 1

    def sensitivity(self):
        return self.tp / (self.tp + self.fn)

    def specificity(self):
        return self.tn / (self.fp + self.tn)

    def fpr(self):
        return self.fp / (self.fp + self.tn)

    def tpr(self):
        return self.tp / (self.tp + self.fn)


def compute_fprs_tprs(y_true, y_pred, num_threshold=100):
    fprs = np.zeros(num_threshold, dtype='float32')
    tprs = np.zeros(num_threshold, dtype='float32')
    for i, thre in enumerate(np.linspace(min(y_pred), max(y_pred), num_threshold)):
        cm = ConfusionMatrix(y_true, (y_pred <= thre).astype('int'))
        fprs[i] = cm.fpr()
        tprs[i] = cm.tpr()

    return fprs, tprs


def compute_auc(fprs, tprs):
    auc = 0
    num_rectangle = len(fprs) - 1
    for i in range(num_rectangle):
        upper = tprs[i]
        lower = tprs[i + 1]
        height = fprs[i + 1] - fprs[i]
        auc = auc + (upper + lower) * height / 2

    return auc


def eval_model():
    # 载入测试集数据
    test_set = pd.read_csv(os.path.join(root_path, 'dataset/test.csv'))
    x_test, y_test = test_set.drop(['COG'], axis=1).to_numpy(), test_set['COG'].values

    # 初始化
    result = {
        'epoch': [],
        'sensitivity': [],
        'specificity': [],
        'auc': [],
        'accuracy': []
    }

    # 遍历模型
    checkpoint_dir = os.path.join(root_path, 'checkpoints')
    filenames = os.listdir(checkpoint_dir)
    for fn in filenames:
        # 载入模型
        model = CatBoostRegressor()
        model.load_model(os.path.join(checkpoint_dir, fn))

        # 计算指标
        y_pred = model.predict(x_test)
        pred_labels = pred_to_category(y_pred)
        sensitivity, specificity, auc = 0, 0, 0
        for i in range(3):
            category_true = (y_test == i).astype('int')
            cm = ConfusionMatrix(
                y_true=category_true,
                y_pred=(pred_labels == i).astype('int')
            )
            sensitivity = sensitivity + cm.sensitivity()
            specificity = specificity + cm.specificity()
            auc = auc + compute_auc(*compute_fprs_tprs(category_true, np.abs(y_pred - i)))
        accuracy = sum(y_test == pred_labels) / y_test.shape[0]
        result['epoch'].append(int(fn.rsplit('.', 1)[0]))
        result['sensitivity'].append(sensitivity / 3)
        result['specificity'].append(specificity / 3)
        result['auc'].append(auc / 3)
        result['accuracy'].append(accuracy)

    # 保存指标
    result = pd.DataFrame(result).sort_values(by='epoch')
    result.to_csv(os.path.join(root_path, 'eval_result/test_performance.csv'), index=False)


if __name__ == '__main__':
    eval_model()
