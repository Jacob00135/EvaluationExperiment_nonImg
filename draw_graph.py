import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from config import root_path, category_list
from compute_performance import compute_fprs_tprs, compute_auc

images_path = os.path.join(root_path, 'eval_result/images')
if not os.path.exists(images_path):
    os.mkdir(images_path)


def training_accuracy(show=False):
    # 载入数据
    train_history_path = os.path.join(root_path, 'eval_result/train_history.csv')
    train_history = pd.read_csv(train_history_path)
    x = train_history['epoch'].values
    y1 = train_history['train_accuracy'].values
    y2 = train_history['test_accuracy'].values

    # 画图
    plt.figure()
    plt.plot(x, y1, label='train_acc')
    plt.plot(x, y2, label='test_acc')
    plt.legend()
    plt.savefig(os.path.join(root_path, 'eval_result/images/training_accuracy.png'))
    if show:
        plt.show()


def test_performance(show=False):
    # 载入数据
    data = pd.read_csv(os.path.join(root_path, 'eval_result/test_performance.csv'))
    x = data['epoch'].values
    yticks = np.around(np.arange(0.0, 1.1, 0.1), 1)

    # 绘图
    plt.figure(figsize=(5, 4), dpi=300)
    for y_name in ['sensitivity', 'specificity', 'auc', 'accuracy']:
        y = data[y_name].values
        plt.plot(x, y, label=y_name)
        # plt.scatter(x, y, s=10, lw=0.2, ec='#000000', zorder=2)
    plt.xlim(0, 100)
    plt.ylim(0.0, 1.0)
    plt.xticks(range(0, 101, 10), range(0, 101, 10))
    plt.yticks(yticks, yticks)
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.title('test performance')
    plt.legend(loc='lower right')
    plt.grid(True, c='#eeeeee', ls='--', zorder=0)
    plt.savefig(os.path.join(images_path, 'test_performance.png'))
    if show:
        plt.show()


def best_model(show=False):
    # 选出最好的模型
    data = pd.read_csv(os.path.join(root_path, 'eval_result/test_performance.csv'))
    best_index = data['auc'].argmax()
    row = data.iloc[best_index, :]
    model_path = os.path.join(root_path, 'checkpoints/{}.model'.format(int(row['epoch'])))
    model = CatBoostRegressor()
    model.load_model(model_path)

    # 计算ROC数据
    test_set = pd.read_csv(os.path.join(root_path, 'dataset/test.csv'))
    x_test, y_test = test_set.drop(['COG'], axis=1).to_numpy(), test_set['COG'].values
    y_pred = model.predict(x_test)
    fprs_list, tprs_list, auc_list = [], [], []
    for i in range(3):
        fprs, tprs = compute_fprs_tprs(
            y_true=(y_test == i).astype('int'),
            y_pred=np.abs(y_pred - i),
            num_threshold=1000
        )
        auc = compute_auc(fprs, tprs)
        fprs_list.append(fprs)
        tprs_list.append(tprs)
        auc_list.append(auc)

    # 绘图
    ticks = np.around(np.arange(0.0, 1.1, 0.1), 1)
    plt.figure(figsize=(5, 4), dpi=300)
    for category, fprs, tprs in zip(category_list, fprs_list, tprs_list):
        plt.plot(fprs, tprs, label=category)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('best model ROC')
    plt.grid(True, c='#eeeeee', ls='--', zorder=0)
    legend_text_list = []
    for category, auc in zip(category_list, auc_list):
        legend_text_list.append('{}_AUC = {:.4f}'.format(category, auc))
    plt.legend(legend_text_list, loc='lower right')
    plt.savefig(os.path.join(images_path, 'best_model.png'))
    if show:
        plt.show()


if __name__ == '__main__':
    # training_accuracy()
    # test_performance()
    best_model()
