import os
import numpy as np
import pandas as pd
from collections import Counter
from config import root_path

if __name__ == '__main__':
    # 读取数据
    data_path = os.path.join(root_path, 'dataset/data.csv')
    data = pd.read_csv(data_path)

    # 打乱样本次序
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    data = data.iloc[index, :]

    y = data['COG'].values
    counter = Counter(y)
    train_boolean = np.zeros(y.shape[0], dtype='bool')

    # 划分CN类
    num_train_cn = int(0.8 * counter[0])
    train_cn_index = np.where(y == 0)[0][:num_train_cn]
    train_boolean[train_cn_index] = True

    # 划分MCI类
    num_train_mci = int(0.8 * counter[1])
    train_mci_index = np.where(y == 1)[0][:num_train_mci]
    train_boolean[train_mci_index] = True

    # 划分AD类
    num_train_ad = int(0.8 * counter[2])
    train_ad_index = np.where(y == 2)[0][:num_train_ad]
    train_boolean[train_ad_index] = True

    # 划分出两个子集
    train_set = data[train_boolean]
    test_set = data[~train_boolean]

    # 保存
    train_set.to_csv(os.path.join(root_path, 'dataset/train.csv'), index=False)
    test_set.to_csv(os.path.join(root_path, 'dataset/test.csv'), index=False)
