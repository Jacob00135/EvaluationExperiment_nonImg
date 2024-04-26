import os
import numpy as np
import pandas as pd
from config import root_path
from catboost import CatBoostRegressor


def pred_to_category(pred, thresholds=(0.5, 1.5)):
    category = np.zeros(pred.shape[0], dtype='int')
    for i, p in enumerate(pred):
        if p >= thresholds[1]:
            category[i] = 2
        elif p >= thresholds[0]:
            category[i] = 1

    return category


def main():
    # 载入数据
    train_set = pd.read_csv(os.path.join(root_path, 'dataset/train.csv'))
    test_set = pd.read_csv(os.path.join(root_path, 'dataset/test.csv'))
    x_train, y_train = train_set.drop(['COG'], axis=1).to_numpy(), train_set['COG'].values
    x_test, y_test = test_set.drop(['COG'], axis=1).to_numpy(), test_set['COG'].values

    # 载入模型
    learning_rate = 0.01
    model = CatBoostRegressor(iterations=5, learning_rate=learning_rate)
    checkpoint_dir = os.path.join(root_path, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # 训练
    train_history = {
        'epoch': [],
        'train_accuracy': [],
        'test_accuracy': []
    }
    train_history_path = os.path.join(root_path, 'eval_result/train_history.csv')
    num_epoch = 100
    for epoch in range(1, num_epoch + 1):
        # 训练、保存模型
        if epoch != 1:
            init_model = model
        else:
            init_model = None
        model.fit(x_train, y_train, init_model=init_model, verbose=False)
        model.save_model(os.path.join(checkpoint_dir, '{}.model'.format(epoch)))

        # 验证模型
        pred_train = pred_to_category(model.predict(x_train))
        train_accuracy = sum(pred_train == y_train) / y_train.shape[0]
        pred_test = pred_to_category(model.predict(x_test))
        test_accuracy = sum(pred_test == y_test) / y_test.shape[0]

        # 保存训练历史
        train_history['epoch'].append(epoch)
        train_history['train_accuracy'].append(train_accuracy)
        train_history['test_accuracy'].append(test_accuracy)
        pd.DataFrame(train_history).to_csv(train_history_path, index=False)

        # 将训练信息输出到控制台
        print('Epoch {}: train_accuracy={:.4f} -- test_accuracy={:.4f}'.format(
            epoch, train_accuracy, test_accuracy
        ))


if __name__ == '__main__':
    main()
