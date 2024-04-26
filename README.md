[TOC]

本项目使用CatBoost回归算法，对阿尔茨海默症受试者进行分类，数据主要是受试者的人口统计学特征以及部分量表测试分数，将受试者分成三类：正常人（CN）、轻度认知障碍（MCI）、阿尔茨海默症患者（AD）。

### 1.配置Python环境

首先安装Python3.9，并使用以下命令安装项目运行所需的三方库：

```bash
pip install matplotlib==3.8.4
pip install pandas==2.2.2
pip install catboost==1.2.3
```

### 2.创建项目必需的目录

在项目根路径下创建以下目录：

```bash
checkpoints
eval_result
```

### 3.数据预处理

首先，要对项目的数据进行预处理，主要是将数据`dataset/data.csv`分割成训练集`train.csv`和测试集`test.csv`，对于数据中的每一个类（CN、MCI、AD）都按照8:2的比例划分。

在终端执行以下命令来进行数据预处理：

```bash
python data_preprocess.py
```

运行完毕后，`dataset`目录下将会生成两个文件：`train.csv`和`test.csv`。

### 4.训练模型

训练模型的代码写在`train.py`中，默认设置的训练相关变量如下，可以在`main`函数里面更改：

|变量名|含义|默认值|
|:-:|:-:|:-:|
|learning_rate|学习率|0.01|
|num_epoch|训练轮次|100|

在训练的过程中，对于每一个训练轮次（epoch），训练后将会做如下操作：
1. 保存模型的文件到目录`checkpoints`下，命名为`<epoch>.model`，用于后续指标计算等
2. 计算本轮次训练后的模型在训练集、测试集的上的准确率，并输出到控制台，用于监控模型训练情况
3. 将第2步计算的准确率保存为csv文件，保存在`eval_result/train_history.csv`中，当所有训练轮次完毕后，将使用此文件来画折线图，判断模型是否过拟合

在终端执行以下命令来训练模型：

```bash
python train.py
```

### 5.绘制训练时准确率折线图

本步将使用`eval_result/train_history.csv`来绘制折线图，在图中展示模型在训练过程中计算的训练集、测试集的准确率，x轴为训练轮次（epoch），y轴为准确率（Accuracy），可以判断模型是否过拟合。

本项目所有绘图的代码都在`draw_graph.py`中。要绘制本步所需折线图，需要调用代码中的函数`training_accuracy`，所以首先需要编辑代码文件`draw_graph.py`，编辑后如下所示：

```python
if __name__ == '__main__':
    training_accuracy()
```

然后在终端执行以下命令绘图：

```
python draw_graph.py
```

代码执行完毕后，会把绘制好的折线图保存在：`eval_result/images/training_accuracy.png`

### 6.计算评估指标

为了更准确地评估模型的性能，需计算指标`Sensitivity`、`Specificity`、`Accuracy`、`AUC`，用于后续绘制模型评估曲线图。

***需要注意的是，由于此项目是三分类任务，而Sensitivity、Specificity、AUC指标通常只能用于二分类，所以在计算它们时，使用宏平均（Macro-average）处理方法：计算每个类别的二分类指标，再取平均值***

计算指标的代码为`compute_performance.py`。首先读取checkpoints中所有的模型，并对于每一个模型，在测试集上进行预测得到预测分数，然后使用预测分数和测试集的真实标签来计算指标，计算好的指标将保存在`eval_result/test_performance.csv`。

在终端执行以下命令来计算评估指标：

```
python compute_performance.py
```

### 7.绘制模型评估曲线图

本步使用前置步骤计算好的指标来绘制折线图，在折线图中展示出训练过程中的模型在测试集上的Sensitivity、Specificity、Accuracy、AUC，并根据AUC选出最优的模型，再绘制最优模型的ROC曲线图。

首先需要编辑代码文件`draw_graph.py`，编辑后如下所示：

```python
if __name__ == '__main__':
    test_performance()
    best_model()
```

然后在终端执行以下命令绘图：

```
python draw_graph.py
```

代码执行完毕后，会把绘制好的两个折线图保存在：
- `eval_result/images/test_performance.png`
- `eval_result/images/best_model.png`