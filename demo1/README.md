# BERT 中文文本分类项目


## 📂 项目结构

```
├── README.md # 项目说明文档
├── requirements.txt # 项目依赖环境
│
├── config.py # 参数配置文件
│ # 包含数据路径、模型名称、训练参数等
│
├── dataset.py # 数据处理模块
│ # 包含数据读取、标签编码、Dataset构建
│
├── model.py # 模型定义模块
│ # 负责加载BERT文本分类模型
│
├── train.py # 模型训练入口
│ # 包含训练流程、验证、模型保存
│
├── metrics.py # 模型评估模块
│ # 用于测试模型性能
│
├── utils.py # 工具函数
│ # 包含随机种子设置等辅助功能
│
└── checkpoints/
└── best_model.pth # 保存验证集效果最佳的模型参数
```


## 📄 文件说明

### demo1.py

包含完整模型训练流程：

- 数据集读取
- 数据预处理
- Tokenizer 分词处理
- BERT 模型构建
- 模型训练
- 验证集评估
- 测试集评估



# 📊 数据集信息

本项目使用中文新闻文本分类数据集。


| 数据集 | 数量 |
| :---: | :---: |
| 训练集 | 3000 条 |
| 验证集 | 1000 条 |
| 测试集 | 1064 条 |


分类类别数量：

```
15 类
```


## 类别信息


| 类别ID | 类别名称 |
| :---: | :---: |
| 100 | 新闻故事 |
| 101 | 新闻文化 |
| 102 | 新闻娱乐 |
| 103 | 新闻体育 |
| 104 | 新闻财经 |
| 106 | 新闻房产 |
| 107 | 新闻汽车 |
| 108 | 新闻教育 |
| 109 | 新闻科技 |
| 110 | 新闻军事 |
| 112 | 新闻旅游 |
| 113 | 新闻国际 |
| 114 | 股票 |
| 115 | 新闻农业 |
| 116 | 新闻游戏 |



# ⚙️ 实验环境

训练设备：

- GPU


依赖环境：

```bash
pip install torch transformers tqdm numpy swanlab
```



# 🔧 实验超参数设置


本实验固定：

- max_len = 128
- batch_size = 16

通过调整 epoch 和 learning rate 进行对比实验。


| 实验组 | max_len | batch_size | epoch | learning_rate | Dev Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 第一组 | 128 | 16 | 3 | 2e-5 | 0.812 |
| 第二组 | 128 | 16 | 10 | 2e-5 | 0.822 |
| 第三组 | 128 | 16 | 10 | 1e-5 | 0.834 |



# 📈 实验结果分析


通过三组实验对比，研究不同训练参数对 BERT 文本分类模型性能的影响。


## 1. epoch 对模型性能的影响

在保持 learning rate=2e-5 不变的情况下，将训练轮数从 3 增加到 10。

实验结果：

- epoch=3 时，Dev Accuracy 为 0.812
- epoch=10 时，Dev Accuracy 提升至 0.822


说明增加训练轮数能够使模型更加充分地学习文本语义特征，提高分类性能。


## 2. learning rate 对模型性能的影响

在 epoch=10 的条件下，将学习率从 2e-5 调整为 1e-5。

实验结果：

- learning rate=2e-5 时，Dev Accuracy 为 0.822
- learning rate=1e-5 时，Dev Accuracy 提升至 0.834


说明较小的学习率能够使 BERT 微调过程更加稳定，使模型参数更新更加平缓，从而提升模型泛化能力。



# ⭐ 最优实验配置


综合三组实验结果，第三组实验取得最佳效果。


最终模型参数：


```
max_len: 128

batch_size: 16

epoch: 10

learning_rate: 1e-5
```


最佳验证集准确率：

```
Dev Accuracy = 0.834
```



# 📊 实验可视化


使用 SwanLab 对训练过程进行可视化，包括：

- Train Loss
- Train Accuracy
- Dev Loss
- Dev Accuracy


实验结果如下：


<img width="1122" height="670" alt="image" src="https://github.com/user-attachments/assets/5a177def-8151-46ca-b414-8b144d374b7f" />



# 📝 现象总结


实验结果表明，BERT 模型性能受到训练轮数和学习率等超参数影响。

增加 epoch 可以提高模型对文本语义特征的学习程度，使验证集准确率得到提升；降低 learning rate 可以增强模型微调过程的稳定性，使模型获得更好的泛化能力。


通过调整超参数，模型 Dev Accuracy 从初始实验的 0.812 提升至 0.834。

最终实验中：

```
epoch=10

learning_rate=1e-5
```

的参数组合取得最佳分类效果。
