# BERT Chinese Text Classification

## 📌 项目简介

本项目基于 BERT 预训练模型，实现中文文本分类任务。

使用：

- PyTorch
- HuggingFace Transformers
- BERT-base-Chinese
- SwanLab


项目目标：

通过微调 BERT 模型，实现对中文新闻文本进行自动分类，并记录模型训练过程。



项目特点

- ✅ 基于预训练 BERT 模型
- ✅ 支持 GPU 加速训练
- ✅ 使用 HuggingFace Transformers
- ✅ SwanLab 实验可视化
- ✅ 保存最佳模型参数
- ✅ 支持训练集、验证集、测试集评估



项目结构

BERT-Text-Classification

│
├── train.py # 模型训练代码
│
├── best_model.pth # 最佳模型参数
│
├── README.md # 项目说明
│
└── images
└── swanlab.png # 实验结果截图


数据说明
数据格式：
news_id_!_label_code_!_label_name_!_title_!_keywords
| 字段         | 说明   |
| ---------- | ---- |
| news_id    | 新闻编号 |
| label_code | 类别编号 |
| label_name | 文本类别 |
| title      | 新闻标题 |
| keywords   | 关键词  |
示例：
1_!_100_!_财经_!_股票市场上涨_!_股票 金融
