import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

import swanlab

# 1. 基本配置
# 数据路径
TRAIN_PATH = "D:/data/0.demo1文本分类/train_3k.txt"
DEV_PATH = "D:/data/0.demo1文本分类/dev_1k.txt"
TEST_PATH = "D:/data/0.demo1文本分类/test_1k.txt"

# BERT模型
MODEL_NAME = "google-bert/bert-base-chinese"

# 超参数
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 10
DROPOUT = 0.1

# 随机种子
SEED = 42

# 2. 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# 4. 读取数据
columns = [
    "news_id",
    "label_code",
    "label_name",
    "title",
    "keywords"
]

print("\n" + "=" * 60)
print("读取数据")
print("=" * 60)

train_df = pd.read_csv(
    TRAIN_PATH,
    sep="_!_",
    names=columns,
    engine="python"
)

dev_df = pd.read_csv(
    DEV_PATH,
    sep="_!_",
    names=columns,
    engine="python"
)

test_df = pd.read_csv(
    TEST_PATH,
    sep="_!_",
    names=columns,
    engine="python"
)

print("训练集:", train_df.shape)
print("验证集:", dev_df.shape)
print("测试集:", test_df.shape)

# 5. 标签编码
labels = sorted(train_df["label_name"].unique())

label2id = {
    label: i
    for i, label in enumerate(labels)
}

id2label = {
    i: label
    for label, i in label2id.items()
}

num_labels = len(labels)

print("\n" + "=" * 60)
print("类别信息")
print("=" * 60)

print("类别数量:", num_labels)
print("label2id:", label2id)
print("id2label:", id2label)


# 把文字类别转换成数字
train_df["label"] = train_df["label_name"].map(label2id)
dev_df["label"] = dev_df["label_name"].map(label2id)
test_df["label"] = test_df["label_name"].map(label2id)


# 检查有没有无法转换的标签
if train_df["label"].isna().any():
    raise ValueError("训练集中存在无法映射的标签！")

if dev_df["label"].isna().any():
    raise ValueError("验证集中存在无法映射的标签！")

if test_df["label"].isna().any():
    raise ValueError("测试集中存在无法映射的标签！")

# 6. 加载Tokenizer
print("\n" + "=" * 60)
print("加载Tokenizer")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer加载完成")

# 7. Dataset
class ToutiaoDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length=128):

        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        # 新闻标题
        text = str(row["title"])

        # 新闻类别
        label = int(row["label"])

        # Tokenizer
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# 8. 创建Dataset
train_dataset = ToutiaoDataset(
    train_df,
    tokenizer,
    MAX_LENGTH
)

dev_dataset = ToutiaoDataset(
    dev_df,
    tokenizer,
    MAX_LENGTH
)

test_dataset = ToutiaoDataset(
    test_df,
    tokenizer,
    MAX_LENGTH
)

print("\n" + "=" * 60)
print("Dataset信息")
print("=" * 60)

print("训练集:", len(train_dataset))
print("验证集:", len(dev_dataset))
print("测试集:", len(test_dataset))


# 9. 创建DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\n" + "=" * 60)
print("DataLoader信息")
print("=" * 60)

print("Batch Size:", BATCH_SIZE)
print("训练Batch数量:", len(train_loader))
print("验证Batch数量:", len(dev_loader))
print("测试Batch数量:", len(test_loader))


# 10. 测试一个Batch
batch = next(iter(train_loader))

print("\n" + "=" * 60)
print("Batch测试")
print("=" * 60)

print("input_ids:", batch["input_ids"].shape)
print("attention_mask:", batch["attention_mask"].shape)
print("labels:", batch["labels"].shape)

# 11. 加载BERT分类模型

print("\n" + "=" * 60)
print("加载BERT模型")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=DROPOUT,
    attention_probs_dropout_prob=DROPOUT
)

# 把模型放到GPU/CPU
model = model.to(device)

print("模型加载完成")
print("模型设备:", device)

# 12. 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# 13. SwanLab实验记录
swanlab.init(
    project="toutiao-bert-classification",
    experiment_name="bert-base-chinese-baseline",
    config={
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "dropout": DROPOUT,
        "num_labels": num_labels,
        "seed": SEED
    }
)


# 14. 训练一个Epoch

def train_one_epoch(model, dataloader, optimizer, device):

    model.train()

    total_loss = 0

    all_preds = []
    all_labels = []

    for batch in dataloader:

        # 把数据放到GPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 记录Loss
        total_loss += loss.item()

        # 获取预测结果
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(
            preds.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    avg_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    return avg_loss, accuracy


# 15. 验证
def evaluate(model, dataloader, device):

    model.eval()

    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(
                logits,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    avg_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    return avg_loss, accuracy


# 16. 开始训练
print("\n" + "=" * 60)
print("开始训练")
print("=" * 60)

best_dev_accuracy = 0.0

for epoch in range(EPOCHS):

    print("\n")
    print("-" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 60)

    # 训练
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device
    )


    # 验证
    dev_loss, dev_accuracy = evaluate(
        model,
        dev_loader,
        device
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Dev Loss: {dev_loss:.4f}"
    )

    print(
        f"Dev Accuracy: {dev_accuracy:.4f}"
    )

    # SwanLab记录
    swanlab.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "dev_loss": dev_loss,
        "dev_accuracy": dev_accuracy
    })

    # 保存验证集最好的模型
    if dev_accuracy > best_dev_accuracy:

        best_dev_accuracy = dev_accuracy

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print(
            f"保存最佳模型，Dev Accuracy = {dev_accuracy:.4f}"
        )


# 17. 加载最佳模型
print("\n" + "=" * 60)
print("加载最佳模型")
print("=" * 60)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

print(
    f"最佳验证集准确率: {best_dev_accuracy:.4f}"
)


# 18. 测试集评估
print("\n" + "=" * 60)
print("测试集评估")
print("=" * 60)

test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    device
)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)


# 19. 生成详细分类报告
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        preds = torch.argmax(
            logits,
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


print("\n" + "=" * 60)
print("Classification Report")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=labels,
        digits=4,
        zero_division=0
    )
)


# 20. SwanLab记录最终测试结果
swanlab.log({
    "best_dev_accuracy": best_dev_accuracy,
    "test_loss": test_loss,
    "test_accuracy": test_accuracy
})


print("\n" + "=" * 60)
print("实验完成")
print("=" * 60)

print("最佳验证集准确率:", best_dev_accuracy)
print("最终测试集准确率:", test_accuracy)