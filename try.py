import pandas as pd
import torch
from torch.utils.data import Dataset

class ToutiaoDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        text = row["title"]
        label = row["label"]

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



#尝试ing
columns = [
    "news_id",
    "label_code",
    "label_name",
    "title",
    "keywords"
]

train_df = pd.read_csv(
    "D:/data/0.demo1文本分类/train_3k.txt",
    sep="_!_",
    names=columns,
    engine="python"
)

dev_df = pd.read_csv(
    "D:/data/0.demo1文本分类/dev_1k.txt",
    sep="_!_",
    names=columns,
    engine="python"
)

test_df = pd.read_csv(
    "D:/data/0.demo1文本分类/test_1k.txt",
    sep="_!_",
    names=columns,
    engine="python"
)

print("训练集：", train_df.shape)
print("验证集：", dev_df.shape)
print("测试集：", test_df.shape)

print("\n类别：")
print(train_df["label_name"].value_counts())

print("\n训练集前5条：")
print(train_df[["label_name", "title"]].head())

# 获取所有类别
labels = sorted(train_df["label_name"].unique())

# 建立映射
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

print("类别数量：", len(labels))
print("label2id：", label2id)
print("id2label：", id2label)

train_df["label"] = train_df["label_name"].map(label2id)
dev_df["label"] = dev_df["label_name"].map(label2id)
test_df["label"] = test_df["label_name"].map(label2id)
print(train_df[["label_name", "label", "title"]].head())

from transformers import AutoTokenizer

model_name = "google-bert/bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = train_df["title"].iloc[0]

encoded = tokenizer(
    text,
    max_length=128,
    padding="max_length",
    truncation=True
)

print("原始文本：")
print(text)

print("\ninput_ids：")
print(encoded["input_ids"])

print("\nattention_mask：")
print(encoded["attention_mask"])