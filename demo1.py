import random
import numpy as np
import pandas as pd
import torch
import swanlab

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

#参数配置
TRAIN_PATH = "D:/data/0.demo1文本分类/train_3k.txt"
DEV_PATH = "D:/data/0.demo1文本分类/dev_1k.txt"
TEST_PATH = "D:/data/0.demo1文本分类/test_1k.txt"

MODEL_NAME = "google-bert/bert-base-chinese"


MAX_LENGTH = 128
BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 10


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# 随机种子
def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything()



# 读取数据
columns = [
    "news_id",
    "label_code",
    "label_name",
    "title",
    "keywords"
]


def load_data(path):

    return pd.read_csv(
        path,
        sep="_!_",
        names=columns,
        engine="python"
    )


train_df = load_data(TRAIN_PATH)
dev_df = load_data(DEV_PATH)
test_df = load_data(TEST_PATH)



# =========================
# 标签编码
# =========================

labels = sorted(
    train_df["label_name"].unique()
)


label2id = {
    label:i
    for i,label in enumerate(labels)
}


id2label = {
    i:label
    for label,i in label2id.items()
}


for df in [
    train_df,
    dev_df,
    test_df
]:

    df["label"] = (
        df["label_name"]
        .map(label2id)
    )


num_labels = len(labels)



# Dataset
class NewsDataset(Dataset):

    def __init__(
        self,
        df,
        tokenizer
    ):

        self.texts = df["title"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer


    def __len__(self):

        return len(self.labels)


    def __getitem__(self,index):

        text = str(
            self.texts[index]
        )

        label = self.labels[index]


        encode = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )


        return {

            "input_ids":
            encode["input_ids"].squeeze(0),

            "attention_mask":
            encode["attention_mask"].squeeze(0),

            "labels":
            torch.tensor(
                label,
                dtype=torch.long
            )
        }



# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)



train_loader = DataLoader(
    NewsDataset(train_df,tokenizer),
    batch_size=BATCH_SIZE,
    shuffle=True
)


dev_loader = DataLoader(
    NewsDataset(dev_df,tokenizer),
    batch_size=BATCH_SIZE
)


test_loader = DataLoader(
    NewsDataset(test_df,tokenizer),
    batch_size=BATCH_SIZE
)



# 模型
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)


model.to(device)



optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)



# SwanLab
swanlab.init(

    project="bert-text-classification",

    experiment_name="bert-base-chinese",

    config={

        "model":MODEL_NAME,

        "batch_size":BATCH_SIZE,

        "learning_rate":LR,

        "epochs":EPOCHS,

        "max_length":MAX_LENGTH,

        "num_labels":num_labels
    }
)



# 训练
def train_epoch():

    model.train()

    total_loss = 0

    preds = []

    labels = []


    for batch in train_loader:


        input_ids = batch["input_ids"].to(device)

        mask = batch["attention_mask"].to(device)

        y = batch["labels"].to(device)


        optimizer.zero_grad()


        output = model(

            input_ids=input_ids,

            attention_mask=mask,

            labels=y
        )


        loss = output.loss


        loss.backward()


        optimizer.step()



        total_loss += loss.item()



        preds.extend(

            torch.argmax(
                output.logits,
                dim=1
            )
            .cpu()
            .numpy()

        )


        labels.extend(
            y.cpu().numpy()
        )



    return (

        total_loss / len(train_loader),

        accuracy_score(
            labels,
            preds
        )

    )



# 验证
def evaluate(loader):

    model.eval()

    total_loss = 0

    preds=[]

    labels=[]


    with torch.no_grad():

        for batch in loader:


            input_ids=batch["input_ids"].to(device)

            mask=batch["attention_mask"].to(device)

            y=batch["labels"].to(device)



            output=model(

                input_ids=input_ids,

                attention_mask=mask,

                labels=y

            )


            total_loss += output.loss.item()



            preds.extend(

                torch.argmax(
                    output.logits,
                    dim=1
                )
                .cpu()
                .numpy()

            )


            labels.extend(
                y.cpu().numpy()
            )


    return (

        total_loss/len(loader),

        accuracy_score(
            labels,
            preds
        )

    )



# 开始训练
best_acc = 0



for epoch in range(EPOCHS):


    train_loss,train_acc = train_epoch()


    dev_loss,dev_acc = evaluate(
        dev_loader
    )


    print(
        f"""
Epoch {epoch+1}/{EPOCHS}

Train Loss: {train_loss:.4f}
Train Acc : {train_acc:.4f}

Dev Loss: {dev_loss:.4f}
Dev Acc : {dev_acc:.4f}

"""
    )


    # SwanLab记录

    swanlab.log({

        "epoch":epoch+1,

        "train/loss":train_loss,

        "train/accuracy":train_acc,

        "dev/loss":dev_loss,

        "dev/accuracy":dev_acc

    })



    if dev_acc > best_acc:


        best_acc = dev_acc


        torch.save(

            model.state_dict(),

            "best_model.pth"

        )


        print("保存最佳模型")




# 测试
model.load_state_dict(

    torch.load(
        "best_model.pth",
        map_location=device
    )

)


test_loss,test_acc = evaluate(
    test_loader
)



print(
    "最佳验证准确率:",
    best_acc
)


print(
    "最终测试准确率:",
    test_acc
)



swanlab.log({

    "test/accuracy":test_acc

})


swanlab.finish()