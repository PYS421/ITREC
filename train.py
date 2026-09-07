import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from sklearn.metrics import accuracy_score


from config import *

from dataset import (
    load_data,
    NewsDataset
)

from model import build_model

from utils import seed_everything



device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



def evaluate(model,loader):

    model.eval()

    preds=[]

    labels=[]


    with torch.no_grad():

        for batch in loader:


            input_ids=batch["input_ids"].to(device)

            mask=batch["attention_mask"].to(device)

            y=batch["labels"].to(device)


            output=model(
                input_ids=input_ids,
                attention_mask=mask
            )


            pred=torch.argmax(
                output.logits,
                dim=1
            )


            preds.extend(
                pred.cpu().numpy()
            )


            labels.extend(
                y.cpu().numpy()
            )


    return accuracy_score(
        labels,
        preds
    )



def train():


    seed_everything()


    train_df=load_data(TRAIN_PATH)

    dev_df=load_data(DEV_PATH)

    test_df=load_data(TEST_PATH)



    labels=sorted(
        train_df["label_name"].unique()
    )


    label2id={
        label:i
        for i,label in enumerate(labels)
    }


    id2label={
        i:label
        for label,i in label2id.items()
    }



    for df in [
        train_df,
        dev_df,
        test_df
    ]:

        df["label"]=df["label_name"].map(label2id)



    tokenizer=AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    train_loader=DataLoader(
        NewsDataset(train_df,tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    dev_loader=DataLoader(
        NewsDataset(dev_df,tokenizer),
        batch_size=BATCH_SIZE
    )



    model=build_model(
        len(labels),
        id2label,
        label2id
    )


    model.to(device)



    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )


    best=0



    for epoch in range(EPOCHS):


        model.train()


        for batch in train_loader:


            optimizer.zero_grad()


            output=model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )


            output.loss.backward()

            optimizer.step()



        acc=evaluate(
            model,
            dev_loader
        )


        print(
            "epoch:",
            epoch+1,
            "dev acc:",
            acc
        )


        if acc>best:

            best=acc

            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth"
            )



if __name__=="__main__":

    train()