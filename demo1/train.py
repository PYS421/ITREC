import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


# 自己封装的评价指标
from metrics import (
    accuracy,
    precision,
    recall,
    f1
)


from config import *

#添加get-collate-fn
from dataset import (
    load_data,
    NewsDataset,
    get_collate_fn
)


from model import build_model


from utils import seed_everything

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)






# 评价函数
def evaluate(model, loader):


    model.eval()


    preds = []

    labels = []



    with torch.no_grad():


        for batch in loader:



            input_ids = batch["input_ids"].to(device)


            mask = batch["attention_mask"].to(device)


            y = batch["labels"].to(device)



            logits = model(

                input_ids=input_ids,

                attention_mask=mask

            )



            pred = torch.argmax(

                logits,

                dim=1

            )



            preds.extend(

                pred.cpu().numpy()

            )


            labels.extend(

                y.cpu().numpy()

            )





    # 调用自己的metrics.py


    acc = accuracy(

        preds,

        labels

    )


    p = precision(

        preds,

        labels,

        num_classes=len(set(labels))

    )


    r = recall(

        preds,

        labels,

        num_classes=len(set(labels))

    )


    score_f1 = f1(

        p,

        r

    )



    return {


        "accuracy": acc,


        "precision": p,


        "recall": r,


        "f1": score_f1

    }








def train():

    seed_everything()

    # 1. 加载数据
    train_data = load_data(

        TRAIN_PATH

    )
    print(train_data[:3])

    dev_data = load_data(

        DEV_PATH

    )


    test_data = load_data(

        TEST_PATH

    )


    # 2. 获取类别
    # list dict写法
    label_names = set()
    for item in train_data:

        label_names.add(

            item["label_name"]

        )



    labels = sorted(

        list(label_names)

    )

    label2id = {


        label:i


        for i,label in enumerate(labels)


    }

    id2label = {


        i:label


        for label,i in label2id.items()


    }

    # 3. 添加数字标签


    for dataset in [

        train_data,

        dev_data,

        test_data

    ]:


        for item in dataset:


            item["label"] = label2id[

                item["label_name"]

            ]

    tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
        )


    collate_fn = get_collate_fn(
        tokenizer
        )

    # 4. Dataset
    train_loader = DataLoader(

    NewsDataset(
        train_data,
        tokenizer
    ),

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=collate_fn

    )




    dev_loader = DataLoader(

    NewsDataset(
        dev_data,
        tokenizer
    ),

    batch_size=BATCH_SIZE,

    collate_fn=collate_fn

    )


    # 5. 创建模型
    model = build_model(


        len(labels),


        id2label,


        label2id


    )


    model.to(device)






    optimizer = torch.optim.AdamW(


        model.parameters(),


        lr=LR


    )





    # 自己计算loss
    loss_fn = torch.nn.CrossEntropyLoss()
    best = 0







    # 6.训练
    for epoch in range(EPOCHS):



        model.train()



        total_loss = 0





        for batch in train_loader:



            optimizer.zero_grad()





            y = batch["labels"].to(device)





            logits = model(


                input_ids=batch["input_ids"].to(device),


                attention_mask=batch["attention_mask"].to(device)


            )





            loss = loss_fn(


                logits,


                y


            )




            loss.backward()



            optimizer.step()



            total_loss += loss.item()






        # 7.验证
        result = evaluate(


            model,


            dev_loader


        )





        print("====================")


        print(

            "epoch:",

            epoch + 1

        )



        print(

            "loss:",

            total_loss / len(train_loader)

        )



        print(

            "accuracy:",

            result["accuracy"]

        )



        print(

            "precision:",

            result["precision"]

        )



        print(

            "recall:",

            result["recall"]

        )



        print(

            "f1:",

            result["f1"]

        )







        # 保存最佳模型
        if result["f1"] > best:



            best = result["f1"]



            torch.save(


                model.state_dict(),


                "checkpoints/best_model.pth"


            )



            print(

                "保存最佳模型"

            )




if __name__ == "__main__":


    train()