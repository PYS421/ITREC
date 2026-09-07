import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding

columns = [
    "news_id",
    "label_code",
    "label_name",
    "title",
    "keywords"
]

def load_data(path):


    data=[]


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:


        for line in f:


            line=line.strip()


            row=line.split("_!_")


            item={

                "news_id": row[0],

                "label_code": row[1],

                "label_name": row[2],

                "title": row[3],

                "keywords": row[4]

            }


            data.append(item)



    return data

class NewsDataset(Dataset):


    def __init__(
            self,
            data,
            tokenizer
    ):

        self.data=data

        self.tokenizer=tokenizer



    def __len__(self):

        return len(self.data)



    def __getitem__(self,index):


        item=self.data[index]


        text=str(
            item["title"]
        )


        label=item["label"]



        encode=self.tokenizer(
            text,
            max_length=128,
            truncation=True,

        )
        encode["labels"] = label

        return encode


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

#添加动态pad
def get_collate_fn(tokenizer):


    return DataCollatorWithPadding(

        tokenizer=tokenizer

    )