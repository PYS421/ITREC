import pandas as pd
import torch

from torch.utils.data import Dataset


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
            max_length=128,
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