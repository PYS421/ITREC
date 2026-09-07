import torch
from torch import nn
from transformers import AutoModel
from config import MODEL_NAME


#创建一个BERT文本分类器
class BertClassifier(nn.Module):


    def __init__(
        self,
        model_name,
        num_labels
    ):

        super().__init__()


        # BERT编码器
        self.bert = AutoModel.from_pretrained(
            model_name
        )


        # 防止过拟合
        self.dropout = nn.Dropout(
            0.1
        )


        # 分类层
        self.classifier = nn.Linear(

            self.bert.config.hidden_size,

            num_labels

        )



    def forward(
        self,
        input_ids,
        attention_mask
    ):


        output = self.bert(

            input_ids=input_ids,

            attention_mask=attention_mask

        )


        # CLS向量
        cls = output.last_hidden_state[:,0,:]


        cls = self.dropout(cls)


        logits = self.classifier(cls)


        return logits





def build_model(
        num_labels,
        id2label=None,
        label2id=None
):


    model = BertClassifier(

        model_name=MODEL_NAME,

        num_labels=num_labels

    )


    return model