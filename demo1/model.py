from transformers import AutoModelForSequenceClassification

from config import MODEL_NAME



def build_model(
        num_labels,
        id2label,
        label2id
):

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )


    return model