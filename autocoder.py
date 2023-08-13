import torch
from transformers import AutoTokenizer, BertForMultipleChoice

# Mapping of codes to their meanings
codes = {
    "01": "good",
    "02": "fair",
    "03": "bad",
    "04": "expensive",
    "05": "long hold times",
    "06": "website problems",
    "07": "app problems",
    "50": "other",
    "97": "No/None",
    "99": "Don't know/Refused"
}

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForMultipleChoice.from_pretrained("bert-base-uncased")


def autocode_open_ends(open_end: str) -> str:
    choices = list(codes.values())
    open_end_list = [open_end] * len(choices)

    labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

    encoding = tokenizer(open_end_list, choices, return_tensors="pt", padding=True)

    outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

    logits = outputs.logits

    # get the predicted class
    result = torch.argmax(logits)
    index = result.item()
    code = list(codes.keys())[index]
    return code


while True:
    open_end = input("Enter open end: ")
    code = autocode_open_ends(open_end)
    print(f"Code: {code}, Code Label: {codes[code]}")

