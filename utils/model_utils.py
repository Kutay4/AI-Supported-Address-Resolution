from transformers import AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CustomClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.05)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0]
        x = self.dropout(cls_rep)
        logits = self.classifier(x)
        return logits


class AddressDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        address = self.df.iloc[idx]["address"]
        label = self.df.iloc[idx]["label"]

        encoding = self.tokenizer(
            address,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)

        return item


class AddressTestDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        address = self.df.iloc[idx]["address"]
        encoding = self.tokenizer(
            address,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        return item
