import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from .model_utils import AddressTestDataset
from torch.utils.data import DataLoader
import streamlit as st

def predict_address_onnx(address, model_path, tokenizer_path):
    ort_session = ort.InferenceSession(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    enc = tokenizer(
        address,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )

    ort_input = {
        "input_ids": enc["input_ids"].cpu().numpy().astype(np.int64),
        "attention_mask": enc["attention_mask"].cpu().numpy().astype(np.int64),
    }
    logit = ort_session.run(["logits"], ort_input)[0]
    pred = logit.argmax(axis=1)[0] + 1

    return pred

def predict_csv_onnx(csv_file, model_path, tokenizer_path, progress_bar, batch_size=4, streeamlit = True):

    ort_session = ort.InferenceSession(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    test_dataset = AddressTestDataset(csv_file, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    preds = []

    for i,batch in enumerate(test_loader):
        ort_inputs = {
            "input_ids": batch["input_ids"].cpu().numpy().astype(np.int64),
            "attention_mask": batch["attention_mask"].cpu().numpy().astype(np.int64),
        }
        logits = ort_session.run(["logits"], ort_inputs)[0]
        batch_preds = logits.argmax(axis=1)
        preds.extend(batch_preds.tolist())
        if streeamlit:
            progress_percentage = (i + 1) / len(test_loader)
            progress_bar.progress(progress_percentage, text=f"{(i+1)*batch_size}%")

    preds = np.array(preds)
    return pd.DataFrame({"address": csv_file["address"], "label": preds + 1})



if __name__ == "__main__":
    address = "İzmir, Karşıyaka, Bostanlı Mahallesi, Cemal Gürsel Caddesi No:10"

    print(predict_address_onnx(address))
