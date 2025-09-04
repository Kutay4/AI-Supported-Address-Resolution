import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np


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


if __name__ == "__main__":
    address = "İzmir, Karşıyaka, Bostanlı Mahallesi, Cemal Gürsel Caddesi No:10"

    print(predict_address_onnx(address))
