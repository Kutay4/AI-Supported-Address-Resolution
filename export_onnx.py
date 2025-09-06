import torch
from transformers import AutoConfig, AutoTokenizer
from utils.model_utils import CustomClassifier
import onnxruntime as ort
import os

model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cpu"
num_labels = 10390

model = CustomClassifier(model_name, num_labels=num_labels)
model.load_state_dict(
    torch.load("./model_files/checkpoint_latest.pth", map_location=device)[
        "model_state_dict"
    ]
)

# sample address to compare model results.
address = "İzmir, Karşıyaka, Bostanlı Mahallesi, Cemal Gürsel Caddesi No:10"

enc = tokenizer(
    address,
    padding="max_length",
    max_length=128,
    truncation=True,
    return_tensors="pt",
)

input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)
with torch.no_grad():
    logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    pred_class = torch.argmax(logits, dim=1).item() + 1

# default torch model result:
print("Predicted id with torch model:", pred_class)

hf_dir = "./model_files/model_onnx"
os.makedirs(hf_dir, exist_ok=True)
onnx_model_path = "./model_files/model_onnx/model.onnx"

dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "logits": {0: "batch_size"},
}

with torch.inference_mode():
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        export_params=True,
        opset_version=17,
    )

ort_session = ort.InferenceSession(onnx_model_path)

ort_inputs = {
    "input_ids": input_ids.cpu().numpy(),
    "attention_mask": attention_mask.cpu().numpy(),
}

logits_onnx = ort_session.run(["logits"], ort_inputs)[0]
onnx_output = int(logits_onnx.argmax(axis=1)[0]) + 1
# onnx model result:
print("Predicted id with onnx model:", onnx_output)

cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
cfg.save_pretrained(hf_dir)

tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tok.save_pretrained(hf_dir)
