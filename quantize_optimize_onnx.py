from transformers import AutoTokenizer
import onnxruntime as ort
from optimum.onnxruntime import ORTQuantizer, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig, AutoQuantizationConfig
import os

# sample address to compare model results.
address = "İzmir, Karşıyaka, Bostanlı Mahallesi, Cemal Gürsel Caddesi No:10"

# DEFAULT ONNX
onnx_path = "./model_onnx/model.onnx"
tokenizer = AutoTokenizer.from_pretrained("./model_onnx")
enc = tokenizer(
    address, padding="max_length", max_length=128, truncation=True, return_tensors="np"
)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

logits = sess.run(
    ["logits"], {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
)[0]
pred = int(logits.argmax(axis=1)[0]) + 1

print("Prediction:", pred)

# QUANTIZING
quantizer = ORTQuantizer.from_pretrained(model_or_path="./model_onnx")
# qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=True)
quantizer.quantize(save_dir="./model_quantized_onnx", quantization_config=qconfig)

# QUANTIZED ONNX
onnx_path = "./model_quantized_onnx/model_quantized.onnx"
tokenizer = AutoTokenizer.from_pretrained("./model_onnx")
enc = tokenizer(
    address, padding="max_length", max_length=128, truncation=True, return_tensors="np"
)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

logits = sess.run(
    ["logits"], {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
)[0]
pred = int(logits.argmax(axis=1)[0]) + 1

print("Prediction:", pred)

# need to rename onnx file for optimizer
os.rename(
    "./model_quantized_onnx/model_quantized.onnx", "./model_quantized_onnx/model.onnx"
)
# OPTIMIZING
optimizer = ORTOptimizer.from_pretrained(model_or_path="./model_quantized_onnx")
optimization_config = OptimizationConfig(
    optimization_level=1,
)
optimizer.optimize(
    optimization_config=optimization_config, save_dir="./model_quantized_optimized_onnx"
)

# OPTIMIZED ONNX
onnx_path = "./model_quantized_optimized_onnx/model_optimized.onnx"
tokenizer = AutoTokenizer.from_pretrained("./model_quantized_optimized_onnx")

enc = tokenizer(
    address, padding="max_length", max_length=128, truncation=True, return_tensors="np"
)

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

logits = sess.run(
    ["logits"], {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
)[0]
pred = int(logits.argmax(axis=1)[0]) + 1
print("Prediction:", pred)
