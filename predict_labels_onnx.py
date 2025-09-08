if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    import pandas as pd
    import onnxruntime as ort
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from utils.model_utils import AddressTestDataset
    import numpy as np

    tokenizer_path = "./model_files/model_quantized_onnx"
    onnx_model_path = "./model_files/model_quantized_onnx/model.onnx"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    ort_session = ort.InferenceSession(
        onnx_model_path, providers=["CPUExecutionProvider"]
    )

    test_df = pd.read_csv("data/test.csv")
    test_dataset = AddressTestDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    preds = []

    for batch in tqdm(test_loader, desc="ONNX Inference"):
        ort_inputs = {
            "input_ids": batch["input_ids"].cpu().numpy().astype(np.int64),
            "attention_mask": batch["attention_mask"].cpu().numpy().astype(np.int64),
        }
        logits = ort_session.run(["logits"], ort_inputs)[0]
        batch_preds = logits.argmax(axis=1)
        preds.extend(batch_preds.tolist())

    preds = np.array(preds)

    submission = pd.DataFrame({"id": test_df["id"], "label": preds + 1})
    submission.to_csv("data/submission_onnx.csv", index=False)
