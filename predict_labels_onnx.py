if __name__ == "__main__":
    import os 
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  
    import pandas as pd
    import onnxruntime as ort
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from utils import AddressTestDataset
    import numpy as np

    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    onnx_model_path = "model.onnx"
    ort_session = ort.InferenceSession(onnx_model_path)

    test_df = pd.read_csv("test.csv")
    test_dataset = AddressTestDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, prefetch_factor = 2, persistent_workers=True)

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

    submission = pd.DataFrame({
        "id": test_df["id"],
        "label": preds + 1
    })
    submission.to_csv("submission_onnx.csv", index=False)
