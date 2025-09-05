if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from transformers import AutoTokenizer
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from utils import AddressTestDataset, CustomClassifier

    if "fbgemm" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "fbgemm"
    elif "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"

    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(8)

    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cpu"

    model = CustomClassifier(model_name, num_labels=10390)
    model.load_state_dict(
        torch.load("checkpoint_latest.pth", map_location=device)["model_state_dict"]
    )
    model.eval()

    model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    model.to(device)

    test_df = pd.read_csv("data/test.csv")
    test_dataset = AddressTestDataset(test_df, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
    )

    compiled_model = torch.compile(model, backend="inductor", mode="max-autotune")

    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            logits = compiled_model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(logits, dim=-1)
            preds.append(batch_preds)

    preds = torch.cat(preds).cpu().numpy()

    submission = pd.DataFrame({"id": test_df["id"], "label": np.array(preds) + 1})
    submission.to_csv("data/submission.csv", index=False)
