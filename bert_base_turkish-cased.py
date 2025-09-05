import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
data = pd.read_csv("data/train.csv")
data["label"] = data["label"] - 1


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


train, val = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["label"]
)

train_dataset = AddressDataset(train, tokenizer, max_length=128)
val_dataset = AddressDataset(val, tokenizer, max_length=128)

train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=2
)
val_dataloader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=2
)


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    lr,
    n_epochs=10,
    patience=3,
    tol=1e-4,
    verbose=True,
    max_norm=None,
    device="cuda",
    checkpoint_path="checkpoint.pth",
    resume=False,
    accumulation_steps=1,
    eary_stop_f1=True,
):
    best_f1 = 0
    best_loss = float("inf")
    best_state = None
    no_improve_count = 0
    start_epoch = 0

    model.to(device)
    scaler = GradScaler(device=device)

    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_loss = ckpt.get("best_loss", float("inf"))
        no_improve_count = ckpt.get("no_improve_count", 0)
        start_epoch = ckpt.get("epoch", -1) + 1
        if verbose:
            print(
                f"[resume] {start_epoch}.(best_val_loss={best_loss:.6f}, "
                f"no_improve_count={no_improve_count})"
            )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    for epoch in range(start_epoch, n_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch} [TRAIN]")
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=device):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels) / accumulation_steps

            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm
                ) if max_norm is not None else None
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            n_batches += 1

        if (i + 1) % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm
            ) if max_norm is not None else None
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch} [VAL]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with autocast(device_type=device):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    vloss = loss_fn(logits, labels)

                val_loss += vloss.item()
                n_val_batches += 1
                all_preds.append(logits.argmax(dim=1))
                all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        f1 = f1_score(
            all_labels.cpu().numpy(), all_preds.cpu().numpy(), average="macro"
        )
        avg_val_loss = val_loss / n_val_batches

        if not eary_stop_f1:
            if avg_val_loss + tol < best_loss:
                best_loss = avg_val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve_count = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "no_improve_count": no_improve_count,
                    },
                    "checkpoint_latest.pth",
                )

            else:
                no_improve_count += 1
        else:
            if f1 > best_f1 + tol:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve_count = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "no_improve_count": no_improve_count,
                    },
                    "checkpoint_latest.pth",
                )

        if verbose:
            current_lrs = [g["lr"] for g in optimizer.param_groups]
            avg_train_loss = running_loss / n_batches
            print(
                f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Best Val Loss: {best_loss:.4f}, Val F1: {f1:.4f}, lr: {current_lrs}"
            )

        if no_improve_count >= patience:
            if verbose:
                print(
                    f"Early stopping @ epoch {epoch + 1} (best_val_loss={best_loss:.6f})"
                )
            if best_state is not None:
                model.load_state_dict(best_state)
            break

    return model


if __name__ == "__main__":
    model = CustomClassifier(model_name, num_labels=len(data["label"].unique()))
    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    """
    trained_model = train(
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimizer,
        lr = 1e-4,
        n_epochs=20,
        patience=5,
        tol=1e-4,
        verbose=True,
        device="cuda",
        accumulation_steps = 2,
        checkpoint_path="checkpoint_latest.pth"
    )
    """

    """
    optimizer = AdamW(model.parameters(), lr=5e-5)
    trained_model = train(
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimizer,
        lr = 5e-5,
        n_epochs=40,
        patience=5,
        tol=1e-4,
        verbose=True,
        device="cuda",
        accumulation_steps = 16,
        resume=True,
        checkpoint_path="checkpoint_latest.pth"
    )"""

    test_df = pd.read_csv("data/test.csv")
    test_dataset = AddressTestDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    device = "cuda"
    model = CustomClassifier(model_name, num_labels=len(data["label"].unique()))
    model.load_state_dict(
        torch.load("checkpoint_latest.pth", map_location=device)["model_state_dict"]
    )
    model.eval()
    model.to(device)

    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(logits, dim=-1)
            preds.extend(batch_preds.cpu().numpy())

    submission = pd.DataFrame({"id": test_df["id"], "label": np.array(preds) + 1})
    submission.to_csv("data/submission.csv", index=False)
