import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
import private_transformers  # replaces all manual DP machinery


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_dataloaders(tokenizer, batch_size):
    dataset = load_dataset("glue", "sst2")

    def tokenize(example):
        return tokenizer(example["sentence"], truncation=True)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collator)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    losses, correct, total = [], 0, 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
        losses.append(outputs.loss.item())
    model.train()
    return {"loss": float(np.mean(losses)), "accuracy": correct / total}


def train(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs):
    """Standard non-DP training for sanity checking."""
    for epoch in range(num_epochs):
        model.train()
        total_loss, steps = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": total_loss / steps})

        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: train_loss={total_loss/steps:.4f}, val_acc={metrics['accuracy']:.4f}")

    return model


def train_dp(
    model, train_loader, val_loader, optimizer, scheduler, device,
    num_epochs, train_batch_size, per_device_train_batch_size,
):
    """
    DP-SGD training using private_transformers.PrivacyEngine.

    Key ideas (mirroring the ViT example):
      - PrivacyEngine is already attached to the optimizer before this function runs.
      - Loss MUST be per-sample (reduction="none"), not a scalar mean.
      - Gradient accumulation:
          * Call optimizer.virtual_step(loss=loss) for micro-steps that should NOT
            update weights yet (just accumulate clipped gradients).
          * Call optimizer.step(loss=loss) on the final micro-step to add noise,
            update weights, and advance the privacy accountant.
      - optimizer.zero_grad() is called once per logical batch, not per micro-step.
    """
    gradient_accumulation_steps = train_batch_size // per_device_train_batch_size

    for epoch in range(num_epochs):
        model.train()
        total_loss, steps = 0.0, 0
        optimizer.zero_grad()  # reset once at the start of every epoch

        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch+1} (DP)")
        for global_step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass — must request per-sample losses so the privacy engine
            # can clip each sample's gradient individually.
            outputs = model(**batch)
            # outputs.loss is the mean; recompute with reduction="none" for DP.
            import torch.nn.functional as F
            loss = F.cross_entropy(outputs.logits, batch["labels"], reduction="none")

            total_loss += loss.mean().item()
            steps += 1

            if global_step % gradient_accumulation_steps == 0:
                # Full logical batch complete: clip + noise + weight update.
                optimizer.step(loss=loss)
                optimizer.zero_grad()
                scheduler.step()
            else:
                # Micro-step: clip and accumulate clipped gradients, no noise/update yet.
                optimizer.virtual_step(loss=loss)

            pbar.set_postfix({"loss": total_loss / steps})

        # Privacy engine tracks epsilon internally; read it back after each epoch.
        eps = optimizer.privacy_engine.get_privacy_spent()
        metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch+1}: "
            f"train_loss={total_loss/steps:.4f}, "
            f"val_acc={metrics['accuracy']:.4f}, "
            f"ε={eps}"
        )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    # train_batch_size = logical (large) batch; per_device is the micro-batch.
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Logical batch size used for DP accounting.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32,
                        help="Physical mini-batch that fits in GPU memory.")
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--target_epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=0.00047505938242280285)
    parser.add_argument("--model_name", type=str, default="FacebookAI/roberta-large")
    parser.add_argument("--freeze_layers", type=int, default=20,
                        help="Number of bottom transformer layers to freeze "
                             "(roberta-large has 24 total). Fewer trainable "
                             "params = less noise injected per parameter.")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Freeze bottom N layers; train only top layers + classifier head.
    for p in model.parameters():
        p.requires_grad = False
    trainable_layers = [model.roberta.encoder.layer[-1], model.classifier]

    total_params = 0
    trainable_params = 0

    for p in model.parameters():
            p.requires_grad = False
            total_params += p.numel()

    for layer in trainable_layers:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()

    print(f"Total parameters count: {total_params:,}") # ~108M
    print(f"Trainable parameters count: {trainable_params:,}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    model.to(device)

    # Use the physical (per-device) batch size for the data loader.
    train_loader, val_loader = get_dataloaders(tokenizer, args.per_device_train_batch_size)

    sample_size = len(train_loader.dataset)  # 67,349 for SST-2 train split

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_steps = (len(train_loader) // (args.train_batch_size // args.per_device_train_batch_size)) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    if args.use_dp:
        # PrivacyEngine replaces: noise injection, per-sample clipping,
        # and epsilon accounting — no manual code needed.
        privacy_engine = private_transformers.PrivacyEngine(
            model,
            batch_size=args.train_batch_size,       # logical batch for accounting
            sample_size=sample_size,
            epochs=args.num_epochs,
            max_grad_norm=args.max_grad_norm,
            target_epsilon=args.target_epsilon,
        )
        privacy_engine.attach(optimizer)            # monkey-patches step() / virtual_step()

        model = train_dp(
            model, train_loader, val_loader, optimizer, scheduler, device,
            args.num_epochs, args.train_batch_size, args.per_device_train_batch_size,
        )
    else:
        model = train(
            model, train_loader, val_loader, optimizer, scheduler, device, args.num_epochs,
        )

    os.makedirs("model", exist_ok=True)
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
    print("Done. Model saved to ./model")


if __name__ == "__main__":
    main()