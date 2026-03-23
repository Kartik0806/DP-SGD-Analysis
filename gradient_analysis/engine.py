import copy
import math
import os

import evaluate
import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_scheduler, set_seed

from .config import TrainConfig
from .models import move_batch_to_device
from .tasks import PRIMARY_METRICS, is_regression_task


def _maybe_init_wandb(config: TrainConfig):
    if not config.use_wandb:
        return None

    import wandb

    run_name = config.wandb_run_name or f"{config.task}_{'lora' if config.use_lora else 'full'}"
    project = config.wandb_project or f"{config.task}_comparison_estimation"

    wandb.init(
        project=project,
        name=run_name,
        tags=[config.task, "LoRA" if config.use_lora else "Full"],
        config={
            "task": config.task,
            "model_name": config.model_name,
            "use_lora": config.use_lora,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "grad_accumulation_steps": config.grad_accumulation_steps,
        },
    )
    return wandb


def evaluate_model(model, dataloader, metric, device, task: str):
    model.eval()
    losses = []
    all_predictions = []
    all_labels = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch)

        losses.append(outputs.loss.detach().cpu().item())

        if is_regression_task(task):
            predictions = outputs.logits.squeeze(-1).detach().cpu().numpy()
        else:
            predictions = outputs.logits.argmax(dim=-1).detach().cpu().numpy()

        labels = batch["labels"].detach().cpu().numpy()
        all_predictions.append(predictions)
        all_labels.append(labels)

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    metrics = metric.compute(predictions=predictions, references=labels)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train(model, dataloaders: dict, config: TrainConfig):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metric = evaluate.load("glue", config.task)
    wandb = _maybe_init_wandb(config)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(dataloaders["train_loader"]) / config.grad_accumulation_steps)
    total_training_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_training_steps * config.warmup_ratio)
    scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_metric_name = PRIMARY_METRICS[config.task]
    best_metric_value = float("-inf")
    best_state_dict = None
    global_step = 0

    os.makedirs(config.output_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        progress_bar = tqdm(dataloaders["train_loader"], desc=f"Epoch {epoch + 1}/{config.num_epochs}")

        for step, batch in enumerate(progress_bar, start=1):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / config.grad_accumulation_steps
            loss.backward()

            running_loss += loss.item() * config.grad_accumulation_steps

            if step % config.grad_accumulation_steps == 0 or step == len(dataloaders["train_loader"]):

                # optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.log_every == 0:
                    avg_loss = running_loss / step
                    progress_bar.set_postfix(train_loss=f"{avg_loss:.4f}")
                    if wandb is not None:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/lr": scheduler.get_last_lr()[0],
                                "train/step": global_step,
                                "epoch": epoch + 1,
                            }
                        )

        eval_metrics = evaluate_model(
            model=model,
            dataloader=dataloaders["eval_loader"],
            metric=metric,
            device=device,
            task=config.task,
        )
        train_loss = running_loss / max(len(dataloaders["train_loader"]), 1)
        current_metric = eval_metrics[best_metric_name]

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"eval_loss={eval_metrics['loss']:.4f}, "
            f"{best_metric_name}={current_metric:.4f}"
        )

        if wandb is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_loss,
                    "eval/loss": eval_metrics["loss"],
                    **{f"eval/{key}": value for key, value in eval_metrics.items() if key != "loss"},
                }
            )

        if current_metric > best_metric_value:
            best_metric_value = current_metric
            best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    save_path = os.path.join(config.output_dir, f"{config.task}_{'lora' if config.use_lora else 'full'}")
    model.save_pretrained(save_path)

    if wandb is not None:
        wandb.finish()

    return {
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "saved_model_path": save_path,
    }
