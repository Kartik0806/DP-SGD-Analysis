import copy
import os

import evaluate
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import set_seed

from .config import TrainConfig
from .engine import _maybe_init_wandb, evaluate_model
from .models import move_batch_to_device, unwrap_model
from .tasks import PRIMARY_METRICS


def _accumulate_unclipped_batch_gradients(optimizer, gradient_buffer: list[torch.Tensor | None]):
    for index, parameter in enumerate(optimizer.params):
        grad_sample = optimizer._get_flat_grad_sample(parameter)
        batch_gradient = grad_sample.detach().sum(dim=0)

        if gradient_buffer[index] is None:
            gradient_buffer[index] = batch_gradient.clone()
        else:
            gradient_buffer[index] += batch_gradient

    return gradient_buffer


def _collect_clipped_batch_gradients(optimizer):
    clipped_gradients = []
    for parameter in optimizer.params:
        clipped_gradients.append(
            parameter.summed_grad.detach().clone() if parameter.summed_grad is not None else None
        )
    return clipped_gradients


def _sample_tensor_values(tensors: list[torch.Tensor | None], max_values: int) -> torch.Tensor:
    flat_tensors = [tensor.reshape(-1).float().cpu() for tensor in tensors if tensor is not None]
    if not flat_tensors:
        return torch.empty(0)

    total_values = sum(tensor.numel() for tensor in flat_tensors)
    if total_values <= max_values:
        return torch.cat(flat_tensors)

    sampled_tensors = []
    for tensor in flat_tensors:
        sample_size = max(1, int(round(max_values * tensor.numel() / total_values)))
        sample_size = min(sample_size, tensor.numel())
        sample_indices = torch.linspace(0, tensor.numel() - 1, steps=sample_size).long()
        sampled_tensors.append(tensor[sample_indices])

    return torch.cat(sampled_tensors)[:max_values]


def _summarize_gradients(tensors: list[torch.Tensor | None], histogram_max_values: int):
    total_sum = 0.0
    total_sq_sum = 0.0
    total_numel = 0
    max_abs = 0.0

    for tensor in tensors:
        if tensor is None:
            continue

        current = tensor.detach().float()
        total_sum += current.sum().item()
        total_sq_sum += current.pow(2).sum().item()
        total_numel += current.numel()
        max_abs = max(max_abs, current.abs().max().item())

    if total_numel == 0:
        return {
            "norm": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "max_abs": 0.0,
            "histogram_values": torch.empty(0),
        }

    mean = total_sum / total_numel
    variance = max((total_sq_sum / total_numel) - (mean**2), 0.0)

    return {
        "norm": total_sq_sum**0.5,
        "mean": mean,
        "std": variance**0.5,
        "max_abs": max_abs,
        "histogram_values": _sample_tensor_values(tensors, histogram_max_values),
    }


def _log_gradient_summaries(wandb_module, unclipped_gradients, clipped_gradients, logical_step, epoch, config: TrainConfig):
    if wandb_module is None or not config.log_batch_gradients:
        return

    pre_clip = _summarize_gradients(unclipped_gradients, config.gradient_histogram_max_values)
    post_clip = _summarize_gradients(clipped_gradients, config.gradient_histogram_max_values)

    payload = {
        "epoch": epoch,
        "gradient/logical_step": logical_step,
        "gradient/pre_clip_norm": pre_clip["norm"],
        "gradient/pre_clip_mean": pre_clip["mean"],
        "gradient/pre_clip_std": pre_clip["std"],
        "gradient/pre_clip_max_abs": pre_clip["max_abs"],
        "gradient/post_clip_norm": post_clip["norm"],
        "gradient/post_clip_mean": post_clip["mean"],
        "gradient/post_clip_std": post_clip["std"],
        "gradient/post_clip_max_abs": post_clip["max_abs"],
    }

    if pre_clip["histogram_values"].numel() > 0:
        payload["gradient/pre_clip_histogram"] = wandb_module.Histogram(pre_clip["histogram_values"].numpy())
    if post_clip["histogram_values"].numel() > 0:
        payload["gradient/post_clip_histogram"] = wandb_module.Histogram(post_clip["histogram_values"].numpy())

    wandb_module.log(payload)


def train_private(model, dataloaders: dict, config: TrainConfig):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = evaluate.load("glue", config.task) if config.run_eval else None
    wandb = _maybe_init_wandb(config)

    print(
        "Warning: logging pre-clip gradients to W&B invalidates any meaningful differential privacy guarantee. "
        "This run should be treated as gradient analysis, not private training."
    )

    model = PrivacyEngine.get_compatible_module(model)
    model.to(device)

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    privacy_engine = PrivacyEngine(
        accountant=config.dp_accountant,
        secure_mode=config.dp_secure_mode,
    )
    model, optimizer, private_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloaders["train_loader"],
        noise_multiplier=config.noise_multiplier,
        max_grad_norm=config.max_grad_norm,
        poisson_sampling=config.dp_poisson_sampling,
    )

    best_metric_name = PRIMARY_METRICS[config.task] if config.run_eval else None
    best_metric_value = float("-inf") if config.run_eval else None
    best_state_dict = None
    logical_step = 0

    os.makedirs(config.output_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        physical_steps = 0
        logical_batch_unclipped_gradients = [None for _ in optimizer.params]

        with BatchMemoryManager(
            data_loader=private_train_loader,
            max_physical_batch_size=config.max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            progress_bar = tqdm(memory_safe_data_loader, desc=f"DP Epoch {epoch + 1}/{config.num_epochs}")

            for batch in progress_bar:
                optimizer.zero_grad()
                batch = move_batch_to_device(batch, device)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                running_loss += loss.detach().cpu().item()
                physical_steps += 1

                logical_batch_unclipped_gradients = _accumulate_unclipped_batch_gradients(
                    optimizer=optimizer,
                    gradient_buffer=logical_batch_unclipped_gradients,
                )
                optimizer.clip_and_accumulate()

                skip_next_step = optimizer._check_skip_next_step()
                optimizer._is_last_step_skipped = skip_next_step

                if not skip_next_step:
                    logical_step += 1
                    clipped_gradients = _collect_clipped_batch_gradients(optimizer)

                    _log_gradient_summaries(
                        wandb_module=wandb,
                        unclipped_gradients=logical_batch_unclipped_gradients,
                        clipped_gradients=clipped_gradients,
                        logical_step=logical_step,
                        epoch=epoch + 1,
                        config=config,
                    )

                    if config.update_weights:
                        optimizer.add_noise()
                        optimizer.scale_grad()
                        optimizer.original_optimizer.step()

                    logical_batch_unclipped_gradients = [None for _ in optimizer.params]

                    if logical_step % config.log_every == 0:
                        avg_loss = running_loss / max(physical_steps, 1)
                        progress_bar.set_postfix(train_loss=f"{avg_loss:.4f}")
                        if wandb is not None:
                            wandb.log(
                                {
                                    "train/loss": avg_loss,
                                    "train/logical_step": logical_step,
                                    "epoch": epoch + 1,
                                }
                            )

        train_loss = running_loss / max(physical_steps, 1)

        if config.run_eval and metric is not None:
            eval_metrics = evaluate_model(
                model=model,
                dataloader=dataloaders["eval_loader"],
                metric=metric,
                device=device,
                task=config.task,
            )
            current_metric = eval_metrics[best_metric_name]

            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f}, "
                f"eval_loss={eval_metrics['loss']:.4f}, "
                f"{best_metric_name}={current_metric:.4f}, "
                f"weights_updated={config.update_weights}"
            )

            if wandb is not None:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/epoch_loss": train_loss,
                        "eval/loss": eval_metrics["loss"],
                        "train/weights_updated": int(config.update_weights),
                        **{f"eval/{key}": value for key, value in eval_metrics.items() if key != "loss"},
                    }
                )

            if current_metric > best_metric_value:
                best_metric_value = current_metric
                best_state_dict = copy.deepcopy(unwrap_model(model).state_dict())
        else:
            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f}, "
                f"logical_steps={logical_step}, "
                f"weights_updated={config.update_weights}"
            )
            if wandb is not None:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/epoch_loss": train_loss,
                        "train/logical_steps": logical_step,
                        "train/weights_updated": int(config.update_weights),
                    }
                )

    unwrapped_model = unwrap_model(model)
    if best_state_dict is not None and config.run_eval:
        unwrapped_model.load_state_dict(best_state_dict)

    save_path = os.path.join(config.output_dir, f"{config.task}_{'lora' if config.use_lora else 'full'}_dp")
    unwrapped_model.save_pretrained(save_path)

    if wandb is not None:
        wandb.finish()

    return {
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "saved_model_path": save_path,
        "noise_multiplier": config.noise_multiplier,
        "max_grad_norm": config.max_grad_norm,
        "logical_batch_size": config.batch_size,
        "max_physical_batch_size": config.max_physical_batch_size,
        "weights_updated": config.update_weights,
        "gradient_logging_enabled": config.log_batch_gradients,
    }
