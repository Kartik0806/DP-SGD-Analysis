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


def _compute_per_sample_norms_from_grad_samples(optimizer) -> torch.Tensor | None:
    """
    Compute global per-sample gradient L2 norms across all parameters.

    Returns:
        Tensor of shape [batch_size, 1], or None if no grad_sample exists.
    """
    per_sample_sq_norms = None

    for parameter in optimizer.params:
        grad_sample = optimizer._get_flat_grad_sample(parameter)
        if grad_sample is None:
            continue

        grad_sample = grad_sample.detach()

        # [batch_size]
        param_sq_norms = grad_sample.reshape(grad_sample.shape[0], -1).pow(2).sum(dim=1)

        if per_sample_sq_norms is None:
            per_sample_sq_norms = param_sq_norms
        else:
            per_sample_sq_norms = per_sample_sq_norms + param_sq_norms

    if per_sample_sq_norms is None:
        return None

    return per_sample_sq_norms.sqrt().unsqueeze(1)  # [batch_size, 1]


def _accumulate_logical_batch_per_sample_sq_norms(
    optimizer,
    accumulated_sq_norms: torch.Tensor | None,
    exclude_params = None
) -> torch.Tensor | None:

    current_sq_norms = None

    for parameter in optimizer.params:
        if parameter in exclude_params:
            # print("check")  # ADD THIS CHECK
            continue
        grad_sample = optimizer._get_flat_grad_sample(parameter)
        if grad_sample is None:
            continue

        grad_sample = grad_sample.detach()
        param_sq_norms = grad_sample.reshape(grad_sample.shape[0], -1).pow(2).sum(dim=1)

        if current_sq_norms is None:
            current_sq_norms = param_sq_norms
        else:
            current_sq_norms = current_sq_norms + param_sq_norms

    if current_sq_norms is None:
        return accumulated_sq_norms

    if accumulated_sq_norms is None:
        return current_sq_norms.clone()

    return torch.cat([accumulated_sq_norms, current_sq_norms], dim=0)


def _compute_post_clip_norms(
    pre_clip_norms: torch.Tensor,
    max_grad_norm,
) -> torch.Tensor:
    """
    For flat clipping, clipped per-sample norms are min(pre_clip_norm, max_grad_norm).

    Returns:
        Tensor of shape [batch_size, 1]
    """
    if isinstance(max_grad_norm, (list, tuple)):
        raise ValueError(
            "This helper assumes flat clipping with scalar max_grad_norm. "
            "Per-layer clipping needs a different post-clip computation."
        )

    clip_value = float(max_grad_norm)
    return pre_clip_norms.clamp(max=clip_value)


def _make_per_sample_norm_table(wandb_module, pre_clip_norms: torch.Tensor, post_clip_norms: torch.Tensor):
    """
    Create a W&B table with one row per sample in the logical batch.
    """
    pre_cpu = pre_clip_norms.detach().cpu()
    post_cpu = post_clip_norms.detach().cpu()

    return wandb_module.Table(
        data=[
            [i, pre_cpu[i, 0].item(), post_cpu[i, 0].item()]
            for i in range(pre_cpu.shape[0])
        ],
        columns=["sample_idx", "pre_clip_grad_norm", "post_clip_grad_norm"],
    )


def _log_gradient_norms(
    wandb_module,
    logical_batch_pre_clip_sq_norms: torch.Tensor | None,
    logical_step: int,
    epoch: int,
    config: TrainConfig,
):
    if wandb_module is None or not config.log_batch_gradients:
        return

    if logical_batch_pre_clip_sq_norms is None or logical_batch_pre_clip_sq_norms.numel() == 0:
        return

    pre_clip_norms = logical_batch_pre_clip_sq_norms.sqrt().unsqueeze(1)   # [batch_size, 1]
    post_clip_norms = _compute_post_clip_norms(pre_clip_norms, config.max_grad_norm)

    pre_np = pre_clip_norms.squeeze(1).detach().cpu().numpy()
    post_np = post_clip_norms.squeeze(1).detach().cpu().numpy()

    payload = {
        "epoch": epoch,
        "gradient/logical_step": logical_step,
        "gradient/pre_clip_mean": pre_clip_norms.mean().item(),
        "gradient/pre_clip_std": pre_clip_norms.std(unbiased=False).item(),
        "gradient/pre_clip_max": pre_clip_norms.max().item(),
        "gradient/post_clip_mean": post_clip_norms.mean().item(),
        "gradient/post_clip_std": post_clip_norms.std(unbiased=False).item(),
        "gradient/post_clip_max": post_clip_norms.max().item(),
        "gradient/clipping_rate": (pre_clip_norms > float(config.max_grad_norm)).float().mean().item(),
        "gradient/pre_clip_histogram": wandb_module.Histogram(pre_np),
        "gradient/post_clip_histogram": wandb_module.Histogram(post_np),
    }

    wandb_module.log(payload)

def _evaluate_and_log_per_sample_loss(
    model,
    dataloader,
    device,
    wandb_module,
    epoch: int,
    config: TrainConfig,
):
    """
    Evaluate model and log per-sample losses to wandb.
    """
    if wandb_module is None or not config.log_batch_gradients:
        return None
    
    model.eval()
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval per-sample loss"):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            
            # Get per-sample losses (assuming reduction='none' or we can compute it)
            # If the model doesn't support it, we need to recompute
            logits = outputs.logits
            labels = batch["labels"]
            
            # Compute per-sample cross-entropy loss
            print(torch.max(logits))
            print(torch.min(logits))
            logits = torch.clamp(logits, -1,1)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_sample_losses = loss_fct(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1)
            )
            
            all_losses.append(per_sample_losses.detach().cpu())
    
    # Concatenate all losses
    all_losses = torch.cat(all_losses, dim=0)
    
    # Log statistics
    payload = {
        "epoch": epoch,
        "eval/per_sample_loss_mean": all_losses.mean().item(),
        "eval/per_sample_loss_std": all_losses.std(unbiased=False).item(),
        "eval/per_sample_loss_min": all_losses.min().item(),
        "eval/per_sample_loss_max": all_losses.max().item(),
        "eval/per_sample_loss_median": all_losses.median().item(),
        "eval/per_sample_loss_histogram": wandb_module.Histogram(all_losses.numpy()),
    }
    
    wandb_module.log(payload)
    
    # Create a table with per-sample losses
    loss_table = wandb_module.Table(
        data=[[i, all_losses[i].item()] for i in range(len(all_losses))],
        columns=["sample_idx", "loss"]
    )
    wandb_module.log({f"eval/per_sample_losses_epoch_{epoch}": loss_table})
    
    model.train()
    return all_losses

def train_private(model, dataloaders: dict, config: TrainConfig):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = evaluate.load("glue", config.task) if config.run_eval else None
    wandb = _maybe_init_wandb(config)

    print(
        "Warning: logging pre-clip gradients to W&B invalidates any meaningful differential privacy guarantee. "
        "This run should be treated as gradient analysis, not private training."
    )
    model.train()
    model = PrivacyEngine.get_compatible_module(model)
    model.to(device)

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # privacy_engine = PrivacyEngine(
    #     accountant=config.dp_accountant,
    #     secure_mode=config.dp_secure_mode,
    # )
    # model, optimizer, train_dataloader = privacy_engine.make_private(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=dataloaders["train_loader"],
    #     noise_multiplier=config.noise_multiplier,
    #     max_grad_norm=config.max_grad_norm,
    #     poisson_sampling=config.dp_poisson_sampling,
    # )

    train_dataloader = dataloaders["train_loader"]
    EPOCHS = 3
    LOGGING_INTERVAL = 2000 
    EPSILON = 7.5
    DELTA = 1 / len(train_dataloader)

    MAX_GRAD_NORM = 100000.0  # effectively no clipping, to analyze raw gradient norms

    MAX_GRAD_NORM = config.max_grad_norm
    # privacy_engine = PrivacyEngine(accountant=config.dp_accountant)
    lr = config.learning_rate
    optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, eps=1e-8, )

    trainable_layers = [model.roberta.encoder.layer[-1]]
    # trainable_layers = [model.classifier]
    exclude_params = set(model.classifier.parameters())
    exclude_params = set()

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
    print(f"Trainable parameters count: {trainable_params:,}") # ~7M

    # model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=train_dataloader,
    #     target_delta=DELTA,
    #     target_epsilon=EPSILON,
    #     epochs=EPOCHS,
    #     max_grad_norm=MAX_GRAD_NORM,
    # )

    privacy_engine = PrivacyEngine(
        accountant=config.dp_accountant,
        secure_mode=config.dp_secure_mode,
    )
    model, optimizer, train_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        noise_multiplier=config.noise_multiplier,
        max_grad_norm=MAX_GRAD_NORM,
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

        # stores squared norms for all samples belonging to the current logical batch
        logical_batch_pre_clip_sq_norms = None

        with BatchMemoryManager(
            data_loader=train_dataloader,
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

                # collect pre-clip per-sample squared norms for this physical batch,
                # append them to the current logical batch
                logical_batch_pre_clip_sq_norms = _accumulate_logical_batch_per_sample_sq_norms(
                    optimizer=optimizer,
                    accumulated_sq_norms=logical_batch_pre_clip_sq_norms,
                    exclude_params = exclude_params
                )

                # clip and accumulate inside Opacus, but do not update weights unless configured
                optimizer.clip_and_accumulate()

                skip_next_step = optimizer._check_skip_next_step()
                optimizer._is_last_step_skipped = skip_next_step

                if not skip_next_step:
                    logical_step += 1
                    _log_gradient_norms(
                        wandb_module=wandb,
                        logical_batch_pre_clip_sq_norms=logical_batch_pre_clip_sq_norms,
                        logical_step=logical_step,
                        epoch=epoch + 1,
                        config=config,
                    )

                    if config.update_weights:
                        optimizer.add_noise()
                        optimizer.scale_grad()
                        optimizer.original_optimizer.step()

                    logical_batch_pre_clip_sq_norms = None

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

            _evaluate_and_log_per_sample_loss(
                model=model,
                dataloader=dataloaders["eval_loader"],
                device=device,
                wandb_module=wandb,
                epoch=epoch + 1,
                config=config,
            )

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
    
    eval_metrics = evaluate_model(
        model=model,
        dataloader=dataloaders["test_loader"],
        metric=metric,
        device=device,
        task=config.task,
    )
    # print("epsilon after training:", privacy_engine.get_epsilon(DELTA))
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