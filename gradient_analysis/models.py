import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import TrainConfig
from .tasks import is_regression_task


def build_tokenizer(config: TrainConfig):
    return AutoTokenizer.from_pretrained(config.model_name)


def build_model_and_tokenizer(config: TrainConfig, num_labels: int):
    tokenizer = build_tokenizer(config)

    model_kwargs = {
        "num_labels": 1 if is_regression_task(config.task) else num_labels,
    }
    if is_regression_task(config.task):
        model_kwargs["problem_type"] = "regression"

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        **model_kwargs,
    )

    if config.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
        )
        before_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model = get_peft_model(model, lora_config)
        after_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        percent = round((after_lora / before_lora) * 100, 3)
        print(f"Before LoRA: {before_lora}")
        print(f"After LoRA: {after_lora}")
        print(f"Percentage of Trainable Parameters: {percent}")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable}")

    return model, tokenizer


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def unwrap_model(model):
    if hasattr(model, "_module"):
        return model._module
    if hasattr(model, "module"):
        return model.module
    return model
