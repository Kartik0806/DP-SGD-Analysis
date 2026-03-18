from dataclasses import dataclass


@dataclass
class TrainConfig:
    model_name: str = "FacebookAI/roberta-base"
    task: str = "mrpc"
    use_lora: bool = False
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 5e-4
    num_epochs: int = 5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.0
    max_length: int = 256
    grad_accumulation_steps: int = 1
    seed: int = 42
    num_workers: int = 0
    output_dir: str = "outputs"
    log_every: int = 20
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("query", "value")

    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    target_delta: float | None = None
    max_physical_batch_size: int = 128
    dp_accountant: str = "prv"
    dp_secure_mode: bool = False
    dp_poisson_sampling: bool = True
    update_weights: bool = False
    run_eval: bool = False
    log_batch_gradients: bool = True
    gradient_histogram_max_values: int = 16384
