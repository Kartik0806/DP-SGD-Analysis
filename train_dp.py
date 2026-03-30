import argparse

from gradient_analysis import (
    TrainConfig,
    build_model_and_tokenizer,
    build_tokenizer,
    create_dataloaders,
    train_private,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GLUE classifier with Opacus differential privacy and BatchMemoryManager."
    )
    parser.add_argument("--task", type=str, default="mrpc")
    parser.add_argument("--model-name", type=str, default="FacebookAI/roberta-base")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-physical-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-delta", type=float, default=None)
    parser.add_argument("--dp-accountant", type=str, default="prv")
    parser.add_argument("--secure-mode", action="store_true")
    parser.add_argument("--disable-poisson-sampling", action="store_true")
    parser.add_argument("--update-weights", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--disable-gradient-logging", action="store_true")
    parser.add_argument("--gradient-histogram-max-values", type=int, default=16384)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        task=args.task,
        model_name=args.model_name,
        use_lora=args.use_lora,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        log_every=args.log_every,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        target_delta=args.target_delta,
        max_physical_batch_size=args.max_physical_batch_size,
        dp_accountant=args.dp_accountant,
        dp_secure_mode=args.secure_mode,
        dp_poisson_sampling=not args.disable_poisson_sampling,
        update_weights=args.update_weights,
        run_eval=args.run_eval,
        log_batch_gradients=not args.disable_gradient_logging,
        gradient_histogram_max_values=args.gradient_histogram_max_values,
    )

    print(
        "Preparing DP training "
        f"task={config.task}, model={config.model_name}, use_lora={config.use_lora}, "
        f"logical_batch_size={config.batch_size}, max_physical_batch_size={config.max_physical_batch_size}, "
        f"update_weights={config.update_weights}"
    )

    tokenizer = build_tokenizer(config)
    dataloaders = create_dataloaders(tokenizer, config)
    model, tokenizer = build_model_and_tokenizer(config, num_labels=dataloaders["num_labels"])

    results = train_private(model=model, dataloaders=dataloaders, config=config)
    tokenizer.save_pretrained(results["saved_model_path"])
    print(results)


if __name__ == "__main__":
    main()
