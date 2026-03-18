import argparse

from gradient_analysis import (
    TrainConfig,
    build_model_and_tokenizer,
    build_tokenizer,
    create_dataloaders,
    train,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GLUE classifier with a custom PyTorch loop.")
    parser.add_argument("--task", type=str, default="mrpc")
    parser.add_argument("--model-name", type=str, default="FacebookAI/roberta-base")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-every", type=int, default=20)
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
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        grad_accumulation_steps=args.grad_accumulation_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        log_every=args.log_every,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    print(f"Preparing task={config.task}, model={config.model_name}, use_lora={config.use_lora}")
    tokenizer = build_tokenizer(config)
    dataloaders = create_dataloaders(tokenizer, config)
    model, tokenizer = build_model_and_tokenizer(config, num_labels=dataloaders["num_labels"])

    results = train(model=model, dataloaders=dataloaders, config=config)
    tokenizer.save_pretrained(results["saved_model_path"])
    print(results)


if __name__ == "__main__":
    main()
