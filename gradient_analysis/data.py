from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from .config import TrainConfig
from .tasks import TASK_TO_KEYS, get_validation_split, is_regression_task


def _tokenize_dataset(raw_dataset, tokenizer, config: TrainConfig):
    sentence1_key, sentence2_key = TASK_TO_KEYS[config.task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                truncation=True,
                max_length=config.max_length,
            )
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
            max_length=config.max_length,
        )

    dataset = raw_dataset.map(preprocess_function, batched=True)

    columns_to_remove = ["idx", sentence1_key]
    if sentence2_key is not None:
        columns_to_remove.append(sentence2_key)

    existing_columns = [col for col in columns_to_remove if col in dataset["train"].column_names]
    dataset = dataset.remove_columns(existing_columns)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")
    return dataset


def create_dataloaders(tokenizer, config: TrainConfig):
    raw_dataset = load_dataset("glue", config.task)
    tokenized_dataset = _tokenize_dataset(raw_dataset, tokenizer, config)

    validation_split = get_validation_split(config.task)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=config.num_workers,
    )
    eval_loader = DataLoader(
        tokenized_dataset[validation_split],
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=config.num_workers,
    )

    if is_regression_task(config.task):
        num_labels = 1
    else:
        num_labels = raw_dataset["train"].features["label"].num_classes

    return {
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "num_labels": num_labels,
        "train_size": len(tokenized_dataset["train"]),
        "eval_size": len(tokenized_dataset[validation_split]),
    }
