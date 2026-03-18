TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

PRIMARY_METRICS = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "f1",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "wnli": "accuracy",
    "stsb": "pearson",
    "mnli": "accuracy",
}


def is_regression_task(task: str) -> bool:
    return task == "stsb"


def get_validation_split(task: str) -> str:
    return "validation_matched" if task == "mnli" else "validation"
