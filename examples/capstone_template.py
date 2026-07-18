"""毕业项目实验记录骨架。先预注册，再填真实结果。"""

EXPERIMENT = {
    "data": "fixed TinyStories slice or fixed local text",
    "tokenizer": "same tokenizer for all variants",
    "training_tokens": None,
    "sequence_length": None,
    "dtype": "float32",
    "seeds": [7, 17, 27],
    "models": {
        "dense": {"total_params": None, "active_params": None},
        "attention_variant": {"total_params": None, "active_params": None},
        "moe": {"total_params": None, "active_params": None, "top_k": 2},
    },
    "metrics": ["validation_loss", "tokens_per_second", "peak_memory", "failure_cases"],
    "limitations": [],
}


if __name__ == "__main__":
    from pprint import pprint

    pprint(EXPERIMENT)

