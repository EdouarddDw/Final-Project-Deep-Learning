from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from utils import dataset_to_row_dict, trace_path_to_root

IGNORE_INDEX = -100


# =========================================================
# 1. Loading raw dataset
# =========================================================

def load_oasst1() -> Dict[str, Dataset]:
    """
    Load the OpenAssistant OASST1 dataset.
    Returns a dict-like object with train and validation splits.
    """
    return load_dataset("OpenAssistant/oasst1")


# =========================================================
# 2. Tree -> conversation path examples
# =========================================================

def normalize_role(raw_role: str) -> str:
    """
    Normalize dataset roles into the format we want.
    """
    raw_role = raw_role.lower().strip()
    if raw_role == "prompter":
        return "user"
    if raw_role == "assistant":
        return "assistant"
    return raw_role


def path_to_messages(path: List[Dict[str, Any]]) -> Optional[List[Dict[str, str]]]:
    """
    Convert a raw path into a clean list of chat messages.

    Returns None if the path is malformed.
    """
    messages: List[Dict[str, str]] = []

    for node in path:
        text = node.get("text")
        role = normalize_role(node.get("role", ""))

        if not text or role not in {"user", "assistant"}:
            return None

        content = text.strip()
        if not content:
            return None

        messages.append({"role": role, "content": content})

    if len(messages) < 2:
        return None
    if messages[0]["role"] != "user":
        return None
    if messages[-1]["role"] != "assistant":
        return None

    for i in range(1, len(messages)):
        if messages[i]["role"] == messages[i - 1]["role"]:
            return None

    return messages


def extract_supervised_examples(dataset: Dataset) -> List[Dict[str, Any]]:
    """
    Build root-to-assistant examples.
    One training example per assistant node.
    """
    row_dict = dataset_to_row_dict(dataset)
    examples: List[Dict[str, Any]] = []

    for row in dataset:
        if normalize_role(row.get("role", "")) != "assistant":
            continue

        path = trace_path_to_root(row["message_id"], row_dict)
        messages = path_to_messages(path)
        if messages is None:
            continue

        examples.append(
            {
                "messages": messages,
                "target_message_id": row["message_id"],
            }
        )

    return examples


# =========================================================
# 3. Chat formatting
# =========================================================

def format_chat(messages: List[Dict[str, str]]) -> str:
    """
    Turn structured messages into one flat training string.
    """
    chunks: List[str] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()

        if role == "user":
            chunks.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            chunks.append(f"<|assistant|>\n{content}\n")
        else:
            raise ValueError(f"Unknown role: {role}")

    return "".join(chunks)


def get_tokenizer(model_name: str, add_chat_tokens: bool = True):
    """
    Load tokenizer and optionally add chat special tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if add_chat_tokens:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|user|>",
                    "<|assistant|>",
                ]
            }
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# =========================================================
# 4. Tokenization with assistant-only loss
# =========================================================

def count_valid_targets(labels: List[int]) -> int:
    return sum(1 for x in labels if x != IGNORE_INDEX)


def _tokenize_text(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def tokenize_messages(
    messages: List[Dict[str, str]],
    tokenizer,
    max_length: int = 1024,
    train_on_all_assistant_tokens: bool = True,
) -> Optional[Dict[str, List[int]]]:
    """
    Tokenize a conversation and create labels.

    Supervision policy:
    - user turn tokens -> ignored
    - assistant role marker -> ignored
    - assistant content tokens -> supervised when enabled

    Returns None if the tokenized example has no valid targets after truncation.
    """
    model_limit = min(max_length, tokenizer.model_max_length)

    input_ids: List[int] = []
    labels: List[int] = []

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"].strip()

        prefix_ids = _tokenize_text(tokenizer, f"<|{role}|>\n")
        content_ids = _tokenize_text(tokenizer, f"{content}\n")

        if role == "assistant":
            supervise_assistant = train_on_all_assistant_tokens or i == len(messages) - 1
        else:
            supervise_assistant = False

        segments = [
            (prefix_ids, False),
            (content_ids, supervise_assistant),
        ]

        for seg_ids, supervise in segments:
            if not seg_ids:
                continue

            remaining = model_limit - len(input_ids)
            if remaining <= 0:
                break

            kept_ids = seg_ids[:remaining]
            input_ids.extend(kept_ids)

            if supervise:
                labels.extend(kept_ids)
            else:
                labels.extend([IGNORE_INDEX] * len(kept_ids))

        if len(input_ids) >= model_limit:
            break

    if not input_ids:
        return None

    attention_mask = [1] * len(input_ids)

    if count_valid_targets(labels) == 0:
        return None

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def tokenize_example(
    example: Dict[str, Any],
    tokenizer,
    max_length: int = 1024,
    train_on_all_assistant_tokens: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Tokenize a single extracted example.
    Returns None if the example contains no valid supervised targets.
    """
    return tokenize_messages(
        messages=example["messages"],
        tokenizer=tokenizer,
        max_length=max_length,
        train_on_all_assistant_tokens=train_on_all_assistant_tokens,
    )


def build_tokenized_dataset(
    examples: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 1024,
    train_on_all_assistant_tokens: bool = True,
) -> List[Dict[str, Any]]:
    """
    Tokenize all examples into a plain Python list.
    Drop examples with zero valid targets after truncation.
    """
    tokenized_examples: List[Dict[str, Any]] = []
    dropped_zero_target = 0

    for ex in examples:
        tokenized = tokenize_example(
            ex,
            tokenizer=tokenizer,
            max_length=max_length,
            train_on_all_assistant_tokens=train_on_all_assistant_tokens,
        )

        if tokenized is None:
            dropped_zero_target += 1
            continue

        tokenized_examples.append(tokenized)

    print(f"Dropped {dropped_zero_target} examples with zero valid targets.")
    return tokenized_examples


# =========================================================
# 5. Collator for dynamic padding
# =========================================================

@dataclass
class ChatCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Received an empty batch in ChatCollator.")

        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])

            batch_input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            batch_attention_mask.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [IGNORE_INDEX] * pad_len)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

        valid_target_count = int((batch["labels"] != IGNORE_INDEX).sum().item())
        if valid_target_count == 0:
            raise ValueError("ChatCollator produced a batch with zero valid targets.")

        return batch


# =========================================================
# 6. High level helper
# =========================================================

def prepare_oasst1_for_sft(
    model_name: str = "gpt2",
    max_length: int = 1024,
    train_on_all_assistant_tokens: bool = True,
):
    """
    Full pipeline:
    - load raw data
    - extract root-to-assistant examples
    - create tokenizer
    - tokenize train and validation
    - drop tokenized examples with zero valid targets
    """
    data = load_oasst1()

    train_examples = extract_supervised_examples(data["train"])
    val_examples = extract_supervised_examples(data["validation"])

    tokenizer = get_tokenizer(model_name)

    train_tokenized = build_tokenized_dataset(
        train_examples,
        tokenizer,
        max_length=max_length,
        train_on_all_assistant_tokens=train_on_all_assistant_tokens,
    )
    val_tokenized = build_tokenized_dataset(
        val_examples,
        tokenizer,
        max_length=max_length,
        train_on_all_assistant_tokens=train_on_all_assistant_tokens,
    )

    return {
        "tokenizer": tokenizer,
        "train_examples": train_examples,
        "val_examples": val_examples,
        "train_tokenized": train_tokenized,
        "val_tokenized": val_tokenized,
    }


def create_dataloaders(
    train_tokenized: List[Dict[str, Any]],
    val_tokenized: List[Dict[str, Any]],
    tokenizer,
    batch_size: int = 4,
    num_workers: int = 0,
):
    """
    Create PyTorch dataloaders.
    """
    if len(train_tokenized) == 0:
        raise ValueError("No train examples remain after tokenization/filtering.")
    if len(val_tokenized) == 0:
        raise ValueError("No validation examples remain after tokenization/filtering.")

    collator = ChatCollator(tokenizer)

    train_loader = DataLoader(
        train_tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_tokenized,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def preview_example(example: Dict[str, Any]) -> None:
    for msg in example["messages"]:
        print(f"[{msg['role'].upper()}]")
        print(msg["content"])
        print()


def main():
    bundle = prepare_oasst1_for_sft(
        model_name="gpt2",
        max_length=256,
        train_on_all_assistant_tokens=True,
    )

    tokenizer = bundle["tokenizer"]
    train_loader, val_loader = create_dataloaders(
        bundle["train_tokenized"],
        bundle["val_tokenized"],
        tokenizer=tokenizer,
        batch_size=2,
    )

    batch = next(iter(train_loader))

    preview_example(bundle["train_examples"][0])
    print(batch["input_ids"].shape)
    print(batch["attention_mask"].shape)
    print(batch["labels"].shape)
    print(max(len(x["input_ids"]) for x in bundle["train_tokenized"][: min(1000, len(bundle["train_tokenized"]))]))


if __name__ == "__main__":
    main()

