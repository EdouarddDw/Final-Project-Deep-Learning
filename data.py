
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from utils import dataset_to_row_dict, build_children_index, trace_path_to_root  

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

        messages.append({
            "role": role,
            "content": text.strip()
        })

    # Basic sanity checks
    if len(messages) < 2:
        return None

    if messages[0]["role"] != "user":
        return None

    if messages[-1]["role"] != "assistant":
        return None

    # Make sure turns alternate user / assistant / user / assistant...
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

        examples.append({
            "messages": messages,
            "target_message_id": row["message_id"]
        })

    return examples


# =========================================================
# 4. Chat formatting
# =========================================================

def format_chat(messages: List[Dict[str, str]]) -> str:
    """
    Turn structured messages into one flat training string.
    """
    chunks = []

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


def get_tokenizer(
    model_name: str,
    add_chat_tokens: bool = True
):
    """
    Load tokenizer and optionally add chat special tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if add_chat_tokens:
        special_tokens = {
            "additional_special_tokens": [
                "<|user|>",
                "<|assistant|>",
            ]
        }
        tokenizer.add_special_tokens(special_tokens)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# =========================================================
# 5. Tokenization with assistant-only loss
# =========================================================

def tokenize_messages(
    messages: List[Dict[str, str]],
    tokenizer,
    max_length: int = 1024,
    train_on_all_assistant_tokens: bool = True,
) -> Dict[str, List[int]]:
    """
    Tokenize a conversation and create labels.

    labels:
    - assistant tokens => same as input_ids
    - user tokens => -100
    """
    input_ids: List[int] = []
    labels: List[int] = []

    model_limit = min(max_length, tokenizer.model_max_length)

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"].strip()
        text = f"<|{role}|>\n{content}\n"

        remaining = model_limit - len(input_ids)
        if remaining <= 0:
            break

        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=remaining,
        )["input_ids"]

        if not ids:
            continue

        input_ids.extend(ids)

        if role == "assistant":
            if train_on_all_assistant_tokens or i == len(messages) - 1:
                labels.extend(ids)
            else:
                labels.extend([-100] * len(ids))
        else:
            labels.extend([-100] * len(ids))

    attention_mask = [1] * len(input_ids)

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
) -> Dict[str, Any]:
    """
    Tokenize a single extracted example.
    """
    tokenized = tokenize_messages(
        messages=example["messages"],
        tokenizer=tokenizer,
        max_length=max_length,
        train_on_all_assistant_tokens=train_on_all_assistant_tokens,
    )

    return tokenized


def build_tokenized_dataset(
    examples: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 1024,
    train_on_all_assistant_tokens: bool = True,
) -> List[Dict[str, Any]]:
    """
    Tokenize all examples into a plain Python list.
    """
    tokenized_examples = []

    for ex in examples:
        tokenized = tokenize_example(
            ex,
            tokenizer=tokenizer,
            max_length=max_length,
            train_on_all_assistant_tokens=train_on_all_assistant_tokens,
        )
        tokenized_examples.append(tokenized)

    return tokenized_examples


# =========================================================
# 6. Collator for dynamic padding
# =========================================================

@dataclass
class ChatCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])

            batch_input_ids.append(
                f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            )
            batch_attention_mask.append(
                f["attention_mask"] + [0] * pad_len
            )
            batch_labels.append(
                f["labels"] + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# =========================================================
# 7. High level helper
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
    #sanity checks making sure everything works.
    bundle = prepare_oasst1_for_sft(
        model_name="gpt2",
        max_length=1024,
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

    print(preview_example(bundle["train_examples"][0]))
    print(batch["input_ids"].shape)
    print(batch["attention_mask"].shape)
    print(batch["labels"].shape)
    print(max(len(x["input_ids"]) for x in bundle["train_tokenized"]))

if __name__ == "__main__":
    main() 
