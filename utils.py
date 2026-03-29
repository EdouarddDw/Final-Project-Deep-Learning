import random
import torch

#------------------------------
#data.py utils
#-------------------------

def dataset_to_row_dict(dataset: Dataset) -> Dict[str, Dict[str, Any]]:
    """
    Convert a Hugging Face dataset split into a dictionary indexed by message_id.
    """
    row_dict = {}
    for row in dataset:
        row_dict[row["message_id"]] = row
    return row_dict


def build_children_index(dataset: Dataset) -> Dict[Optional[str], List[str]]:
    """
    Build parent_id -> [child_message_id, ...] mapping.
    """
    children: Dict[Optional[str], List[str]] = {}

    for row in dataset:
        parent_id = row.get("parent_id")
        msg_id = row["message_id"]
        children.setdefault(parent_id, []).append(msg_id)
    return children


def trace_path_to_root(
    message_id: str,
    row_dict: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Return the path from root to the given node.
    """
    path = []
    current_id = message_id

    while current_id is not None:
        row = row_dict[current_id]
        path.append(row)
        current_id = row.get("parent_id")

    path.reverse()
    return path

#-----------------------
#train.py utils
#-------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def global_grad_diagnostics(model: torch.nn.Module) -> tuple[float, float]:
    """
    Returns:
        grad_norm: global L2 norm of gradients
        grad_maxabs: largest absolute gradient entry
    """
    total_norm_sq = 0.0
    maxabs_overall = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        l2 = g.norm(2).item()
        total_norm_sq += l2 ** 2
        maxabs = g.abs().max().item()
        if maxabs > maxabs_overall:
            maxabs_overall = maxabs
    return math.sqrt(total_norm_sq), maxabs_overall


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def decode_sample(tokenizer, token_ids: torch.Tensor) -> str:
    ids = token_ids.detach().cpu().tolist()
    return tokenizer.decode(ids, skip_special_tokens=False)



def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf 8") as f:
        json.dump(obj, f, indent=2)


