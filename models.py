from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2


class CausalSelfAttentionHead(nn.Module):
    """
    One causal self attention head.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        returns: (B, T, head_size)
        """
        _, T, _ = x.shape

        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)

        att = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        v = self.value(x)  # (B, T, hs)
        out = att @ v      # (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi head causal self attention.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        head_size = n_embd // n_head

        self.heads = nn.ModuleList(
            [
                CausalSelfAttentionHead(
                    n_embd=n_embd,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_head)
            ]
        )

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """
    Standard transformer feed forward network.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre norm transformer block.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd=n_embd, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    """
    Small GPT style decoder only language model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx: (B, T)
        targets: (B, T) or None

        returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
        """
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )

        device = idx.device
        pos = torch.arange(T, device=device)

        tok_emb = self.token_embedding_table(idx)      # (B, T, C)
        pos_emb = self.position_embedding_table(pos)   # (T, C)

        x = tok_emb + pos_emb                          # (B, T, C)
        x = self.blocks(x)                             # (B, T, C)
        x = self.ln_f(x)                               # (B, T, C)
        logits = self.lm_head(x)                       # (B, T, vocab_size)

        loss = None
        if targets is not None:
            logits_flat = logits.reshape(B * T, self.config.vocab_size)
            targets_flat = targets.reshape(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        idx: (B, T)
        returns: (B, T + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature <= 0:
                raise ValueError("temperature must be > 0")

            logits = logits / temperature

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def build_model(
    vocab_size: int,
    block_size: int = 256,
    n_embd: int = 384,
    n_head: int = 6,
    n_layer: int = 6,
    dropout: float = 0.2,
) -> MiniLLM:
    """
    Convenience helper to build the model quickly.
    """
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    )
    return MiniLLM(config)
