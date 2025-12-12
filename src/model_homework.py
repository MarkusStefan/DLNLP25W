# model.py
"""
Minimal model and tokenizer utilities used by agent.py / the notebook.

Students must implement the neural architecture so that checkpoints
trained in the notebook load and decode with the same code here.

Keep the public API stable:

- SPECIAL_TOKENS : Dict[str, str]
- simple_tokenize(s: str) -> List[str]
- encode(tokens: List[str], stoi: Dict[str, int], add_sos_eos: bool=False) -> List[int]
- class Encoder(nn.Module): forward(src, src_lens)
- class Decoder(nn.Module): forward(tgt_in, hidden)
- class Seq2Seq(nn.Module):
    - forward(src, src_lens, tgt_in) -> logits [B,T,V]
    - greedy_decode(src, src_lens, max_len, sos_id, eos_id) -> LongTensor[B, max_len]
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn


# -------------------------
# Tokenization utilities
# -------------------------

SPECIAL_TOKENS = {
    "pad": "<pad>",
    "sos": "<sos>",
    "eos": "<eos>",
    "unk": "<unk>",
}


def simple_tokenize(s: str) -> List[str]:
    """Lowercase whitespace tokenizer used by both training and inference."""
    return s.strip().lower().split()


def encode(tokens: List[str], stoi: Dict[str, int], add_sos_eos: bool = False) -> List[int]:
    """Map tokens to ids using `stoi`. Optionally wrap with <sos>/<eos>."""
    ids = [stoi.get(t, stoi[SPECIAL_TOKENS["unk"]]) for t in tokens]
    if add_sos_eos:
        ids = [stoi[SPECIAL_TOKENS["sos"]]] + ids + [stoi[SPECIAL_TOKENS["eos"]]]
    return ids


# -------------------------
# Model scaffolding
# -------------------------

class Encoder(nn.Module):
    """
    Student-implemented encoder.
    Expected behavior:
      forward(src: LongTensor[B, S], src_lens: LongTensor[B]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
    Returns:
      - outputs: Tensor[B, S, H] (padded time-major outputs)
      - hidden:  RNN-style tuple (h, c) or similar state your decoder expects
    """
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        # TODO: define embeddings and encoder layers to match your notebook training
        raise NotImplementedError("Implement Encoder __init__.")

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor):
        # TODO: implement packed sequence handling and return (outputs, hidden)
        raise NotImplementedError("Implement Encoder.forward.")


class Decoder(nn.Module):
    """
    Student-implemented decoder.
    Expected behavior:
      forward(tgt_in: LongTensor[B, T], hidden) -> Tuple[Tensor, Any]
    Returns:
      - logits: Tensor[B, T, V] (distributions before softmax over target vocab)
      - hidden: updated recurrent state
    """
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        # TODO: define embeddings, recurrent layers, and output projection
        raise NotImplementedError("Implement Decoder __init__.")

    def forward(self, tgt_in: torch.Tensor, hidden):
        # TODO: return (logits, hidden)
        raise NotImplementedError("Implement Decoder.forward.")


class Seq2Seq(nn.Module):
    """
    Student-implemented Seq2Seq wrapper that ties Encoder and Decoder.

    Required methods:
      - forward(src, src_lens, tgt_in) -> logits [B, T, V]
      - greedy_decode(src, src_lens, max_len, sos_id, eos_id) -> LongTensor[B, max_len]
        Greedy decoding should stop at <eos> per sequence and pad remainder with <eos>.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        # TODO: encode, then decode with teacher forcing; return logits [B,T,V]
        raise NotImplementedError("Implement Seq2Seq.forward.")

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lens: torch.Tensor,
        max_len: int,
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """
        TODO: implement token-by-token greedy decoding.
        Must return LongTensor[B, max_len]. If <eos> is emitted at step t,
        set positions > t to <eos> for that sequence.
        """
        raise NotImplementedError("Implement Seq2Seq.greedy_decode.")
