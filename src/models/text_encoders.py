"""Text encoders. Each returns (sentence [B, D], token_states [B, L, D], mask [B, L])."""

from collections import Counter

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    """Whitespace word vocabulary. Built from train(+val) captions only (rules: test
    captions must not be used to fit preprocessing)."""

    def __init__(self, captions, lowercase=True, min_freq=1):
        self.lowercase = lowercase
        counts = Counter(w for c in captions for w in self.split(c))
        self.itos = [PAD, UNK] + sorted(w for w, n in counts.items() if n >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def split(self, caption):
        return (caption.lower() if self.lowercase else caption).split()

    def __len__(self):
        return len(self.itos)

    def encode_batch(self, captions):
        """Captions -> (LongTensor [B, L] padded with 0, lengths LongTensor [B])."""
        ids = [[self.stoi.get(w, 1) for w in self.split(c)] or [1] for c in captions]
        lengths = torch.tensor([len(x) for x in ids])
        out = torch.zeros(len(ids), int(lengths.max()), dtype=torch.long)
        for i, x in enumerate(ids):
            out[i, : len(x)] = torch.tensor(x)
        return out, lengths


class GRUEncoder(nn.Module):
    """Word embeddings + (bi)GRU trained from scratch, as the FiLM generator / MAC reader.

    The sentence vector is the concatenation of the last forward and the first backward
    hidden state; token states feed attention-based heads.
    """

    def __init__(self, vocab_size, embed_dim=200, hidden_dim=512, num_layers=1, bidirectional=True, dropout=0.1, lowercase=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, tokens, lengths):
        x = self.drop(self.embed(tokens))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=tokens.shape[1])
        if self.gru.bidirectional:
            sentence = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            sentence = h[-1]
        mask = tokens != 0
        return sentence, out, mask


class HFTextEncoder(nn.Module):
    """BERT-like encoder via transformers. Not installed here (no PyPI access)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise ImportError("HFTextEncoder needs transformers: uv add transformers")


class ClipTextEncoder(nn.Module):
    """CLIP text tower via open_clip. Not installed here (no PyPI access)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise ImportError("ClipTextEncoder needs open_clip_torch: uv add open_clip_torch")
