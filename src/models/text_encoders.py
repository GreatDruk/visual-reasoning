"""Text encoders. Each returns (sentence [B, D], token_states [B, L, D], mask [B, L])."""

from collections import Counter

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    """Whitespace word vocabulary, built from train(+val) captions. Test captions add no
    words (all 55 tokens of test occur in train), so including them would change nothing."""

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


class ClipTokenizer:
    """Fixed pretrained tokenizer with Vocab's batching/serialization interface.

    Remember raw captions for the encoder's feature cache. Captions with identical
    pretrained ids share a cache key; no vocabulary is fitted on supplied captions.
    """

    def __init__(self, model_name, cache_dir=None):
        import open_clip

        self.tokenizer = open_clip.get_tokenizer(model_name, cache_dir=cache_dir)
        self.itos = self.tokenizer.tokenizer.convert_ids_to_tokens(range(self.tokenizer.vocab_size))
        self.lowercase = True  # SigLIP2's pretrained canonicalize cleaner lowercases.
        self._encoded = {}
        self._caption_by_ids = {}

    def __len__(self):
        return len(self.itos)

    def encode_batch(self, captions):
        missing = list(dict.fromkeys(c for c in captions if c not in self._encoded))
        if missing:
            ids, mask = self.tokenizer(missing, output_mask=True)
            for caption, row, length in zip(missing, ids, mask.sum(dim=1).tolist()):
                self._encoded[caption] = (row.clone(), length)
                self._caption_by_ids.setdefault(tuple(row.tolist()), caption)
        ids = torch.stack([self._encoded[c][0] for c in captions])
        lengths = torch.tensor([self._encoded[c][1] for c in captions], dtype=torch.long)
        return ids, lengths

    def captions_for(self, tokens):
        return [self._caption_by_ids[tuple(row)] for row in tokens.cpu().tolist()]


class ClipTextEncoder(nn.Module):
    """Frozen SigLIP text tower; only an optional output projection is trainable.

    Full padded inputs preserve pretrained bidirectional attention and last-position
    pooling. Cache detached outputs on CPU by caption, trimming token states to their
    real length (including EOS). The trainable projection always runs after retrieval.
    """

    def __init__(self, model_name, pretrained, freeze=True, projection_dim=None, cache_dir=None):
        super().__init__()
        import open_clip

        if not freeze:
            raise NotImplementedError("ClipTextEncoder is frozen-only (text features are cached)")
        model = open_clip.create_model(model_name, pretrained=pretrained, cache_dir=cache_dir, device="cpu")
        self.tower = model.text.eval()  # discard the image tower
        self.tower.output_tokens = True
        self.tower.requires_grad_(False)
        self.tokenizer = ClipTokenizer(model_name, cache_dir=cache_dir)
        self.out_dim = projection_dim if projection_dim is not None else self.tower.width
        self.projection = nn.Identity() if projection_dim is None else nn.Linear(self.tower.width, projection_dim)
        self._cache = {}

    def train(self, mode=True):
        super().train(mode)
        self.tower.eval()
        return self

    def forward(self, tokens, lengths):
        captions = self.tokenizer.captions_for(tokens)
        missing = {}
        for i, caption in enumerate(captions):
            if caption not in self._cache:
                missing.setdefault(caption, i)
        if missing:
            rows = list(missing.values())
            with torch.no_grad():
                sentence, states = self.tower(tokens[rows])
            for j, (caption, i) in enumerate(missing.items()):
                self._cache[caption] = (
                    sentence[j].detach().cpu().clone(),
                    states[j, : int(lengths[i])].detach().cpu().clone(),
                )
        cached = [self._cache[c] for c in captions]
        sentence = torch.stack([s for s, _ in cached]).to(tokens.device)
        states = pad_sequence([t for _, t in cached], batch_first=True).to(tokens.device)
        mask = torch.arange(states.shape[1], device=tokens.device)[None, :] < lengths.to(tokens.device)[:, None]
        return self.projection(sentence), self.projection(states), mask
