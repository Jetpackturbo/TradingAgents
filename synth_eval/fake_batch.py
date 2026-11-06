"""Build verl ``DataProto`` batches from plain response strings, for testing reward functions without a model."""

from __future__ import annotations

import numpy as np
import torch


class CharTokenizer:
    """Byte-level stand-in for a HF tokenizer: only ``decode``/``batch_decode``/``pad_token_id`` are used by rewards."""

    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [b + 1 for b in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return bytes(int(i) - 1 for i in ids if int(i) != self.pad_token_id).decode("utf-8", errors="ignore")

    def batch_decode(self, rows, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(r.tolist() if hasattr(r, "tolist") else r) for r in rows]


def make_batch(responses: list[str], extra_infos: list[dict], uids: list[str] | None = None,
               prompt_len: int = 8, tokenizer: CharTokenizer | None = None):
    """Right-padded responses after a fixed-length prompt, laid out the way verl's rollout returns them."""
    from verl.protocol import DataProto

    tok = tokenizer or CharTokenizer()
    enc = [tok.encode(r) for r in responses]
    rlen = max(len(e) for e in enc)
    n = len(responses)
    resp = torch.zeros((n, rlen), dtype=torch.long)
    resp_mask = torch.zeros((n, rlen), dtype=torch.long)
    for i, e in enumerate(enc):
        resp[i, : len(e)] = torch.tensor(e)
        resp_mask[i, : len(e)] = 1
    prompts = torch.ones((n, prompt_len), dtype=torch.long)
    attn = torch.cat([torch.ones((n, prompt_len), dtype=torch.long), resp_mask], dim=1)
    non_tensors = {"extra_info": np.array(extra_infos, dtype=object)}
    if uids is not None:
        non_tensors["uid"] = np.array(uids, dtype=object)
    return DataProto.from_dict(
        tensors={"prompts": prompts, "responses": resp, "response_mask": resp_mask, "attention_mask": attn},
        non_tensors=non_tensors,
    ), [len(e) for e in enc]
