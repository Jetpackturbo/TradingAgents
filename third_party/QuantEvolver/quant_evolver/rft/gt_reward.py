"""Ground-truth reward: trains on the evaluator's proxy, logs gold next to it (never used for the update).

The evaluator is any callable ``ev(expr, start_date, end_date) -> dict | None`` returning at least
``proxy``, ``se`` and ``gold``; ``None`` means the expression is invalid.
"""

from __future__ import annotations

import importlib

import numpy as np
import torch

from quant_evolver.dsl.compiler import extract_expr

EXTRA_KEYS = ("expr", "valid", "proxy", "proxy_se", "gold", "gold_se", "mispricing_corr")


def load_evaluator(spec: str, kwargs: dict):
    """``spec`` is ``"package.module:factory"``; the factory is called with ``kwargs``."""
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)(**kwargs)


class GroundTruthReward:
    def __init__(self, tokenizer, evaluator, invalid_score: float = -1.0):
        self.tok = tokenizer
        self.ev = evaluator
        self.invalid = float(invalid_score)

    @staticmethod
    def _extra(data, key, default=None):
        ex = data.non_tensor_batch.get("extra_info")
        if ex is None:
            return [default] * len(data)
        return [(e or {}).get(key, default) for e in ex]

    def _decode(self, data) -> list[str]:
        resp = data.batch["responses"]
        mask = data.batch["attention_mask"][:, data.batch["prompts"].shape[-1]:]
        return [self.tok.decode(resp[i][mask[i].bool()].tolist(), skip_special_tokens=True) for i in range(resp.shape[0])]

    def __call__(self, data, return_dict: bool = False):
        R = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        plen = data.batch["prompts"].shape[-1]
        rlen = data.batch["attention_mask"][:, plen:].sum(dim=1)
        texts = self._decode(data)
        starts, ends = self._extra(data, "start_date"), self._extra(data, "end_date")
        extra = {k: [] for k in EXTRA_KEYS}
        scores = []
        for i, t in enumerate(texts):
            e = extract_expr(t)
            out = self.ev(e, starts[i], ends[i]) if e else None
            ok = out is not None
            s = float(out["proxy"]) if ok else self.invalid
            scores.append(s)
            extra["expr"].append(e)
            extra["valid"].append(float(ok))
            extra["proxy"].append(s)
            extra["proxy_se"].append(float(out["se"]) if ok else 0.0)
            extra["gold"].append(float(out.get("gold", np.nan)) if ok else np.nan)
            extra["gold_se"].append(float(out.get("gold_se", np.nan)) if ok else np.nan)
            extra["mispricing_corr"].append(float(out.get("mispricing_corr", np.nan)) if ok else np.nan)
        R[torch.arange(len(scores)), (rlen - 1).clamp(min=0)] = torch.tensor(scores, dtype=torch.float32)
        return {"reward_tensor": R, "reward_extra_info": extra} if return_dict else R
