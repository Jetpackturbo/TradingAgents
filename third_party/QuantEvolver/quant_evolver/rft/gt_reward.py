"""Ground-truth reward: trains on the evaluator's proxy, logs gold next to it (never used for the update).

The evaluator is any callable ``ev(expr, start_date, end_date) -> dict | None`` returning at least
``proxy``, ``se`` and ``gold``; ``None`` means the expression is invalid.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from quant_evolver.dsl.compiler import extract_expr

EXTRA_KEYS = ("expr", "valid", "proxy", "proxy_se", "gold", "gold_se", "mispricing_corr")

_WORKER_EV = None


def _score_in_worker(args):
    expr, start, end = args
    return _WORKER_EV(expr, start, end)


def load_evaluator(spec: str, kwargs: dict):
    """``spec`` is ``"package.module:factory"``; the factory is called with ``kwargs``."""
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)(**kwargs)


class GroundTruthReward:
    def __init__(self, tokenizer, evaluator, invalid_score: float = -1.0, workers: int = 1):
        self.tok = tokenizer
        self.ev = evaluator
        self.invalid = float(invalid_score)
        self.workers = int(workers)
        self._pool = None
        self._memo: dict[tuple, dict | None] = {}

    def _get_pool(self):
        # Workers are forked with the evaluator already loaded, so it is never pickled. Needs "fork" (Linux).
        if self._pool is None and self.workers > 1 and "fork" in mp.get_all_start_methods():
            global _WORKER_EV
            _WORKER_EV = self.ev
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("fork"))
        return self._pool

    def _evaluate(self, keys: list[tuple]) -> None:
        todo = [k for k in dict.fromkeys(keys) if k[0] and k not in self._memo]
        pool = self._get_pool() if len(todo) > 1 else None
        if pool is None:
            for k in todo:
                self._memo[k] = self.ev(*k)
        else:
            for k, out in zip(todo, pool.map(_score_in_worker, todo)):
                self._memo[k] = out

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
        keys = [(extract_expr(t), starts[i], ends[i]) for i, t in enumerate(texts)]
        self._evaluate(keys)
        extra = {k: [] for k in EXTRA_KEYS}
        scores = []
        for key in keys:
            e = key[0]
            out = self._memo.get(key) if e else None
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
