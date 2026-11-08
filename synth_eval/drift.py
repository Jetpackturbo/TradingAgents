"""True-drift panel for ramp-schedule worlds (e.g. ``uwtc_resting_drift``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluator import Split


def drift_panel(sp: Split, bar_minutes: int = 5) -> pd.DataFrame:
    """Per-bar log-drift of the target F, aligned to ``sp.close`` (constant per symbol in a ramp world)."""
    cols = {}
    for sym in sp.symbols:
        t = pd.read_parquet(sp.path / "truth" / f"{sym}_1m.parquet")
        t.index = pd.to_datetime(t.index)
        f = t["fair"].astype(float).resample(f"{bar_minutes}min", label="right", closed="right").last()
        cols[sym] = np.log(f).diff()
    return pd.DataFrame(cols).reindex(sp.close.index)
