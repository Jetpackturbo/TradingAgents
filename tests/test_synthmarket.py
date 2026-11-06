from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uwtc.bots.registry import BOT_TYPES  # noqa: E402

from synthmarket import WorldConfig, simulate_world  # noqa: E402
from synthmarket.bots import FlowMarketMaker, TargetTrader  # noqa: E402
from synthmarket.simulate import build_schedules, simulate_symbol  # noqa: E402

WORLDS = ROOT / "configs" / "worlds"


def tiny(name: str = "uwtc_informed", **upd) -> WorldConfig:
    w = WorldConfig.from_yaml(WORLDS / f"{name}.yaml")
    base = {**w.model_dump(), "n_symbols": 4, "days": 2, "bars_per_day": 60}
    return WorldConfig.model_validate({**base, **upd})


def read_dir(d: Path) -> dict[str, pd.DataFrame]:
    return {f"{sub}/{f.name}": pd.read_parquet(f) for sub in ("data", "truth")
            for f in sorted((d / sub).glob("*.parquet"))}


def test_bots_registered():
    assert BOT_TYPES["flow_market_maker"] is FlowMarketMaker
    assert BOT_TYPES["target_trader"] is TargetTrader


def test_determinism_and_parallel_equals_serial(tmp_path):
    w = tiny()
    s1 = simulate_world(w, 7, tmp_path / "a", workers=1)
    simulate_world(w, 7, tmp_path / "b", workers=1)
    simulate_world(w, 7, tmp_path / "c", workers=2)
    a, b, c = read_dir(tmp_path / "a"), read_dir(tmp_path / "b"), read_dir(tmp_path / "c")
    assert a.keys() == b.keys() == c.keys() and len(a) == 8
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])
        pd.testing.assert_frame_equal(a[k], c[k])
    assert s1["trades_total"] > 0
    simulate_world(w, 8, tmp_path / "d", workers=1)
    d = read_dir(tmp_path / "d")
    assert not a["data/S00_1m.parquet"].equals(d["data/S00_1m.parquet"])


def test_symbols_and_seeds_have_independent_bot_streams():
    w = tiny("uwtc_placebo")
    seeds = {w.round_seed(seed, s) for seed in (1, 2) for s in w.symbol_list}
    assert len(seeds) == 2 * len(w.symbol_list)
    r = [simulate_symbol(w, 1, s, 50.0, np.full(w.n_bars, 50.0)) for s in ("S00", "S01")]
    assert not np.array_equal(r[0]["bars"]["volume"], r[1]["bars"]["volume"])


def test_invariants_hold_at_end():
    w = tiny()
    p0, sched = build_schedules(w, 3)
    r = simulate_symbol(w, 3, "S00", p0["S00"], sched["S00"], return_engine=True)
    eng = r["engine"]
    assert eng.closed and r["invariants_ok"]
    eng.exchange.check_invariants()
    assert r["n_trades"] > 0


def test_output_contract(tmp_path):
    w = tiny()
    simulate_world(w, 1, tmp_path, workers=1)
    for s in w.symbol_list:
        bars = pd.read_parquet(tmp_path / "data" / f"{s}_1m.parquet")
        truth = pd.read_parquet(tmp_path / "truth" / f"{s}_1m.parquet")
        assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
        assert list(truth.columns) == ["fair", "mispricing"]
        assert all(bars[c].dtype == np.float64 for c in bars.columns)
        assert isinstance(bars.index, pd.DatetimeIndex)
        assert len(bars) == w.n_bars
        assert bars.index[0] == pd.Timestamp(w.calendar_start)
        assert (bars.index.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all()
        assert truth.index.equals(bars.index)
        assert not bars.isna().any().any() and not truth.isna().any().any()
        assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
        assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
        np.testing.assert_allclose(truth["mispricing"], np.log(truth["fair"]) - np.log(bars["close"]))
        empty = (bars["volume"] == 0).to_numpy().copy()
        empty[0] = False
        prev_close = bars["close"].shift(1).to_numpy()
        for c in ("open", "high", "low", "close"):
            assert (bars[c].to_numpy()[empty] == prev_close[empty]).all()


def test_fair_is_public_reference():
    w = tiny()
    p0, sched = build_schedules(w, 2)
    eng = simulate_symbol(w, 2, "S01", p0["S01"], sched["S01"], return_engine=True)["engine"]
    book = eng.exchange.books["S01"]
    m = book.mid()
    expected = m / 100 if m is not None else (eng.exchange.last_price["S01"] / 100)
    assert eng.fair("S01") == expected
    assert eng.fair("S01", step=0) == expected  # step is ignored: no exogenous path


def test_market_makers_never_access_schedule():
    w = tiny()
    p0, sched = build_schedules(w, 5)
    r = simulate_symbol(w, 5, "S02", p0["S02"], sched["S02"], return_engine=True)
    eng = r["engine"]
    readers = set(r["schedule_reads"])
    assert readers, "informed traders should read the schedule"
    for acct in readers:
        assert isinstance(eng.bots.by_account[acct], TargetTrader)
    mm = next(b for b in eng.bots.bots if isinstance(b, FlowMarketMaker))
    with pytest.raises(PermissionError):
        eng.target_view(mm)
    tree = ast.parse(textwrap.dedent(inspect.getsource(FlowMarketMaker)))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
            {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for banned in ("_schedule", "schedule_reads", "target_view", "view", "paths", "fair", "fair_cents"):
        assert banned not in names, banned


def test_placebo_price_independent_of_schedule():
    """Without informed flow the tape must not depend on F at all."""
    w = tiny("uwtc_placebo")
    p0, sched = build_schedules(w, 4)
    other = sched["S00"] * np.exp(np.linspace(0, 0.5, len(sched["S00"])))
    r1 = simulate_symbol(w, 4, "S00", p0["S00"], sched["S00"])
    r2 = simulate_symbol(w, 4, "S00", p0["S00"], other)
    for k in ("open", "high", "low", "close", "volume"):
        np.testing.assert_array_equal(r1["bars"][k], r2["bars"][k])
    # ...whereas the informed world does react to F.
    wi = tiny()
    i1 = simulate_symbol(wi, 4, "S00", p0["S00"], sched["S00"])
    i2 = simulate_symbol(wi, 4, "S00", p0["S00"], other)
    assert not np.array_equal(i1["bars"]["close"], i2["bars"]["close"])


def test_placebo_has_no_target_trader():
    w = WorldConfig.from_yaml(WORLDS / "uwtc_placebo.yaml")
    assert all(b["type"] != "target_trader" for b in w.bot_specs())
    r = simulate_symbol(tiny("uwtc_placebo"), 1, "S00", 50.0, np.full(120, 50.0))
    assert "TargetTrader" not in r["bot_types"]
    assert r["schedule_reads"] == {}
    # Placebo is the informed world with only the informed count zeroed.
    a = yaml.safe_load((WORLDS / "uwtc_informed.yaml").read_text())
    b = yaml.safe_load((WORLDS / "uwtc_placebo.yaml").read_text())
    assert a["informed"]["count"] > 0 and b["informed"]["count"] == 0
    a["informed"]["count"] = 0
    a["name"] = b["name"]
    assert a == b


def test_roster_has_no_exogenous_market_makers():
    for name in ("uwtc_informed", "uwtc_placebo"):
        cfg = tiny(name).round_config("S00", 50.0, 1)
        types = {b.type for b in cfg.bots}
        assert "market_maker" not in types
        assert types <= {"flow_market_maker", "noise_trader", "retail_buyer", "target_trader"}
        assert not cfg.instruments.etf_enabled and not cfg.instruments.letf_enabled
        assert not cfg.instruments.futures_enabled
        assert cfg.market.close_mark_method.value == "mid" and cfg.market.settlement_method.value == "mid"
