"""World configuration (YAML) and its mapping to a per-symbol UWTC ``RoundConfig``.

Each symbol is simulated in its own :class:`~synthmarket.engine.EndogenousRoundEngine`
(symbols are independent by construction). Bot rates are specified per output bar and
converted to the units UWTC's bots expect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from uwtc.config import RoundConfig, load_round_config

from .schedule import ScheduleSpec, stable_seed

UWTC_CONFIG_DIR = Path(__file__).resolve().parent / "uwtc_config"
_UWTC_MINUTE_MS = 60_000  # NoiseTrader's "orders_per_minute" is per 60k sim-ms


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MarketMakerSpec(_Strict):
    name: str = "fmm"
    count: int = 1
    params: dict[str, Any] = Field(default_factory=dict)


class InformedSpec(_Strict):
    count: int = 0
    interval_ms: int = 100
    arrivals_per_bar: float = 0.3
    size_min: int = 10
    size_max: int = 30
    edge_bps: float = 5.0
    lag_steps: int = 1
    target_noise_bps: float = 0.0
    side: Literal["both", "buy", "sell"] = "both"


class NoiseSpec(_Strict):
    count: int = 4
    interval_ms: int = 100
    orders_per_bar: float = 0.3
    size_min: int = 1
    size_max: int = 20
    slippage_bps: float = 10.0


class RetailSpec(_Strict):
    count: int = 0
    interval_ms: int = 100
    orders_per_bar: float = 0.05
    size_min: int = 5
    size_max: int = 30
    offset_bps_min: float = 0.0
    offset_bps_max: float = 25.0
    marketable_prob: float = 0.15


class WorldConfig(_Strict):
    name: str = "world"
    seed: int = 1
    n_symbols: int = 16
    symbols: list[str] | None = None
    symbol_prefix: str = "S"
    price_min: float = 20.0
    price_max: float = 200.0
    days: int = 3
    bars_per_day: int = 1440
    bar_ms: int = 1000
    tick_ms: int = 100
    calendar_start: str = "2024-01-01 00:00"
    schedule: ScheduleSpec = Field(default_factory=ScheduleSpec)
    market_makers: list[MarketMakerSpec] = Field(default_factory=lambda: [MarketMakerSpec()])
    informed: InformedSpec = Field(default_factory=InformedSpec)
    noise: NoiseSpec = Field(default_factory=NoiseSpec)
    retail: RetailSpec = Field(default_factory=RetailSpec)
    extra_bots: list[dict[str, Any]] = Field(default_factory=list)
    workers: int | None = None

    @model_validator(mode="after")
    def _check(self) -> "WorldConfig":
        if self.bar_ms % self.tick_ms:
            raise ValueError("bar_ms must be a multiple of tick_ms")
        if self.symbols is not None and len(set(self.symbols)) != len(self.symbols):
            raise ValueError("duplicate symbols")
        for b in (self.informed, self.noise, self.retail):
            if b.count and b.interval_ms % self.tick_ms:
                raise ValueError("bot interval_ms must be a multiple of tick_ms")
        return self

    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorldConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh) or {})

    @property
    def symbol_list(self) -> list[str]:
        if self.symbols is not None:
            return list(self.symbols)
        return [f"{self.symbol_prefix}{i:02d}" for i in range(self.n_symbols)]

    @property
    def day_ms(self) -> int:
        return self.bars_per_day * self.bar_ms

    @property
    def n_bars(self) -> int:
        return self.days * self.bars_per_day

    def round_seed(self, seed: int, symbol: str) -> int:
        state = np.random.SeedSequence(stable_seed("round", seed, symbol)).generate_state(1)[0]
        return int(state) & 0x7FFFFFFF

    def bot_specs(self) -> list[dict]:
        bots: list[dict] = []
        for mm in self.market_makers:
            if mm.count > 0:
                bots.append({"name": mm.name, "type": "flow_market_maker", "count": mm.count,
                             "params": dict(mm.params)})
        n = self.noise
        if n.count > 0:
            bots.append({"name": "noise", "type": "noise_trader", "count": n.count, "params": {
                "symbols": "equities", "interval_ms": n.interval_ms,
                "orders_per_minute": n.orders_per_bar * _UWTC_MINUTE_MS / self.bar_ms,
                "size_min": n.size_min, "size_max": n.size_max, "slippage_bps": n.slippage_bps}})
        r = self.retail
        if r.count > 0:
            bots.append({"name": "retail", "type": "retail_buyer", "count": r.count, "params": {
                "symbols": "equities", "interval_ms": r.interval_ms,
                "orders_per_day": r.orders_per_bar * self.bars_per_day,
                "size_min": r.size_min, "size_max": r.size_max, "offset_bps_min": r.offset_bps_min,
                "offset_bps_max": r.offset_bps_max, "marketable_prob": r.marketable_prob}})
        i = self.informed
        if i.count > 0:
            bots.append({"name": "informed", "type": "target_trader", "count": i.count, "params": {
                "symbols": "equities", "interval_ms": i.interval_ms,
                "arrivals_per_bar": i.arrivals_per_bar, "size_min": i.size_min, "size_max": i.size_max,
                "edge_bps": i.edge_bps, "lag_steps": i.lag_steps,
                "target_noise_bps": i.target_noise_bps, "side": i.side}})
        bots.extend(dict(b) for b in self.extra_bots)
        return bots

    def round_config(self, symbol: str, initial_price: float, seed: int) -> RoundConfig:
        overrides = {
            "name": f"{self.name}:{symbol}",
            "seed": self.round_seed(seed, symbol),
            "days": self.days,
            "day_ms": self.day_ms,
            "pre_open_ms": 0,
            "tick_ms": self.tick_ms,
            "market": {"market_data": {"publish_ms": self.bar_ms}},
            "instruments": {"equities": [symbol], "letf_underlying": symbol},
            "model": {"steps_per_day": self.bars_per_day,
                      # UWTC still draws its (unused) exogenous path; nonzero vol keeps its stats finite.
                      "equities": {symbol: {"initial_price": float(initial_price), "beta": 0.0,
                                            "idio_vol": 0.01}}},
            "extra_bots": self.bot_specs(),
        }
        return load_round_config(config_dir=UWTC_CONFIG_DIR, overrides=overrides)
