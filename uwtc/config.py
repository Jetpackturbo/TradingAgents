"""Server-side configuration schemas.

Everything in this module is HIDDEN from competitors: it is never shipped in
``uwtc.sdk`` and is only exposed through the admin API. A round config is built
by deep-merging ``config/market.yaml``, ``config/bots.yaml`` and the per-round
override file ``config/rounds/round_XX.yaml``.
"""

from __future__ import annotations

import copy
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Market handling                                                              #
# --------------------------------------------------------------------------- #


class STPMode(str, Enum):
    CANCEL_RESTING = "cancel_resting"
    CANCEL_INCOMING = "cancel_incoming"
    CANCEL_BOTH = "cancel_both"
    OFF = "off"


class MarkMethod(str, Enum):
    FAIR = "fair"
    VWAP = "vwap"
    MID = "mid"


class FeeConfig(_Strict):
    # Fees are charged per share/contract in cents; negative maker fee = rebate.
    maker_cents_per_unit: int = 0
    taker_cents_per_unit: int = 1


class RiskConfig(_Strict):
    max_order_qty: int = 100
    max_open_orders: int = 50
    position_limit: int = 500
    position_limit_overrides: dict[str, int] = Field(default_factory=dict)
    # Sum over instruments of |position| * mark, in cents.
    max_gross_notional_cents: int = 300_000_00
    msgs_per_second: float = 25.0
    msg_burst: int = 50
    # Disconnect after this many rate-limit rejections within one round.
    max_rate_violations: int = 500


class SpeedBumpConfig(_Strict):
    delay_ms: int = 0
    jitter_ms: int = 0
    apply_to_cancels: bool = True


class MarketDataConfig(_Strict):
    book_depth: int = 5
    publish_ms: int = 250
    anonymize_trades: bool = True
    account_publish_ms: int = 1000


class CircuitBreakerConfig(_Strict):
    enabled: bool = True
    # Reject orders priced further than this fraction from the reference price.
    price_band: float = 0.10


class MarketConfig(_Strict):
    tick_cents: int = 1
    lot_size: int = 1
    starting_cash_cents: int = 100_000_00
    stp_mode: STPMode = STPMode.CANCEL_RESTING
    market_order_band: float = 0.05
    cancel_on_eod: bool = True
    day_gap_ms: int = 0
    fees: FeeConfig = Field(default_factory=FeeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    speed_bump: SpeedBumpConfig = Field(default_factory=SpeedBumpConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    settlement_method: MarkMethod = MarkMethod.FAIR
    close_mark_method: MarkMethod = MarkMethod.FAIR
    vwap_window_ms: int = 10_000
    bots_risk_exempt: bool = True


# --------------------------------------------------------------------------- #
# Instruments                                                                  #
# --------------------------------------------------------------------------- #


class InstrumentsConfig(_Strict):
    equities: list[str] = Field(default_factory=lambda: ["A", "B", "C", "D", "E"])
    etf_enabled: bool = True
    etf_symbol: str = "IDX"
    # equal_weight_daily: NAV compounds the mean daily return and the sponsor
    # rebalances to equal weights at EOD. price_weighted: NAV = mean(prices).
    etf_weighting: Literal["equal_weight_daily", "price_weighted"] = "equal_weight_daily"
    letf_enabled: bool = True
    letf_symbol: str = "E2X"
    letf_underlying: str = "E"
    letf_leverage: float = 2.0
    letf_initial_nav: float = 50.0
    futures_enabled: bool = True
    futures_unlock_fee_cents: int = 5_000_00
    # 7-day futures for every equity: series settle on these days (each series
    # lists the day after the previous one settles).
    futures_7d_expiries: list[int] = Field(default_factory=lambda: [7, 14])
    futures_14d_underlyings: list[str] = Field(default_factory=lambda: ["A", "B", "C"])
    futures_14d_expiry: int = 14


# --------------------------------------------------------------------------- #
# Fair-value model and signals                                                 #
# --------------------------------------------------------------------------- #


class EquityModel(_Strict):
    initial_price: float
    beta: float = 1.0
    # Per-day volatilities / drifts of log price.
    idio_vol: float = 0.015
    drift: float = 0.0
    commodity_loadings: dict[str, float] = Field(default_factory=dict)


class CommodityModel(_Strict):
    initial_price: float
    daily_vol: float = 0.02
    drift: float = 0.0


class JumpConfig(_Strict):
    enabled: bool = False
    # Expected jumps per day per equity and jump size std (log).
    rate_per_day: float = 0.1
    size_std: float = 0.03


class ModelConfig(_Strict):
    steps_per_day: int = 60
    market_daily_vol: float = 0.012
    market_drift: float = 0.0
    equities: dict[str, EquityModel] = Field(
        default_factory=lambda: {
            "A": EquityModel(initial_price=100.0, beta=1.2, commodity_loadings={"OIL": 0.4}),
            "B": EquityModel(initial_price=80.0, beta=0.8, commodity_loadings={"GOLD": 0.5}),
            "C": EquityModel(initial_price=120.0, beta=1.0, commodity_loadings={"OIL": -0.3}),
            "D": EquityModel(initial_price=60.0, beta=0.6, idio_vol=0.02),
            "E": EquityModel(initial_price=90.0, beta=1.4, commodity_loadings={"OIL": 0.2, "GOLD": 0.2}),
        }
    )
    commodities: dict[str, CommodityModel] = Field(
        default_factory=lambda: {
            "OIL": CommodityModel(initial_price=75.0, daily_vol=0.02),
            "GOLD": CommodityModel(initial_price=2000.0, daily_vol=0.012),
        }
    )
    jumps: JumpConfig = Field(default_factory=JumpConfig)


class SignalConfig(_Strict):
    # Forward horizon (days) the equity signal predicts.
    horizon_days: int = 1
    base_ic: float = 0.25
    premium_ic: float = 0.45
    futures_ic: float = 0.5
    # shared: every team in a tier sees the same noise draw; per_team: independent.
    noise_mode: Literal["shared", "per_team"] = "shared"
    disclose_strength: bool = True
    # Decay: ic_d = ic_0 * decay_per_day ** (d - 1). 1.0 = no decay.
    decay_per_day: float = 1.0
    # Premium subscribers also receive a live commodity price feed.
    commodity_feed: bool = True
    commodity_feed_lead_steps: int = 0


# --------------------------------------------------------------------------- #
# Mechanics                                                                    #
# --------------------------------------------------------------------------- #


class AuctionRule(str, Enum):
    UNIFORM_K_PLUS_1 = "uniform_k_plus_1"
    OWN_BID = "own_bid"
    GSP = "gsp"


class AuctionConfig(_Strict):
    enabled: bool = False
    winners: int = 5
    payment_rule: AuctionRule = AuctionRule.UNIFORM_K_PLUS_1
    reserve_cents: int = 0
    max_bid_cents: int = 50_000_00
    # Auction runs during PRE_OPEN; winners receive the premium tier all round.
    tie_break: Literal["earliest", "random"] = "earliest"


class SignalShopConfig(_Strict):
    enabled: bool = False
    futures_signal_price_cents: int = 2_000_00
    # Last day on which the product may be purchased.
    last_purchase_day: int = 14


class ResearchConfig(_Strict):
    enabled: bool = False
    # ic += max_boost * (1 - exp(-spend / scale_cents)), capped at ic_cap.
    max_boost: float = 0.3
    scale_cents: int = 3_000_00
    ic_cap: float = 0.8
    max_total_cents: int = 50_000_00


class MechanicsConfig(_Strict):
    auction: AuctionConfig = Field(default_factory=AuctionConfig)
    signal_shop: SignalShopConfig = Field(default_factory=SignalShopConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)


# --------------------------------------------------------------------------- #
# Bots                                                                         #
# --------------------------------------------------------------------------- #


class BotSpec(_Strict):
    type: str
    name: str
    count: int = 1
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Round                                                                        #
# --------------------------------------------------------------------------- #


class RoundConfig(_Strict):
    round_number: int = 1
    name: str = ""
    seed: int = 1
    days: int = 15
    day_ms: int = 60_000
    pre_open_ms: int = 20_000
    tick_ms: int = 100
    market: MarketConfig = Field(default_factory=MarketConfig)
    instruments: InstrumentsConfig = Field(default_factory=InstrumentsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    mechanics: MechanicsConfig = Field(default_factory=MechanicsConfig)
    bots: list[BotSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "RoundConfig":
        if self.day_ms % self.model.steps_per_day:
            raise ValueError("day_ms must be divisible by model.steps_per_day")
        if self.day_ms % self.tick_ms:
            raise ValueError("day_ms must be divisible by tick_ms")
        missing = set(self.instruments.equities) - set(self.model.equities)
        if missing:
            raise ValueError(f"no model for equities {sorted(missing)}")
        if self.instruments.letf_underlying not in self.instruments.equities:
            raise ValueError("letf_underlying must be an equity")
        for d in self.instruments.futures_7d_expiries + [self.instruments.futures_14d_expiry]:
            if not 1 <= d <= self.days:
                raise ValueError(f"futures expiry day {d} outside round")
        if self.instruments.futures_7d_expiries != sorted(set(self.instruments.futures_7d_expiries)):
            raise ValueError("futures_7d_expiries must be strictly increasing")
        for s in self.instruments.futures_14d_underlyings:
            if s not in self.instruments.equities:
                raise ValueError(f"14d future underlying {s} not an equity")
        return self

    @property
    def step_ms(self) -> int:
        return self.day_ms // self.model.steps_per_day

    @property
    def total_steps(self) -> int:
        return self.days * self.model.steps_per_day


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``.

    Lists are replaced, not concatenated, except ``bots`` handled by caller.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def merge_bots(base: list[dict], overrides: dict[str, dict] | None, extra: list[dict] | None) -> list[dict]:
    """Bots are keyed by name. ``bot_overrides`` patch existing bots by name;
    ``extra_bots`` append new ones."""
    by_name = {b["name"]: copy.deepcopy(b) for b in base}
    order = [b["name"] for b in base]
    for name, patch in (overrides or {}).items():
        if name not in by_name:
            raise ValueError(f"bot_overrides references unknown bot {name!r}")
        by_name[name] = deep_merge(by_name[name], patch)
    for b in extra or []:
        if b["name"] in by_name:
            raise ValueError(f"duplicate bot name {b['name']!r}")
        by_name[b["name"]] = copy.deepcopy(b)
        order.append(b["name"])
    return [by_name[n] for n in order]


def load_round_config(
    round_number: int | None = None,
    config_dir: Path | str | None = None,
    overrides: dict | None = None,
) -> RoundConfig:
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    base = _load_yaml(cdir / "market.yaml")
    bots = _load_yaml(cdir / "bots.yaml").get("bots", [])
    rnd: dict = {}
    if round_number is not None:
        rnd = _load_yaml(cdir / "rounds" / f"round_{round_number:02d}.yaml")
        if not rnd:
            raise FileNotFoundError(f"no config for round {round_number}")
    merged = deep_merge(base, {k: v for k, v in rnd.items() if k not in ("bot_overrides", "extra_bots")})
    merged["bots"] = merge_bots(bots, rnd.get("bot_overrides"), rnd.get("extra_bots"))
    if round_number is not None:
        merged.setdefault("round_number", round_number)
    if overrides:
        bot_keys = ("bot_overrides", "extra_bots")
        merged = deep_merge(merged, {k: v for k, v in overrides.items() if k not in bot_keys})
        if any(k in overrides for k in bot_keys):
            merged["bots"] = merge_bots(merged["bots"], overrides.get("bot_overrides"), overrides.get("extra_bots"))
    return RoundConfig.model_validate(merged)


def list_round_numbers(config_dir: Path | str | None = None) -> list[int]:
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    out = []
    for p in sorted((cdir / "rounds").glob("round_*.yaml")):
        try:
            out.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return out
