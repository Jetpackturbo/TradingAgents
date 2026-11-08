from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config import RoundConfig

FUTURES_PERMISSION = "futures"


class InstrumentKind(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    LETF = "letf"
    FUTURE = "future"


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    kind: InstrumentKind
    underlying: str | None = None
    list_day: int = 1
    expiry_day: int | None = None
    permission: str | None = None

    def is_listed(self, day: int) -> bool:
        if day < self.list_day:
            return False
        return self.expiry_day is None or day <= self.expiry_day

    def public(self) -> dict:
        return {
            "symbol": self.symbol,
            "kind": self.kind.value,
            "underlying": self.underlying,
            "list_day": self.list_day,
            "expiry_day": self.expiry_day,
            "requires": self.permission,
        }


def future_symbol(underlying: str, tenor: int, expiry: int) -> str:
    return f"{underlying}-F{tenor}-{expiry:02d}"


def build_instruments(cfg: RoundConfig) -> dict[str, Instrument]:
    ic = cfg.instruments
    if len({*ic.equities, ic.etf_symbol, ic.letf_symbol}) != len(ic.equities) + 2:
        raise ValueError("ETF / LETF symbols must be distinct from equities and each other")
    out: dict[str, Instrument] = {}
    for s in ic.equities:
        out[s] = Instrument(s, InstrumentKind.EQUITY)
    if ic.etf_enabled:
        out[ic.etf_symbol] = Instrument(ic.etf_symbol, InstrumentKind.ETF)
    if ic.letf_enabled:
        out[ic.letf_symbol] = Instrument(ic.letf_symbol, InstrumentKind.LETF, underlying=ic.letf_underlying)
    if ic.futures_enabled:
        prev = 0
        for exp in ic.futures_7d_expiries:
            for s in ic.equities:
                sym = future_symbol(s, 7, exp)
                out[sym] = Instrument(
                    sym, InstrumentKind.FUTURE, underlying=s, list_day=prev + 1,
                    expiry_day=exp, permission=FUTURES_PERMISSION,
                )
            prev = exp
        for s in ic.futures_14d_underlyings:
            sym = future_symbol(s, 14, ic.futures_14d_expiry)
            out[sym] = Instrument(
                sym, InstrumentKind.FUTURE, underlying=s, list_day=1,
                expiry_day=ic.futures_14d_expiry, permission=FUTURES_PERMISSION,
            )
    return out
