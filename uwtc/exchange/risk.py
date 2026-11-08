"""Pre-trade risk checks. Return a reject reason string, or None if accepted."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import OrderType, Side

if TYPE_CHECKING:
    from .accounts import Account
    from .engine import Exchange


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate_shape(ex: "Exchange", symbol, side, qty, price, otype) -> str | None:
    """Checks applied to everyone, including risk-exempt bots."""
    if symbol not in ex.instruments:
        return "unknown_symbol"
    if not ex.tradable.get(symbol, False):
        return "not_tradable"
    if not _is_int(qty) or qty <= 0:
        return "bad_qty"
    if qty % ex.cfg.lot_size:
        return "bad_lot"
    if otype is not OrderType.MARKET:
        if not _is_int(price) or price <= 0:
            return "bad_price"
        if price % ex.cfg.tick_cents:
            return "bad_tick"
    return None


def check_risk(ex: "Exchange", acct: "Account", symbol: str, side: Side, qty: int,
               price: int | None, otype: OrderType) -> str | None:
    inst = ex.instruments[symbol]
    if inst.permission and inst.permission not in acct.permissions:
        return f"{inst.permission}_locked"
    if acct.risk_exempt:
        return None
    rc = ex.cfg.risk
    if qty > rc.max_order_qty:
        return "max_order_qty"
    rests = otype is OrderType.LIMIT
    if rests and len(acct.open_orders) >= rc.max_open_orders:
        return "max_open_orders"
    limit = rc.position_limit_overrides.get(symbol, rc.position_limit)
    pos = acct.positions.get(symbol, 0)
    if side is Side.BUY:
        if pos + acct.open_buy_qty.get(symbol, 0) + qty > limit:
            return "position_limit"
    else:
        if pos - acct.open_sell_qty.get(symbol, 0) - qty < -limit:
            return "position_limit"
    cb = ex.cfg.circuit_breaker
    ref = ex.ref_price.get(symbol)
    if cb.enabled and ref and price is not None and otype is not OrderType.MARKET:
        if abs(price - ref) > cb.price_band * ref:
            return "price_band"
    mark = ref or price or 0
    if mark:
        gross = acct.gross_notional(ex.marks_for_risk())
        # Conservative: assume this order increases gross exposure.
        new_pos = pos + side.sign * qty
        delta = (abs(new_pos) - abs(pos)) * mark
        if delta > 0 and gross + delta > rc.max_gross_notional_cents:
            return "gross_notional"
    return None
