from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import BotSpec
from ..exchange.instruments import FUTURES_PERMISSION
from .base import Bot, BotContext, bot_seed
from .behaviors import DayTrader, FuturesArb, IndexRebalancer, LETFRebalancer, MarketMaker, NoiseTrader, RetailBuyer

if TYPE_CHECKING:
    from ..engine.round import RoundEngine

BOT_TYPES: dict[str, type[Bot]] = {
    "market_maker": MarketMaker,
    "retail_buyer": RetailBuyer,
    "day_trader": DayTrader,
    "noise_trader": NoiseTrader,
    "index_rebalancer": IndexRebalancer,
    "letf_rebalancer": LETFRebalancer,
    "futures_arb": FuturesArb,
}

BOT_PREFIX = "bot:"
HEDGE_ACCOUNT = "bot:__external_hedge__"


def build_bots(specs: list[BotSpec]) -> list[Bot]:
    bots: list[Bot] = []
    for spec in specs:
        if not spec.enabled:
            continue
        cls = BOT_TYPES.get(spec.type)
        if cls is None:
            raise ValueError(f"unknown bot type {spec.type!r}")
        for i in range(spec.count):
            name = spec.name if spec.count == 1 else f"{spec.name}-{i + 1}"
            bots.append(cls(name, dict(spec.params)))
    names = [b.name for b in bots]
    if len(set(names)) != len(names):
        raise ValueError("duplicate bot names")
    return bots


class BotManager:
    def __init__(self, engine: "RoundEngine", specs: list[BotSpec]) -> None:
        self.engine = engine
        self.bots = build_bots(specs)
        self.by_account: dict[str, Bot] = {}
        self._next_due: dict[str, int] = {}
        self._tick_index = 0
        self.hedge_account = HEDGE_ACCOUNT

    def account_of(self, bot: Bot) -> str:
        return BOT_PREFIX + bot.name

    def setup(self) -> None:
        eng = self.engine
        eng.exchange.add_account(HEDGE_ACCOUNT, "bot", cash=0, risk_exempt=True, permissions=[FUTURES_PERMISSION])
        for bot in self.bots:
            acct = self.account_of(bot)
            eng.exchange.add_account(acct, "bot", cash=0, risk_exempt=eng.cfg.market.bots_risk_exempt,
                                     permissions=[FUTURES_PERMISSION])
            bot.ctx = BotContext(eng, acct, bot_seed(eng.cfg.seed, bot.name))
            self.by_account[acct] = bot
        for bot in self.bots:
            bot.on_round_start(bot.ctx)

    def day_start(self) -> None:
        for bot in self.bots:
            self._next_due[bot.name] = 0
            bot.on_day_start(bot.ctx)

    def tick(self, time_in_day: int) -> None:
        n = len(self.bots)
        if not n:
            return
        # Rotate the starting bot each tick so no bot systematically acts first.
        start = self._tick_index % n
        self._tick_index += 1
        for k in range(n):
            bot = self.bots[(start + k) % n]
            if time_in_day >= self._next_due.get(bot.name, 0):
                self._next_due[bot.name] = time_in_day + bot.interval_ms
                bot.on_tick(bot.ctx)

    def day_end(self) -> None:
        for bot in self.bots:
            bot.on_day_end(bot.ctx)

    def on_events(self, account: str, event: dict) -> None:
        bot = self.by_account.get(account)
        if bot is not None:
            bot.on_event(bot.ctx, event)
