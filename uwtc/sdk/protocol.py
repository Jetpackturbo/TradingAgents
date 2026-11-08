"""Wire protocol shared by the exchange and the team SDK.

All messages are JSON objects with a ``type`` field. All prices and cash
amounts are INTEGER CENTS; quantities are integer units.

Client -> server
----------------
``auth``               {api_key}
``place_order``        {symbol, side: "buy"|"sell", qty, price?, order_type: "limit"|"ioc"|"market", client_id?}
``cancel_order``       {order_id}
``cancel_all``         {symbol?}
``unlock_futures``     {}
``auction_bid``        {amount}
``buy_signal``         {product}
``allocate_research``  {amount}
``ping``               {nonce?}

Server -> client
----------------
``welcome``, ``round_start``, ``auction_open``, ``auction_result``, ``day_start``,
``market_data``, ``account``, ``order_ack``, ``order_reject``, ``fill``,
``order_cancelled``, ``cancel_reject``, ``settlement``, ``cash``,
``futures_unlocked``, ``signal``, ``request_reject``, ``day_end``,
``round_end``, ``pong``, ``error``.
"""

from __future__ import annotations

PROTOCOL_VERSION = 1

# Client -> server
AUTH = "auth"
PLACE_ORDER = "place_order"
CANCEL_ORDER = "cancel_order"
CANCEL_ALL = "cancel_all"
UNLOCK_FUTURES = "unlock_futures"
AUCTION_BID = "auction_bid"
BUY_SIGNAL = "buy_signal"
ALLOCATE_RESEARCH = "allocate_research"
PING = "ping"

CLIENT_TYPES = frozenset({
    PLACE_ORDER, CANCEL_ORDER, CANCEL_ALL, UNLOCK_FUTURES, AUCTION_BID, BUY_SIGNAL, ALLOCATE_RESEARCH, PING,
})

# Server -> client
WELCOME = "welcome"
ROUND_START = "round_start"
AUCTION_OPEN = "auction_open"
AUCTION_RESULT = "auction_result"
DAY_START = "day_start"
MARKET_DATA = "market_data"
ACCOUNT = "account"
ORDER_ACK = "order_ack"
ORDER_REJECT = "order_reject"
FILL = "fill"
ORDER_CANCELLED = "order_cancelled"
CANCEL_REJECT = "cancel_reject"
SETTLEMENT = "settlement"
CASH = "cash"
FUTURES_UNLOCKED = "futures_unlocked"
SIGNAL = "signal"
REQUEST_REJECT = "request_reject"
DAY_END = "day_end"
ROUND_END = "round_end"
PONG = "pong"
ERROR = "error"

FUTURES_SIGNALS = "futures_signals"

SIDES = ("buy", "sell")
ORDER_TYPES = ("limit", "ioc", "market")
