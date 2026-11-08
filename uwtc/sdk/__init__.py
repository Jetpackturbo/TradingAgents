"""Team-facing SDK. This package is the ONLY code distributed to competitors.

It must never import anything from the rest of ``uwtc`` (server internals and
hidden configuration live there). ``tests/exploit/test_sdk_isolation.py``
enforces this.
"""

from .base import BaseStrategy, Book, Context

__all__ = ["BaseStrategy", "Book", "Context"]
