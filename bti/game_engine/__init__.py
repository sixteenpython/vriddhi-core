"""Deterministic Beat the Index game engine.

The public surface deliberately exports the game service and value errors only.
The Vriddhi reference provider remains an internal implementation detail.
"""

from .engine import BTIGame, GameRuleError

__all__ = ["BTIGame", "GameRuleError"]
