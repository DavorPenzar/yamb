# -*- coding: utf-8 -*-

from .engine import *

__doc__ = \
"""Simple light-weight object-oriented implementation of the interface of \
the yamb game.

The game is primarily implemented as a solitaire game, but may be used for
implementing a multi-player game (e. g. each player may play at their own
instance of the `Yamb` game class).  However, some columns, such as the
counter announced, are impossible to implement without implementing a custom
back-channel communication amongst the players.
"""

__all__ = list(o for o in dir() if o and not o.startswith('_'))
