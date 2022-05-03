# -*- coding: utf-8 -*-

from numpy import __version__ as __numpy_version__

from ._booster import *

__doc__ = \
"""Boosting of the implementation of the yamb game.

Boosting the yamb game engine sets the back-end of the game to NumPy and makes
use of caching intermediate results-score evaluations.

Requirements:
* Python version 3.2 or higher,
* NumPy version 1.17.0 or higher.
"""

__all__ = \
    [ '__numpy_version__' ] + \
        list(o for o in dir() if not o.startswith('_'))
