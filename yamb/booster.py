# -*- coding: utf-8 -*-

"""Boosting of the implementation of the yamb game.

Boosting the yamb game engine sets the back-end of the game to NumPy and makes
use of caching intermediate results-score evaluations.

Requirements:
* Python version 3.2 or higher,
* NumPy version 1.17.0 or higher.
"""

import sys as _sys
if (_sys.version_info.major, _sys.version_info.minor) < (3, 2):
    raise RuntimeError(
        "Python version is not suitable (found version: {version}). Use " \
            "python >= 3.2 for boosting support.".format(
                version = str.join(
                    '.',
                    (
                        str(_sys.version_info.major),
                        str(_sys.version_info.minor),
                        str(_sys.version_info.micro)
                    )
                )
            )
        )
del _sys

import functools as _functools

_np = None
try:
    import numpy as _np
except ImportError:
    raise ImportError(
        "Missing optional dependency 'numpy'. Install numpy >= 1.17.0 for " \
            "boosting support."
    )
if _np.lib.NumpyVersion(_np.__version__) < _np.lib.NumpyVersion('1.17.0'):
    raise ImportError(
        "Optional dependency 'numpy' is not of a suitable version (found " \
            "version: {np_version}). Install numpy >= 1.17.0 for boosting " \
            "support.".format(np_version = _np.__version__)
    )

import _types
import engine as _engine

def boost_die (cls = _engine.Die, class_name = None):
    """Boosts a die class by enforcing using the NumPy back-end.

When NumPy back-end is enforced, `cls.sides` is converted to a `numpy.ndarray`
and `random_state` is set to a `numpy.random.Generator` unless a different
random state is explicitly provided.  Converting `cls.sides` is not done
in-place, meaning that the original `cls`' class variable `sides` is remained
unaltered, but the resulting class' variable `sides` is set to a
`numpy.ndarray` version of `cls.side`.

Parameters
----------
cls : type[engine.Die], default = engine.Die
    Class to boost.

class_name : string, optional
    Name to set as the `__name__` property of the resulting class.

Returns
-------
type[engine.Die]
    The boosted subclass version of `cls`.

Raises
------
TypeError
    If `cls` is not a subclass of `engine.Die` class.  If `class_name` is not
    a string.
"""
    if not (isinstance(cls, type) and issubclass(cls, _engine.Die)):
        raise TypeError("Base class must be a die subclass.")
    if not (class_name is None or isinstance(class_name, _types.AnyString)):
        raise TypeError("Class name must be a string.")

    if class_name is not None:
        class_name = str(class_name)

    class BoostedDie (cls):
        if class_name is not None:
            __name__ = class_name

        __doc__ = cls.__doc__

        sides = _np.array(cls.sides, dtype = _np.int32)
        sides.flags.writeable = False

        boosted = True

        def __init__ (self, *args, random_state = None, **kwargs):
            if random_state is None:
                super(BoostedDie, self).__init__(
                    *args,
                    random_state = _np.random.default_rng(),
                    **kwargs
                )
            elif isinstance(random_state, _types.AnyNumber):
                super(BoostedDie, self).__init__(
                    *args,
                    random_state = _np.random.default_rng(random_state),
                    **kwargs
                )
            else:
                super(BoostedDie, self).__init__(
                    *args,
                    random_state = random_state,
                    **kwargs
                )

    return BoostedDie

def boost_column (
    cls = _engine.Column,
    max_cache_size = 0x20,
    class_name = None
):
    """Boosts a column class by enforcing using the NumPy back-end and \
caching intermediate evaluations.

When NumPy back-end is enforced, this affects the `cls.lambda_score` class
variable and connected methods (`is_lambda` and `get_lambda_slots`), as well
as the `_slots` instance variable.  In particular, `cls.lambda_score` is set
to `nan`, and `_slots` is set to a `numpy.ndarray` of `float`s (this is
achieved by overriding the `_new_empty_scores` method).  Also, the `is_lambda`
method is implemented with the `numpy.isnan` universal function
(`numpy.ufunc`), therefore multiple scores may be checked at once by passing
an array-like input.

Additionally, the `__array__` method now simply returns the `_slots` instance
variable, and checking for `numpy` dependency is omitted from the `to_numpy`
method.

Parameters
----------
cls : type[engine.Column], default = engine.Column
    Class to boost.

max_cache_size : integer, default = 32
    Maximum cache size to provide to `functools.lru_cache` decorator.  See
    Notes for more details.

class_name : string, optional
    Name to set as the `__name__` property of the resulting class.

Returns
-------
type[engine.Column]
    The boosted subclass version of `cls`.

Raises
------
TypeError
    If `cls` is not a subclass of `engine.Die` class.  If `max_cache_size` is
    not an integral value.  If `class_name` is not a string.

Notes
-----
The `functools.lru_cache` decorator is applied to the following class methods:

* `_ensure_roll_index`,
* `_ensure_slot`,
* `_count_results` (actually, an auxiliary method is implemented which behaves
    the same but expects hashable arguments),
* `_evaluate`.

At most ``4 * max_cache_size`` intermediate argument-result pairs are stored
at any given moment by the class through these four methods.  Total memory
consumption depends on the size of arguments and results, but the former two
methods should expect integers, and maybe very short strings, while the latter
two should also expect quite small structures of integers in a standard game
with 5 dice.

To make the most out of the boosted column class, do not boost the abstract
base class `engine.Column`, but a concrete subclass.  Speciffically, any
overrides of the `get_next_available_slots` method nullify the boost to the 
method implemented by the returned class.
"""
    if not (isinstance(cls, type) and issubclass(cls, _engine.Column)):
        raise TypeError("Base class must be a column subclass.")
    if not isinstance(max_cache_size, _types.AnyInteger):
        raise TypeError("Cache size must be an integral value.")
    if not (class_name is None or isinstance(class_name, _types.AnyString)):
        raise TypeError("Class name must be a string.")

    if max_cache_size < 0:
        raise ValueError("Cache size must be greater than or equal to 0.")

    max_cache_size = int(max_cache_size)
    if class_name is not None:
        class_name = str(class_name)

    class BoostedColumn (cls):
        if class_name is not None:
            __name__ = class_name

        __doc__ = cls.__doc__

        boosted = True

        _number_slots_array = _np.array(
            list(sorted(cls.number_slots)),
            dtype = _np.int32
        )
        _sum_slots_array = _np.array(
            list(sorted(cls.sum_slots)),
            dtype = _np.int32
        )
        _collection_slots_array = _np.array(
            list(sorted(cls.collection_slots)),
            dtype = _np.int32
        )
        _fillable_slots_array = _np.array(
            list(sorted(cls.fillable_slots)),
            dtype = _np.int32
        )
        _auto_slots_array = _np.array(
            list(sorted(cls.auto_slots)),
            dtype = _np.int32
        )
        _slots_array = _np.array(
            list(sorted(cls.slots)),
            dtype = _np.int32
        )

        _number_slots_array.flags.writeable = False
        _sum_slots_array.flags.writeable = False
        _collection_slots_array.flags.writeable = False
        _fillable_slots_array.flags.writeable = False
        _auto_slots_array.flags.writeable = False
        _slots_array.flags.writeable = False

        lambda_score = float('nan')

        @classmethod
        def _new_empty_scores (cls):
            return _np.full(
                len(_engine.Slot),
                cls.lambda_score,
                dtype = _np.float32
            )

        @classmethod
        @_functools.lru_cache(maxsize = max_cache_size)
        def _ensure_roll_index (cls, roll):
            return super(BoostedColumn, cls)._ensure_roll_index(roll)

        @classmethod
        @_functools.lru_cache(maxsize = max_cache_size)
        def _ensure_slot (cls, slot):
            return super(BoostedColumn, cls)._ensure_slot(slot)

        @classmethod
        @_functools.lru_cache(maxsize = max_cache_size)
        def _count_results_tuple (cls, results):
            """Similar to `_count_results`, but expects a hashable sequence \
`results` (e. g. a `tuple`) for caching purposes."""
            return super(BoostedColumn, cls)._count_results(results)

        @classmethod
        def _count_results (cls, results):
            return cls._count_results_tuple(
                results if isinstance (results, _types.AnyHashable)
                    else tuple(results)
            )

        @classmethod
        @_functools.lru_cache(maxsize = max_cache_size)
        def _evaluate (cls, slot, results, counts):
            return super(BoostedColumn, cls)._evaluate(
                slot,
                results,
                counts
            )

        @classmethod
        def is_lambda (cls, score):
            #return _math.isnan(score)
            return _np.isnan(score) # <- allow checking multiple scores at once

        @classmethod
        def get_lambda_slots (cls, scores):
            return _np.flatnonzero(cls.is_lambda(scores))

        def get_available_slots (self):
            if self._available_slots is None:
                available_slots = self._type.get_lambda_slots(self._slots)
                available_slots = available_slots[
                    _np.isin(
                        available_slots,
                        self._type._fillable_slots_array,
                        assume_unique = True
                    )
                ]

                self._available_slots = available_slots

            return self._available_slots

        def to_numpy (self):
            return self.__array__()

        def __array__ (self):
            return self._slots

    return BoostedColumn
