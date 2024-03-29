# -*- coding: utf-8 -*-

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
import math as _math

_np = None
_pd = None
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
try:
    import pandas as _pd
except ImportError:
    raise ImportError(
        "Missing optional dependency 'pandas'. Install pandas >= 1.0.0 for " \
            "boosting support."
    )
if _np.lib.NumpyVersion(_pd.__version__) < _np.lib.NumpyVersion('1.0.0'):
    raise ImportError(
        "Optional dependency 'pandas' is not of a suitable version (found " \
            "version: {pd_version}). Install pandas >= 1.0.0 for boosting " \
            "support.".format(pd_version = _pd.__version__)
    )

from .._types import \
    AnyHashable as _AnyHashable, \
    AnyIterable as _AnyIterable, \
    AnySequence as _AnySequence, \
    AnyString as _AnyString, \
    AnyNumber as _AnyNumber, \
    AnyInteger as _AnyInteger, \
    AnyBoolean as _AnyBoolean

from .. import \
    Die as _Die, \
    Slot as _Slot, \
    Column as _Column, \
    Yamb as _Yamb

def boost_die (cls = _Die, class_name = None):
    """Boosts a die class by enforcing using the NumPy back-end.

When NumPy back-end is enforced, `cls.sides` is converted to a `numpy.ndarray`
and `random_state` is set to a `numpy.random.Generator` unless a different
random state is explicitly provided.  Converting `cls.sides` is not done
in-place, meaning that the original `cls`' class variable `sides` is remained
unaltered, but the resulting class' variable `sides` is set to a
`numpy.ndarray` version of `cls.side`.

Additionally, the resulting class has a class variable `boosted` set to a
value of ``True``.  This is set so one can easily indentify if the class is
boosted or not.  It is not, however, used to determine if the provided class
needs boosting or not: the provided class is always boosted, even if it has
already been boosted.

Parameters
----------
cls : type[Die], default = Die
    Class to boost.

class_name : string, optional
    Name to set as the `__name__` property of the resulting class.

Returns
-------
type[Die]
    The boosted subclass version of `cls`.

Raises
------
TypeError
    If `cls` is not a subclass of `Die` class.  If `class_name` is not a
    string.

Notes
-----
This method returns a dynamically created class.  Do not serialise and
deserialise (e. g. using `pickle`) instances of the resulting class as it may
lead to unexpected/undesired behaviour; only use the instances on a single
thread / in a single process as initiated.  If you need to serialise a boosted
class instance, please refer to the code of this method and implement a custom
static boosted class.
"""
    if not (isinstance(cls, type) and issubclass(cls, _Die)):
        raise TypeError("Base class must be a die subclass.")
    if not (class_name is None or isinstance(class_name, _AnyString)):
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

        def __init__ (self, random_state = None, **kwargs):
            if random_state is None:
                super(BoostedDie, self).__init__(
                    random_state = _np.random.default_rng(),
                    **kwargs
                )
            elif isinstance(random_state, _AnyNumber):
                super(BoostedDie, self).__init__(
                    random_state = _np.random.default_rng(random_state),
                    **kwargs
                )
            else:
                super(BoostedDie, self).__init__(
                    random_state = random_state,
                    **kwargs
                )

    return BoostedDie

def boost_column (
    cls = _Column,
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

The `__array__` method now simply returns the `_slots` instance variable, and
checking for `numpy` dependency is omitted from the `to_numpy` method.  Also,
a check for `numpy` and `pandas` dependencies is removed from the `to_pandas`
method.

Additionally, the resulting class has a class variable `boosted` set to a
value of ``True``.  This is set so one can easily indentify if the class is
boosted or not.  It is not, however, used to determine if the provided class
needs boosting or not: the provided class is always boosted, even if it has
already been boosted.

Parameters
----------
cls : type[Column], default = Column
    Class to boost.

max_cache_size : integer, default = 32
    Maximum cache size to provide to `functools.lru_cache` decorator.  See
    Notes for more details.

class_name : string, optional
    Name to set as the `__name__` property of the resulting class.

Returns
-------
type[Column]
    The boosted subclass version of `cls`.

Raises
------
TypeError
    If `cls` is not a subclass of `Column` class.  If `max_cache_size` is not
    an integral value.  If `class_name` is not a string.

ValueError
    If `max_cache_size` is less than 0.

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
methods should expect integers and/or maybe short strings, while the latter
two should also expect quite small structures (flat or composite sequences) of
integers in a standard game with 5 dice.
"""
    if not (isinstance(cls, type) and issubclass(cls, _Column)):
        raise TypeError("Base class must be a column subclass.")
    if not isinstance(max_cache_size, _AnyInteger):
        raise TypeError("Cache size must be an integral value.")
    if not (class_name is None or isinstance(class_name, _AnyString)):
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

        lambda_score = float('nan')

        @classmethod
        def _new_empty_scores (cls):
            return _np.full(
                len(_Slot),
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
        def _count_hashable_results (cls, results):
            """Similar to `_count_results`, but expects a hashable sequence \
`results` (e. g. a `tuple`) for caching purposes."""
            return super(BoostedColumn, cls)._count_results(results)

        @classmethod
        def _count_results (cls, results):
            return cls._count_hashable_results(
                results if isinstance (results, _AnyHashable)
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
                available_slots = _np.intersect1d(
                    available_slots,
                    self._type.fillable_slots_array,
                    assume_unique = True
                )

                self._available_slots = available_slots

            return self._available_slots

        def to_numpy (self):
            return self.__array__()

        def to_pandas(self, str_index = False):
            if self._check_input and not isinstance(str_index, _AnyBoolean):
                raise TypeError("String index flag must be a boolean value.")

            return _pd.Series(
                self.__array__(),
                index = \
                    self._type.slots_str_index if str_index \
                        else self._type.slots_index,
                name = self._name
            )

        def __array__ (self):
            return self._slots

    return BoostedColumn

def boost_yamb (cls = _Yamb, class_name = None):
    if not (isinstance(cls, type) and issubclass(cls, _Yamb)):
        raise TypeError("Base class must be a yamb subclass.")
    if not (class_name is None or isinstance(class_name, _AnyString)):
        raise TypeError("Class name must be a string.")

    if class_name is not None:
        class_name = str(class_name)

    class BoostedYamb (cls):
        if class_name is not None:
            __name__ = class_name

        __doc__ = cls.__doc__

        boosted = True

        @classmethod
        def _replace_results (
            cls,
            old_results,
            new_results,
            replace,
            sort = False
        ):
            if not isinstance(old_results, _np.ndarray):
                old_results = _np.asanyarray(old_results)
            if replace is None:
                return old_results

            old_results[replace] = new_results
            if sort:
                if old_results.flags.owndata and old_results.flags.writeable:
                    old_results.sort()
                else:
                    old_results = _np.sort(old_results)

            return old_results

        @classmethod
        def _new_empty_results (cls, n):
            if isinstance(n, _AnyIterable):
                if not isinstance(n, _AnySequence):
                    n = list(n)
                n = _np.asanyarray(n)
                n.fill(0)
            else:
                n = _np.zeros(n, dtype = _np.int32)

            return n

    return BoostedYamb
