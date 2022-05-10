# -*- coding: utf-8 -*-

# NOTE. Many methods in `Column` class and its subclasses could have been
# implemented using `lambda`, `filter`, `map` and `functools.partial` objects
# which would represent a noticeable optimisation if the underlying iterables
# were much longer.  However, due to the fact that all iterables involved have
# 17 items at the most, such solutions were sometimes even slower than custom
# `for`-loops (actual or inline `generator`s), or represented a negligible
# micro-optimisations.  If the code seems slow, consider:
#
# * skipping input checks in `Column` instances (setting `check_input` to
#     ``False`` in the initialiser),
# * using the `FiniteDie` subclass instead of the regular `Die` class for
#     dice,
# * using the NumPy back-end (e. g. setting a NumPy (pseudo-)random number
#     generator as the random state of a die),
# * using boosted dynamic classes from `booster` submodule, which are
#     automatically implemented on the NumPy back-end but also include some
#     additional optimisations.

import abc as _abc
import collections as _collections
_collections_abc = getattr(_collections, 'abc', _collections)
import csv as _csv
import enum as _enum
import io as _io
import os as _os
import random as _random
import sys as _sys

_np = None
_pd = None
try:
    import numpy as _np
except ImportError:
    pass
try:
    import pandas as _pd
except ImportError:
    pass

from ._types import \
    AnyIterable as _AnyIterable, \
    AnyCollection as _AnyCollection, \
    AnySequence as _AnySequence, \
    AnyString as _AnyString, \
    AnyBytes as _AnyBytes, \
    AnyNumber as _AnyNumber, \
    AnyInteger as _AnyInteger, \
    AnyBoolean as _AnyBoolean, \
    AnyRandomState as _AnyRandomState, \
    RandomState as _RandomState, \
    NumpyBitGenerator as _NumpyBitGenerator, \
    NumpyRandomState as _NumpyRandomState

_range = xrange if _sys.version_info.major < 3 else range

_iterkeys = lambda m: getattr(m, 'iterkeys', m.keys)()
_itervalues = lambda m: getattr(m, 'itervalues', m.keys)()
_iteritems = lambda m: getattr(m, 'iteritems', m.items)()

class Die (object):
    """Represents a rollable 6-sided game die.

Parameters
----------
random_state : Die or callable or number or random.Random or \
numpy.random.RandomState or numpy.random.Generator or \
numpy.random.BitGenerator or module[random] or module[numpy.random], optional
    Random state of the die (for rolling results).  If callable, it should
    support a single parameter `n` and return a sinlge integer from range
    [1..6] if `n` is ``None`` or a sequence of length `n` of such numbers if
    `n` is a non-negative integer.

Notes
-----
The class `Die` provides a class variable `sides` which is originally a
standard Python list of integers 1 through 6.  It is used by all dice to draw
choices from, unless the die was initiated with a custom (pseudo-)random
number generating function.  If NumPy backend is used (if `random_state`s of
dice are NumPy generators), the variable may be converted to a `numpy.ndarray`
like this:

    >>> Die.sides # <- used only to demonstrate, you may skip this step
    [1, 2, 3, 4, 5, 6]
    >>> Die.sides = numpy.array(Die.sides)
    >>> Die.sides # <- used only to demonstrate, you may skip this step
    array([1, 2, 3, 4, 5, 6])
"""

    sides = [ 1, 2, 3, 4, 5, 6 ]

    def __new__ (cls, *args, **kwargs):
        instance = super(Die, cls).__new__(cls)

        instance._type = None
        instance._random_state = None
        instance._roller = None

        return instance

    def __init__ (self, random_state = None):
        super(Die, self).__init__()

        if (
            isinstance(random_state, type) and
            issubclass(random_state, _AnyRandomState)
        ):
            random_state = random_state()

        self._type = self.__class__

        if random_state is None:
            self._random_state = _random.Random()
        elif isinstance(random_state, Die):
            self._random_state = random_state._random_state
            self._roller = random_state._roller
        elif isinstance(random_state, _AnyNumber):
            self._random_state = _random.Random(random_state)
        elif (
            random_state is _random or
            (_np is not None and random_state is _np.random) or
            isinstance(random_state, _AnyRandomState)
        ):
            self._random_state = random_state
        elif (
            _np is not None and
            _NumpyBitGenerator and
            isinstance(random_state, _NumpyBitGenerator)
        ):
            self._random_state = _np.random.default_rng(random_state)
        elif (
            callable(random_state) or
            isinstance(random_state, _collections_abc.Callable)
        ):
            self._random_state = random_state
            self._roller = self._random_state
        else:
            raise TypeError(
                "Random state `{random_state}` is not understood.".format(
                    random_state = repr(random_state)
                )
            )

        if self._roller is None:
            if (
                self._random_state is _random or
                isinstance(self._random_state, _RandomState)
            ):
                if _sys.version_info.major < 3:
                    self._roller = \
                        lambda n: \
                            self._random_state.choice(self._type.sides) \
                                if n is None \
                                    else list(
                                        self._random_state.choice(
                                            self._type.sides
                                        ) for _ in _range(n)
                                    )
                else:
                    self._roller = \
                        lambda n: \
                            self._random_state.choice(self._type.sides) \
                                if n is None \
                                    else self._random_state.choices(
                                        self._type.sides,
                                        k = n
                                    )
            elif (
                _np is not None and
                (
                    self._random_state is _np.random or
                    isinstance(self._random_state, _NumpyRandomState)
                )
            ):
                self._roller = \
                    lambda n: \
                        self._random_state.choice(
                            self._type.sides,
                            size = n,
                            replace = True
                        )

    def roll (self, n = None):
        """Rolls the die.

Requesting multiple results via the parameter `n` may be interpreted either as
rolling the die multiple times or rolling multiple dice simultaneously.
Consequently, the results are not necessarily mutually different.

Parameters
----------
n : integer, optional
    Number of results to generate.  If ``None``, a single result is generated.

Returns
-------
integer or sequence[integer]
    Result of the roll, i. e. integer(s) from range [1..6].  If `n` is
    ``None``, a single result is returned; otherwise a sequence of `n` results
    is returned.

Notes
-----
If NumPy is available and `random_state` is a NumPy (pseudo-)random number
generator, `n` may be a tuple of integers in which case an array of such shape
is returned.  This is the main reason why the parameter `n` is not checked but
passed as is to the underlying number generator function.  However, do not
rely on this behaviour being preserved in extension subclasses of the `Die`
class.
"""
        return self._roller(n)

    @property
    def random_state (self):
        """Random state of the die."""
        return self._random_state

class FiniteDie (Die):
    """Represents a die with predefined rolling results.

Generating a small amount of (pseudo-)random numbers, especially a single
number each time, many times, may involve a lot of seemingly unneeded work and
may therefore be replaced with generating a large amount of numbers at once
and then sampling from them as a queue.  This is the main idea behind this
extension of the `Die` class: a predefined amount of rolls is generated at the
die initialisation, and then each call to the `roll` method simply returns a
subsample of the requested size that is next in line.

The subsampling is further optimised if NumPy back-end is used since no memory
allocation and object copying is involved for generating the subsample.  The
returned subsamples are merely views to parts of the original total sample.

Parameters
----------
size : integer
    The size of the initial total sample that is generated.

random_state
    Please refer to the documentation for `Die` for more details.

Notes
-----
Do not rely on a die alerting its pregenerated total sample is used up, or any
behaviour of the die in such cases for that matter.  Instead, make sure the
size of the sample is large enough for the die's purposes, or keep track of
the remaining sample and reset the die's inner state when needed.  Use `size`
and `remaining` properties, as well as the `reset` method, for this.
"""

    def __new__ (cls, *args, **kwargs):
        instance = super(FiniteDie, cls).__new__(cls)

        instance._size = None
        instance._results = None
        instance._index = None

        return instance

    def __init__ (self, size = 780, random_state = None):
        super(FiniteDie, self).__init__(random_state = random_state)

        if not isinstance(size, _AnyInteger):
            raise TypeError("Size must be an integral value.")
        if size < 0:
            raise ValueError("Size must be greater than or equal to 0.")

        self._size = int(size)
        self._results = self._roller(self._size)
        self._index = 0

    def roll (self, n = None):
        result = None

        if n is None:
            result = self._results[self._index]
            self._index += 1
        else:
            new_index = self._index + n
            result = self._results[self._index:new_index]
            self._index = new_index

        return result

    def reset (self):
        """Resets the total sample of the die.

After the method is called, the die once again offers a total of `size` roll
results as when it was initialised, but reseting an existing die might
sometimes be more convenient than initialising a new die.
"""
        self._results = self._roller(self._size)
        self._index = 0

    @property
    def size (self):
        """Size of the total roll results sample held by the die (including \
the already returned subsamples)."""
        return self._size

    @property
    def remaining (self):
        """The remaining amount of roll results still available (not yet \
returned)."""
        return self._size - self._index

@_enum.unique
class Slot (_enum.IntEnum):
    """Represents slots in a column of a yamb game table.

Although implementation independence is not a standard in object-oriented
programming, it is guaranteed that all values in the `Slot` enumeration
represent unique and valid non-negative zero-indexed indices of a sequence of
length ``len(Slot)``, i. e. 17.  Furthermore, when iterated over
``iter(Slot)``, their values go from 0 to 16 and they appear in the same order
as they are usually listed in yamb game tables (numbers and their sum;
maximum, minimum and their difference; special collections and their sum) with
an exception that the grand total appears first and corresponds to index 0.
The exception was made to ensure that slots `ONE`, `TWO`, ..., `SIX`
correspond to index values 1, 2, ..., 6 but also that no index is skipped.
"""

    # Grand total (sum of partial sums)
    TOTAL = 0

    # Numbers
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    NUMBERS_SUM = 7 # Sum of fillable number solts

    # Sums
    MAX = 8
    MIN = 9
    SUMS_DIFFERENCE = 10 # Difference of fillable sum slots multiplied by the
                         # number of ones

    # Collections
    TWO_PAIRS = 11
    STRAIGHT = 12
    FULL_HOUSE = 13
    CARRIAGE = 14
    YAMB = 15
    COLLECTIONS_SUM = 16 # Sum of fillable collection slots

class Column (object if _sys.version_info.major < 3 else _abc.ABC):
    """Represents a column to fill in the yamb game table.

This is an abstract class and cannot be used for any actual yamb game column.
To implement a usable column, extend this class and implement instance methods
`get_next_available_slots` and `is_slot_next_available`.  Additionally, one
may override methods `lock`, `unlock`, `requires_pre_filling_action`,
`pre_filling_action`, `requires_post_filling_action` and `post_filling_action`
for special column behaviour (e. g. announced column).

Parameters
----------
name : string, optional
    Name of the column.  If not provided, the name is inferred from the
    column's type.

check_input : boolean, default = True
    If ``True``, arguments are checked in all methods before conducting any
    business logic; otherwise all arguments are assumed to be valid (by type
    and value) and legal (by game and column rules).

Notes
-----
Argument checks done when `check_input` is set to ``True`` may represent an
unwanted decline in code optimisation and increase in time consumption when
validity of arguments is ensured in external code.  For example, when training
an automated computer player in a large number of game iterations, such
obstacles may sum up to a significant slow down of the algorithm although the
player was programmed to follow the rules.  Therefore, an unsafe alternative
is provided to the user by default without the need for manual alteration of
the original class and "hacking" the original code, but this should not be
used in production code for human or automated players which may intentionally
or unintentionally cheat.

This class also provides some auxiliary class variables such as:

* `number_slots`: a set of slots `Slot.ONE` through `Slot.SIX`,
* `sum_slots`: a set of slots `Slot.MAX` and `Slot.MIN`,
* `collection_slots`: a set of collection slots (`Slot.TWO_PAIRS` through
    `Slot.YAMB`),
* `fillable_slots`: a set of all fillable slots (union of the previous three),
* `inner_auto_slots`: a set of slots calculated directly from scores of
    fillable slots, i. e. automatic slots excluding the grand total
    (`Slot.NUMBERS_SUM`, `Slot.SUMS_DIFFERENCE` and `Slot.COLLECTIONS_SUM`),
* `outer_auto_slots`: a set of slots calculated from other slots but not
    directly from fillable slots, i. e. the grand total (`Slot.TOTAL`),
* `auto_slots`: a set of slots calculated from scores of fillable slots
    (union of the previous two),
* `slots`: a set of all slots (union of `fillable_slots` and `auto_slots`).

The former slots are all instances of the built-in `frozenset` class and are
intended not to be altered, but only used for checks and to help implementing
custom columns.  Additionally, if NumPy is available, for each set of slots
there is also a `numpy.ndarray` called `*_array`
(e. g. `fillable_slots_array`), which is a one-dimensional integer array (slot
index values) sorted ascendingly.  If Pandas is available, there are also
`pandas.Index`es called `*_index` and `*_str_index`, of which the former is an
integer index (similar to the array but a `pandas.Index`), and the second is a
string index (slot names).

Another special class variable is the `lambda_score`, originally set to
``None``, which is used to represent a score that is not yet filled in.  If
this value is overriden, override methods `is_lambda` and perhaps
`get_lambda_slots` as well.  Do not use 0 for the `lambda_score` since 0
should indicate an actual score of 0.  Similarly, any non-zero possible score
should also be avoided.

The class also implements auxiliary class methods whose names start with an
uderscore.  These methods are encouraged to be used by a subclass and may even
be overriden for special optimisations (e. g. to ensure NumPy back-end,
argument-result caching etc.).  However, when overriding them, make sure the
outputs are compatible with the original outputs to prevent breaking the
intended behaviour.

Instance variables defined and used by the `Column` class are the following:

* `_type`: actual class of the current column (`type`; originally
    ``self.__class__``; used for referencing the correct class methods in case
    of overrides and may be altered to redirect class method calls),
* `_rollable`: a flag indicating whether or not the column allows next dice
    rolls (`bool`; see `disallow_rolls`, `allow_rolls` and `can_roll` methods
    for more information),
* `_locked`: a flag indicating whether or not the column is locked (`bool`;
    see `lock`, `unlock` and `is_locked` methods for more information)
* `_name`: name of the column (`str`),
* `_check_inputs`: a flag indicating whether or not method arguments should be
    checked (`bool`),
* `_scores`: list of scores per slots (`list` of `int`s and `lambda_score`),
* `_available_slots`: list of unfilled fillable slots (``None`` originally;
    see below for more information),
* `_next_available_slots`: list of unfilled slots fillable in the next turn
    (``None`` originally; see below for more information).

The last two instance variables are intended for optimisation of methods
`get_available_slots` and `get_next_available_slots` respectively.  When a
method of those two is called, the column should check if the corresponding
variable is ``None``.  If yes, the list should be generated, saved to the
corresponding variable and returned; if not, the variable's value should be
returned immediately (optionally, return a copy to prevent unwanted mutation
from the outside code).  Both variables are originally set to ``None``, and
are reset to ``None`` on every call to the `fill_slot` method.  If a
pre-filling action or a post-filling action changes the availability of any of
the two lists (e. g. for the announced column, only the announced slot is
available after the announced up until the end of the turn), set the
appropriate variable to ``None`` again in the corresponding method
(`pre_filling_action` or `post_filling_action`).

Additional to instance properties `name`, `check_input` and `score`, any
`Slot` value's name may be used as a property for getting the score of the
slot like the following: ``self.ONE`` or ``self.TOTAL``.  Such score retrieval
is case-sensitive, unlike the subscripting retrieval via the `__getitem__`
method.
"""

    if _sys.version_info.major < 3:
        __metaclass__ = _abc.ABCMeta

    number_slots = frozenset(
        [
            Slot.ONE,
            Slot.TWO,
            Slot.THREE,
            Slot.FOUR,
            Slot.FIVE,
            Slot.SIX
        ]
    )
    sum_slots = frozenset(
        [
            Slot.MAX,
            Slot.MIN
        ]
    )
    collection_slots = frozenset(
        [
            Slot.TWO_PAIRS,
            Slot.STRAIGHT,
            Slot.FULL_HOUSE,
            Slot.CARRIAGE,
            Slot.YAMB
        ]
    )
    fillable_slots = number_slots | sum_slots | collection_slots
    inner_auto_slots = frozenset(
        [
            Slot.NUMBERS_SUM,
            Slot.SUMS_DIFFERENCE,
            Slot.COLLECTIONS_SUM
        ]
    )
    outer_auto_slots = frozenset(
        [
            Slot.TOTAL
        ]
    )
    auto_slots = inner_auto_slots | outer_auto_slots
    slots = fillable_slots | auto_slots

    if _np is not None:
        number_slots_array = _np.array(
            list(sorted(number_slots)),
            dtype = _np.int32
        )
        sum_slots_array = _np.array(
            list(sorted(sum_slots)),
            dtype = _np.int32
        )
        collection_slots_array = _np.array(
            list(sorted(collection_slots)),
            dtype = _np.int32
        )
        fillable_slots_array = _np.array(
            list(sorted(fillable_slots)),
            dtype = _np.int32
        )
        inner_auto_slots_array = _np.array(
            list(sorted(inner_auto_slots)),
            dtype = _np.int32
        )
        outer_auto_slots_array = _np.array(
            list(sorted(outer_auto_slots)),
            dtype = _np.int32
        )
        auto_slots_array = _np.array(
            list(sorted(auto_slots)),
            dtype = _np.int32
        )
        slots_array = _np.array(
            list(sorted(slots)),
            dtype = _np.int32
        )
        number_slots_array.flags.writeable = False
        sum_slots_array.flags.writeable = False
        collection_slots_array.flags.writeable = False
        fillable_slots_array.flags.writeable = False
        inner_auto_slots_array.flags.writeable = False
        outer_auto_slots_array.flags.writeable = False
        auto_slots_array.flags.writeable = False
        slots_array.flags.writeable = False

    if _pd is not None:
        _pd_1_or_higher = \
            _np.lib.NumpyVersion(_pd.__version__) >= _np.lib.NumpyVersion('1.0.0')
        _int_dtype = int if _np is None else _np.int32
        _str_dtype = _pd.StringDtype() if _pd_1_or_higher else str

        number_slots_index = _pd.Index(
            list(sorted(number_slots)),
            dtype = _int_dtype,
            name = 'Slot'
        )
        sum_slots_index = _pd.Index(
            list(sorted(sum_slots)),
            dtype = _int_dtype,
            name = 'Slot'
        )
        collection_slots_index = _pd.Index(
            list(sorted(collection_slots)),
            dtype = _int_dtype,
            name = 'Slot'
        )
        fillable_slots_index = _pd.Index(
            list(sorted(fillable_slots)),
            dtype = _int_dtype,
            name = 'Slot'
        )
        inner_auto_slots_index = _pd.Index(
            list(sorted(inner_auto_slots)),
            dtype = _int_dtype,
            name = 'Slot'
        )
        outer_auto_slots_index = _pd.Index(
            list(sorted(outer_auto_slots)),
            dtype = _int_dtype,
            name = 'Slot'
        )
        auto_slots_index = _pd.Index(
            list(sorted(auto_slots)),
            dtype = _int_dtype,
            name = 'Slot'
        )
        slots_index = _pd.Index(
            list(sorted(slots)),
            dtype = _int_dtype,
            name = 'Slot'
        )

        number_slots_str_index = _pd.Index(
            list(s.name for s in sorted(number_slots)),
            dtype = _str_dtype,
            name = 'Slot'
        )
        sum_slots_str_index = _pd.Index(
            list(s.name for s in sorted(sum_slots)),
            dtype = _str_dtype,
            name = 'Slot'
        )
        collection_slots_str_index = _pd.Index(
            list(s.name for s in sorted(collection_slots)),
            dtype = _str_dtype,
            name = 'Slot'
        )
        fillable_slots_str_index = _pd.Index(
            list(s.name for s in sorted(fillable_slots)),
            dtype = _str_dtype,
            name = 'Slot'
        )
        inner_auto_slots_str_index = _pd.Index(
            list(s.name for s in sorted(inner_auto_slots)),
            dtype = _str_dtype,
            name = 'Slot'
        )
        outer_auto_slots_str_index = _pd.Index(
            list(s.name for s in sorted(outer_auto_slots)),
            dtype = _str_dtype,
            name = 'Slot'
        )
        slots_str_index = _pd.Index(
            list(s.name for s in sorted(slots)),
            dtype = _str_dtype,
            name = 'Slot'
        )

        del _pd_1_or_higher
        del _str_dtype

    lambda_score = None

    @classmethod
    def _new_empty_scores (cls):
        """Creates and returns a new empty score list.

Returns
-------
list
    A list of ``len(Slot)`` empty scores (`lambda_score`).
"""
        return list(cls.lambda_score for _ in _range(len(Slot)))

    @classmethod
    def _ensure_roll_index (cls, roll):
        """Ensures a valid roll index.

Parameters
----------
roll : integer
    One-indexed roll index.  If 0, it indicates a state before the first roll.

Returns
-------
int
    The value of the original `roll` converted to a Python `int`.

Raises
------
TypeError
    If `roll` is not an integral value.

ValueError
    If `roll` is not greater than or equal to 0.
"""
        if not isinstance(roll, _AnyInteger):
            raise TypeError("Roll index must be an integral numerical value.")
        if roll < 0:
            raise ValueError("Roll index must be greater than or equal to 0.")

        return int(roll)

    @classmethod
    def _ensure_slot (cls, slot):
        """Ensures a valid slot.

Parameters
----------
slot : Slot or integer or string
    Slot index.

Returns
-------
Slot
    The value of the original `slot` converted to a `Slot` value.

Raises
------
TypeError
    If `slot` is not of a supported type.

Notes
-----
Additional errors might be raised (and are not caught by the method) when
converting an integer or a string to a `Slot` via ``Slot(slot)`` and
``Slot[slot.upper()]`` calls respectively.
"""
        if isinstance(slot, _AnyString):
            return Slot[slot.upper()]
        if not (
            isinstance(slot, Slot) or
            isinstance(slot, _AnyInteger)
        ):
            raise TypeError("Slot must be an integral numerical value.")

        if slot < 0:
            slot = len(Slot) + slot

        return Slot(slot)

    @classmethod
    def _ensure_results_sequence (cls, results):
        """Ensures a valid results sequence.

Parameters
----------
results : integer or iterable[integer]
    Dice roll results.

Returns
-------
sequence[integer]
    The value of the original `results` converted to a sequence.

Raises
------
TypeError
    If `results` does not represent a sequence of integers.

ValueError
    If a result is not in the range [1..6].

Notes
-----
Although `numpy.ndarray` is not a subclass of `collections.abc.Sequence`
abstract base class, it is considered a sequence by this method and is not
converted to a `tuple` or a `list`.
"""
        if (
            not isinstance(results, _AnyIterable) or
            isinstance(results, _AnyString)
        ):
            results = (results, )
        if not isinstance(results, _AnySequence):
            results = tuple(results)
        if _np is not None and isinstance(results, _np.ndarray):
            if not results.ndim == 1:
                raise ValueError("Results must be a 1-dimensional sequence.")
            if not _np.issubdtype(results.dtype, _np.integer):
                raise TypeError("Results must be integral values.")
            if not _np.all((results >= 1) & (results <= 6)):
                raise ValueError("Results must be in range [1..6].")
        else:
            for r in results:
                if not isinstance(r, _AnyInteger):
                    raise TypeError("Results must be integral values.")
                if not 1 <= r <= 6:
                    raise ValueError("Results must be in range [1..6].")

        return results

    @classmethod
    def _count_results (cls, results):
        """Counts dice roll results.

Parameters
----------
results : sequence[integer]
    Results of a dice roll.

Returns
-------
tuple[integer]
    Ordered (ascending) unique results of the dice roll.

tuple[tuple[integer, int]]
    Ordered (descending colexicographically) unique counts of results of the
    original dice roll.  Each item represents a pair ``(result, count)``.

See Also
--- ----
_ensure_results_sequence : Method that prepares parameter for this method \
from original dice roll results

Notes
-----
This method assumes valid parameter type and values.  Use
`_ensure_results_sequence` to prepare a raw dice roll results before passing
them to this method.
"""
        counts = _collections.Counter(results)
        results = tuple(sorted(_iterkeys(counts)))
        counts = tuple(
            sorted(
                _iteritems(counts),
                key = lambda c: tuple(reversed(c)),
                reverse = True
            )
        )

        return (results, counts)

    @classmethod
    def _evaluate (cls, slot, results, counts):
        """Evaluates results for a slot.

Parameters
----------
slot : integer or Slot
    The slot to evaluate.

results : sequence[integer]
    Ordered (ascending) unique results of a dice roll.

counts : squence[sequence[integer, integer]]
    Ordered (descending colexicographically) unique counts of results of the
    original dice roll.  Each item represents a pair ``(result, count)``.

Returns
-------
integer
    Score of the `results` and `counts` for the given `slot`.

See Also
--- ----
_count_results : Method that prepares parameters for this method from \
original dice roll results

Notes
-----
This method not only assumes valid parameter types, but also their values
(including ordering).  Because a sequence of `tuple`s is assumed for `counts`
rather than a `dict`, do not pass a `dict` because that may lead to undesired
behaviour.

Use `_count_results` method to build parameters for the method and do not
alter them.
"""
        if slot in cls.number_slots:
            for r, c in counts:
                if r == slot:
                    return c * r
            return 0
        elif slot in cls.sum_slots:
            return sum(map(lambda c: c[0] * c[1], counts))
        elif slot == Slot.TWO_PAIRS:
            return \
                2 * (counts[0][0] + counts[1][0]) + 10 \
                    if (
                        len(counts) >= 2 and
                        min(counts[0][1], counts[1][1]) >= 2
                    ) \
                        else 0
        elif slot == Slot.STRAIGHT:
            n = 0
            for i in _range(1, len(results)):
                if results[i] - results[i - 1] == 1:
                    n += 1
                    if n >= 4:
                        return 10 * results[i - n] + 25
                else:
                    n = 0
            return 0
        elif slot == Slot.FULL_HOUSE:
            return \
                3 * counts[0][0] + 2 * counts[1][0] + 30 \
                    if (
                        len(counts) >= 2 and
                        counts[0][1] >= 3 and
                        counts[1][1] >= 2
                    ) \
                        else 0
        elif slot == Slot.CARRIAGE:
            return \
                4 * counts[0][0] + 40 \
                    if (counts and counts[0][1] >= 4) \
                        else 0
        elif slot == Slot.YAMB:
            return \
                5 * counts[0][0] + 50 \
                    if (counts and counts[0][1] >= 5) \
                        else 0
        else:
            raise KeyError(
                "Slot {slot} is not recognised.".format(slot = slot)
            )

    @classmethod
    def is_lambda (cls, score):
        """Checks if the `score` is undefined (lambda score).

Parameters
----------
score
    The score to evaluate.

Return
------
bool
    ``True`` if the score is undefined, ``False`` otherwise.

See Also
--- ----
lambda_score : Value that represents an undefined score
get_lambda_slots : Method that returns slots with undefined scores from a \
list of scores

Notes
-----
For optimisation purposes, this method does not check parameter type but only
its value.
"""
        return score is None

    @classmethod
    def get_lambda_slots (cls, scores):
        """Returns slots of undefined scores (lambda scores).

Parameters
----------
scores : sequence
    The scores to evaluate.  It must be a sequence of length ``len(Slot)``,
    i. e. 17.

Return
------
list[Slot]
    List of all slots, i. e. indices of `scores` (fillable or not) with
    undefined values for scores.

See Also
--- ----
lambda_score : Value that represents an undefined score
is_lambda_score : Method that checks if a score is undefined

Notes
-----
For optimisation purposes, this method does not check parameter type but only
its values.
"""
        return tuple(s for s in Slot if cls.is_lambda(scores[s]))

    @classmethod
    def display (cls, columns, allow_numpy = True, allow_pandas = True):
        """Returns a "nice" way to display columns.

This function is intended for printing/displaying single and multiple columns
in a human-friendly readable format.  The returned value might not always be a
string (`str`), but calling ``print(Column.display(columns))`` in console or
``display(Column.display(columns))`` in IPython/Jupyter should give you the
best output in the operating environment.

Parameters
----------
columns : Column or iterable[Column]
    Column(s) to display.

allow_numpy : boolean, default = True
    If ``True`` and NumPy is available, the returned value may be a
    `numpy.ndarray`.

allow_pandas : boolean, default = True
    If ``True`` and Pandas is available, the returned value may be a
    `pandas.Series` or a `pandas.DataFrame`.

Returns
-------
str or unicode or numpy.ndarray or pandas.Series or pandas.DataFrame
    The appropriate representation of the `columns`.

Notes
-----
The order of preference for the returned value is the following:

1. Pandas objects (`pandas.Series` or `pandas.DataFrame`),
2. `numpy.ndarray`,
3. `str` or `unicode` (the latter only in Python 2).

Setting flags for Pandas and NumPy to ``False`` eliminates them from the order
of preference, even if the optional dependencies are available.

A `pandas.Series` may only be returned if Pandas is both allowed and available
and if `columns` is a single column (but not an iterable containing a single
column).

When a string is returned, it is actually a tab-separated (``"\\t"``) vertical
table.  Usual CSV/TSV/DSV (comma-, tab- or delimiter-separated values) rules
apply while building the table.
"""
        single_column = False

        if isinstance(columns, Column):
            columns = (columns, )
            single_column = True
        else:
            columns = list(columns)
            for c in columns:
                if not isinstance(c, Column):
                    raise TypeError(
                        "Columns must be of type `Column`, column " \
                            "`{column}` not understood.".format(
                                column = c
                            )
                    )

        if not isinstance(allow_numpy, _AnyBoolean):
            raise TypeError("Allow NumPy flag must be a boolean value.")
        allow_numpy = bool(allow_numpy)

        if not isinstance(allow_pandas, _AnyBoolean):
            raise TypeError("Allow Pandas flag must be a boolean value.")
        allow_pandas = bool(allow_pandas)

        if allow_pandas and _pd is not None:
            table = None
            if single_column:
                table = _pd.Series(
                    columns[0],
                    index = cls.slots_str_index,
                    name = columns[0].name
                )
            elif len(columns):
                table = _pd.concat(
                    tuple(_pd.Series(c, name = c._name) for c in columns),
                    axis = 1,
                    ignore_index = True
                )
                table.index = cls.slots_str_index
            else:
                table = _pd.DataFrame(
                    list([] for _ in Slot),
                    index = cls.slots_str_index
                )

            return table

        if allow_numpy and _np is not None:
            table = None
            if len(columns):
                table = _np.column_stack(columns).astype(_np.object_)
                table = _np.insert(
                    table,
                    0,
                    list(s.name for s in Slot),
                    axis = 1
                )
            else:
                table = _np.array(
                    list([ s.name ] for s in Slot),
                    dtype = _np.object_
                )
            table = _np.insert(
                table,
                0,
                [ 'Slot' ] + list(c.name for c in columns),
                axis = 0
            )

            return table

        table = None
        with (
            _io.BytesIO() if _sys.version_info.major < 3 else _io.StringIO()
        ) as output:
            writer = _csv.writer(
                output,
                delimiter = "\t",
                doublequote = True,
                escapechar = None,
                lineterminator = _os.linesep,
                quotechar = "\"",
                quoting = _csv.QUOTE_MINIMAL
            )

            writer.writerow([ 'Slot' ] + list(c.name for c in columns))
            writer.writerows(
                list([ s.name ] + list(c[s] for c in columns) for s in Slot)
            )

            table = output.getvalue()
            if (
                isinstance(table, _AnyBytes) or
                not isinstance(table, _AnyString)
            ):
                table = table.decode('utf-8')

        return table

    def __new__ (cls, *args, **kwargs):
        instance = super(Column, cls).__new__(cls)

        instance._type = None
        instance._check_input = None
        instance._name = None
        instance._rollable = None
        instance._locked = None
        instance._slots = None
        instance._available_slots = None
        instance._next_available_slots = None

        return instance

    def __init__ (self, name = None, check_input = True):
        super(Column, self).__init__()

        self._type = self.__class__

        if not isinstance(check_input, _AnyBoolean):
            raise TypeError("Check input flag must be a boolean value.")
        self._check_input = bool(check_input)

        if not (name is None or isinstance(name, _AnyString)):
            raise TypeError("Column name must be a string value.")
        self._name = str(
            getattr(self._type, '__name__', 'Column') if name is None
                else name
        )

        self._rollable = True
        self._locked = False

        self._slots = self._type._new_empty_scores()

        self._available_slots = None
        self._next_available_slots = None

    def disallow_rolls (self):
        """Disallows next rolls.

When a column forbids next rolls, it must be immediately filled.  Usually the
column would be locked when next rolls are disallowed.  This is primarily
intended for the "from-hand" column.

Raises
------
TypeError
    If the column cannot disallow next rolls.

See Also
--- ----
allow_rols : Method that allows next rolls
can_roll : Method that checks if next rolls are allowed
lock : Method that locks the column
unlock : Method that unlocks the column
is_locked : Method that checks if the column is locked

Notes
-----
By default, this method raises a `TypeError`.  Override this method if the
column should be lockable/unlockable by setting the `_rollable` instance
variable to ``False``.  Also, when using the method, lock the column via the
`lock` method to prevent users from avoiding the rule.
"""
        raise TypeError(
            "{column} cannot disallow rolls.".format(column = self._type)
        )

    def allow_rolls (self):
        """Allows next rolls.

When a column allows next rolls, it is no longer obligatory to immediately
fill it.  Usually the column would be unlocked when next rolls are allowed.
This is primarily intended for the "from-hand" column.

Raises
------
TypeError
    If the column cannot allow next rolls.

See Also
--- ----
disallow_rols : Method that disallows next rolls
can_roll : Method that checks if next rolls are allowed
lock : Method that locks the column
unlock : Method that unlocks the column
is_locked : Method that checks if the column is locked

Notes
-----
By default, this method raises a `TypeError`.  Override this method if the
column should be lockable/unlockable by setting the `_rollable` instance
variable to ``True``.
"""
        raise TypeError(
            "{column} cannot allow rolls.".format(column = self._type)
        )

    def lock (self):
        """Locks the column.

When a column is locked, it must be filled at the end of a turn (therefore at
most a single column may be locked in a turn).  This is primarily intended for
the announced column.

Raises
------
TypeError
    If the column cannot be locked.

See Also
--- ----
unlock : Method that unlocks the column
is_locked : Method that checks if the column is locked

Notes
-----
By default, this method raises a `TypeError`.  Override this method if the
column should be lockable/unlockable by setting the `_locked` instance
variable to ``True``.
"""
        raise TypeError(
            "{column} cannot be locked.".format(column = self._type)
        )

    def unlock (self):
        """Unlocks the column.

When a column is unlocked, it no longer needs to be filled at the end of a
turn.  This is primarily intended for the announced column.

Raises
------
TypeError
    If the column cannot be unlocked.

See Also
--- ----
lock : Method that locks the column
is_locked : Method that checks if the column is locked

Notes
-----
By default, this method raises a `TypeError`.  Override this method if the
column should be lockable/unlockable by setting the `_locked` instance
variable to ``False``.
"""
        raise TypeError(
            "{column} cannot be unlocked.".format(column = self._type)
        )

    def can_roll (self):
        """Checks if the next dice roll is possible.

If a roll is not possible, the player should immediately proceed to filling
the slot.  This is primarily intended for the "from-hand" column.

Returns
-------
bool
    ``True`` if the next dice roll is allowed, ``False`` otherwise.

See Also
--- ----
lock : Method that locks the column
unlock : Method that unlocks the column
"""
        return self._rollable

    def is_locked (self):
        """Checks if the column is locked.

When a column is unlocked, it no longer needs to be filled at the end of a
turn.  When a column is unlocked, it no longer needs to be filled at the end
of a turn.  This is primarily intended for the announced column.

Returns
-------
bool
    ``True`` if the column is locked, ``False`` otherwise.

See Also
--- ----
lock : Method that locks the column
unlock : Method that unlocks the column
"""
        return self._locked

    def evaluate (self, slot, results):
        """Evaluates `results` for the given `slot`.

Parameters
----------
slot : Slot or integer or string
    The slot to evaluate.

results : integer or sequence[integer]
    Results of a dice roll.

Returns
-------
integer
    Score of the `results` for the given `slot`.

Raises
------
TypeError
    If `slot` is not of a supported type.  If `results` is not of a supported
    type.

ValueError
    If a result is not in the range [1..6].

KeyError
    If `slot` is not fillable.

See Also
--- ----
evaluate_all : Method that evaluates results for all fillable slots

Notes
-----
Additional errors might be raised (and are not caught by the method) when
converting an integer or a string to a `Slot` via ``Slot(slot)`` and
``Slot[slot.upper()]`` calls respectively.

Unlike the `_evaluate` class method, this method does not reuqire counted
unique results.  However, if method parameters are not checked (recall the
`_check_inputs` instance variable), `results` must be a sequence rather than a
single integer.  Also, if parameters are not checked, `slot` may not be a
string.
"""
        if self._check_input:
            slot = self._type._ensure_slot(slot)
            if slot not in self._type.fillable_slots:
                raise KeyError(
                    "Slot {slot} is auto-filled.".format(slot = slot)
                )
            results = self._type._ensure_results_sequence(results)

        results, counts = self._type._count_results(results)

        return self._type._evaluate(slot, results, counts)

    def evaluate_all (self, results):
        """Evaluates `results` for all fillable slots.

Parameters
----------
results : integer or sequence[integer]
    Results of a dice roll.

Returns
-------
iterable[tuple[Slot, int]]
    Scores of the `results` for all fillable slots.

Raises
------
TypeError
    If `results` is not of a supported type.

ValueError
    If a result is not in the range [1..6].

See Also
--- ----
evaluate : Method that evaluates results for a single slot

Notes
-----
Unlike the `_evaluate` class method, this method does not reuqire counted
unique results.  However, if method parameters are not checked (recall the
`_check_inputs` instance variable), `results` must be a sequence rather than a
single integer.
"""
        if self._check_input:
            slot = self._type._ensure_slot(slot)
            if slot not in self._type.fillable_slots:
                raise KeyError(
                    "Slot {slot} is auto-filled.".format(slot = slot)
                )
            if not self.is_slot_next_available(slot):
                raise RuntimeError(
                    "Slot {slot} is unavailable.".format(slot = slot)
                )
            results = self._type._ensure_results_sequence(results)

        results, counts = self._type._count_results(results)

        for s in Slot:
            if s in self._type.fillable_slots:
                yield (s, self._type._evaluate(s, results, counts))

    def get_available_slots (self):
        """Returns all unfilled fillable slots.

Returns
-------
list[Slot]
    List of fillable slots yet to be filled.

Notes
-----
Do not alter the object returned by this method.  Doing so might break the
expected behaviour of the column.  Instead, create a copy using
``copy.copy(slots)`` or ``copy.deepcopy(slots)`` before altering it.
"""
        if self._available_slots is None:
            self._available_slots = list(
                s
                    for s in self._type.get_lambda_slots(self._slots)
                        if s in self._type.fillable_slots
            )

        return self._available_slots

    @_abc.abstractmethod
    def get_next_available_slots (self):
        """Returns all unfilled slots fillable in the next turn.

Returns
-------
list[Slot]
    List of fillable slots that are unfilled but may be filled in the next
    turn.

Notes
-----
Do not alter the object returned by this method.  Doing so might break the
expected behaviour of the column.  Instead, create a copy using
``copy.copy(slots)`` or ``copy.deepcopy(slots)`` before altering it.

This is an abstract method.  Override it for the column-specific rules.
"""
        pass

    def is_slot_available (self, slot):
        """Checks if a slot is unfilled.

Parameters
----------
slot : Slot or integer or string
    Slot to check.

Raises
------
TypeError
    If `slot` is not of a supported type.

KeyError
    If `slot` is not fillable.

Returns
-------
bool
    ``True`` if the `slot` is unfilled, ``False`` otherwise.

Notes
-----
Additional errors might be raised (and are not caught by the method) when
converting an integer or a string to a `Slot` via ``Slot(slot)`` and
``Slot[slot.upper()]`` calls respectively.

If method parameters are not checked (recall the `_check_inputs` instance
variable), ``slot` may not be a string.
"""
        if self._check_input:
            slot = self._type._ensure_slot(slot)
            if slot not in self._type.fillable_slots:
                raise KeyError(
                    "Slot {slot} is auto-filled.".format(slot = slot)
                )

        return self._type.is_lambda(self._slots[slot])

    def is_slot_next_available (self, slot):
        """Checks if a slot is unfilled but may be filled in the next turn.

Parameters
----------
slot : Slot or integer or string
    Slot to check.

Raises
------
TypeError
    If `slot` is not of a supported type.

KeyError
    If `slot` is not fillable.

Returns
-------
bool
    ``True`` if the `slot` is unfilled but may be filled in the next turn,
    ``False`` otherwise.

Notes
-----
Additional errors might be raised (and are not caught by the method) when
converting an integer or a string to a `Slot` via ``Slot(slot)`` and
``Slot[slot.upper()]`` calls respectively.

If method parameters are not checked (recall the `_check_inputs` instance
variable), `slot` may not be a string.

Although this is not an abstract method, its default implementation may be far
from optimal.  It is advised to override this method for the column-specific
rules, because the current implementation actually returns
``slot in self.get_next_available_slots()``. Such implementation requires the
construction of a complete collection returned by the
`get_next_available_slots` method and then checks if `slot` is in it or not,
but for many purposes this is a great overload of work (and even memory) which
may be avoided through intelligent implementation of column-specific logic.
"""
        if self._check_input:
            slot = self._type._ensure_slot(slot)
            if slot not in self._type.fillable_slots:
                raise KeyError(
                    "Slot {slot} is auto-filled.".format(slot = slot)
                )

        return slot in self.get_next_available_slots()

    def requires_pre_filling_action (self, roll):
        """Checks if a pre-filling action is required for the column.

Parameters
----------
roll : integer
    One-indexed roll index.  If 0, the action is required before the first
    roll.

Returns
-------
See Notes for more details.

See Also
--- ----
pre_filling_action : Method that defines the action to perform before \
filling the column
requires_post_filling_action : Method that checks if a post-filling action \
is required
post_filling_action : Method that defines the action to perform after \
filling the column

Notes
-----
If a pre-filling action is required after the `roll`th dice roll, this method
should return:

* a falsey value (e. g. ``None`` or ``False``) if no action is required,
* ``True`` if an action is required, but no parameters should be passed to the
    `pre_filling_method` method,
* a tuple of `tuple` (`args`) and `dict` (`kwargs`) defining the parameters
    for the `pre_filling_action` method.

In the latter case, each argument should be represented by its intended type. 
For instance, if the column is announced and `roll` is 1, the method might
return ``((), {'announced': Slot})``.  The method may also return only one of
the two (e. g. only `args` or only `kwargs`) if the other is
unexpected/non-required, but in that case make sure the structure is non-empty
so it would not be considered falsey.  In fact, if the structure is empty,
that means that no parameter is expected and/or required, therefore the second
option (to return ``True``) may be employed.  By default, this method returns
``None``.

A reuqirement of a pre-filling action means that the action is required if the
player should choose to fill this column at the end of the turn.  It does not
mean that the action must be done if the player does not want to fill the
column.
"""
        return None

    def pre_filling_action (self, roll, *args, **kwargs):
        """Action to perform before filling the column.

Parameters
----------
roll : integer
    One-indexed roll index.  If 0, the action is performed before the first
    roll.

args, kwargs
    Additional parameters to use when implementing the method.

See Also
--- ----
requires_pre_filling_action : Method that checks if a pre-filling action is \
required
requires_post_filling_action : Method that checks if a post-filling action \
is required
post_filling_action : Method that defines the action to perform after \
filling the column

Notes
-----
This method is intended for columns that require an action to be performed
before filling the column.  A reuqirement of a pre-filling action means that
the action is required if the player should choose to fill this column at the
end of the turn.  It does not mean that the action must be done if the player
does not want to fill the column.

By default, this method does nothing.  Implement it for column-specific rules
if required (such as the announced column).
"""
        pass

    def fill_slot (self, slot, results):
        """Fills a slot.

Parameters
----------
slot : Slot or integer or string
    Slot to fill.

results : integer or sequence[integer]
    Results of a dice roll.

Raises
------
TypeError
    If `slot` is not of a supported type.

KeyError
    If `slot` is not fillable.

RuntimeError
    If `slot` is not available in the next (this) turn.

Notes
-----
Additional errors might be raised (and are not caught by the method) when
converting an integer or a string to a `Slot` via ``Slot(slot)`` and
``Slot[slot.upper()]`` calls respectively.

If method parameters are not checked (recall the `_check_inputs` instance
variable), ``slot` may not be a string and `results` must be a sequence rather
than a single integer.
"""
        if self._check_input:
            slot = self._type._ensure_slot(slot)
            if slot not in self._type.fillable_slots:
                raise KeyError(
                    "Slot {slot} is auto-filled.".format(slot = slot)
                )
            if not self.is_slot_next_available(slot):
                raise RuntimeError(
                    "Slot {slot} is unavailable.".format(slot = slot)
                )

        self._slots[slot] = self.evaluate(slot, results)

        self._available_slots = None
        self._next_available_slots = None

    def update_auto_slots (self):
        """Updates any auto-filled slots that may be filled (if all required \
slots are filled)."""
        if (
            self._type.is_lambda(self._slots[Slot.NUMBERS_SUM]) and
            not any(
                self._type.is_lambda(self._slots[s])
                    for s in self._type.number_slots
            )
        ):
            numbers_sum = sum(
                self._slots[s] for s in self._type.number_slots
            )
            if numbers_sum >= 60:
                numbers_sum += 30
            self._slots[Slot.NUMBERS_SUM] = numbers_sum
        if (
            self._type.is_lambda(self._slots[Slot.SUMS_DIFFERENCE]) and
            not (
                self._type.is_lambda(self._slots[Slot.ONE]) or
                any(
                    self._type.is_lambda(self._slots[s])
                        for s in self._type.sum_slots
                )
            )
        ):
            self._slots[Slot.SUMS_DIFFERENCE] = \
                self._slots[Slot.ONE] * \
                    (self._slots[Slot.MAX] - self._slots[Slot.MIN])
        if (
            self._type.is_lambda(self._slots[Slot.COLLECTIONS_SUM]) and
            not any(
                self._type.is_lambda(self._slots[s])
                    for s in self._type.collection_slots
            )
        ):
            self._slots[Slot.COLLECTIONS_SUM] = sum(
                self._slots[s] for s in self._type.collection_slots
            )
        if (
            self._type.is_lambda(self._slots[Slot.TOTAL]) and
            not any(
                self._type.is_lambda(self._slots[s])
                    for s in self._type.fillable_slots
            )
        ):
            self._slots[Slot.TOTAL] = \
                self._slots[Slot.NUMBERS_SUM] + \
                    self._slots[Slot.SUMS_DIFFERENCE] + \
                    self._slots[Slot.COLLECTIONS_SUM]

    def requires_post_filling_action (self):
        """Checks if a post-filling action is required for the column.

Returns
-------
See Notes for more details.

See Also
--- ----
requires_pre_filling_action : Method that checks if a pre-filling action is \
required
pre_filling_action : Method that defines the action to perform before \
filling the column
post_filling_action : Method that defines the action to perform after \
filling the column

Notes
-----
If a post-filling action is required after the column is filled (at the end of
the turn), this method should return:

* a falsey value (e. g. ``None`` or ``False``) if no action is required,
* ``True`` if an action is required, but no parameters should be passed to the
    `post_filling_method` method,
* a tuple of `tuple` (`args`) and `dict` (`kwargs`) defining the parameters
    for the `post_filling_action` method.

In the latter case, each argument should be represented by its intended type.
Please refer to the documentation for `requires_pre_filling_action` for more
details.  By default, this method returns ``None``.

A reuqirement of a post-filling action means that the action is required if
the player chose to fill this column at the end of the turn.  It does not mean
that the action must be done if the player did not fill the column.
"""
        return None

    def post_filling_action (self, *args, **kwargs):
        """Action to perform after filling the column, at the end of a turn.

Parameters
----------
args, kwargs
    Additional parameters to use when implementing the method.

See Also
--- ----
requires_pre_filling_action : Method that checks if a pre-filling action is \
required
pre_filling_action : Method that defines the action to perform before \
filling the column
requires_post_filling_action : Method that checks if a post-filling action \
is required

Notes
-----
This method is intended for columns that require an action to be performed
after filling the column.  A reuqirement of a post-filling action means that
the action is required if the player chose to fill this column at the end of
the turn.  It does not mean that the action must be done if the player did not
fill the column.

By default, this method does nothing.  Implement it for column-specific rules
if required (such as the announced column).
"""
        pass

    def is_full (self, fillable = False):
        """Checks if the column is full (completely filled).

Parameters
----------
fillable : boolean, default = False
    If ``True``, only the fillable slots are cheked; otherwise all slots are
    checked.

Returns
-------
bool
    ``True`` if the column is full, ``False`` otherwise.  If `fillable` is
    ``False``, the method may return ``True`` even if not all slots are
    filled, but all fillable slots are.

Raises
------
TypeError
    If `fillable` is not a boolean value.
"""
        if self._check_input and not isinstance(fillable, _AnyBoolean):
            raise TypeError("Fillable flag must be a boolean value.")

        return not any(
            self._type.is_lambda(self._slots[s]) \
                for s in (self._type.fillable_slots if fillable else Slot)
        )

    def to_numpy (self):
        """Returns the scores aranged into a `numpy.ndarray`.

This method returns a similar structure to the `scores` property, but it is
ensured that the result is a `numpy.ndarray` via the `numpy.asanyarray`
function.  If the object returned by the `scores` property is already a
`numpy.ndarray`, mutating the array returned by this method shall affect the
column's in-memory object.

Please refer to the documentation for `scores` property and `numpy.asanyarray`
function for more details.

Returns
-------
numpy.ndarray
    Scores aranged into a `numpy.ndarray`.

Raises
------
NotImplementedArray
    If NumPy back-end is unavailable (if `numpy` cannot be imported).

See Also
--- ----
scores : Similar property but does not ensure `numpy.ndarray`
__array__ : Similar method but without a dependency availability check
to_pandas : Similar method but returns a `pandas.Series`

Notes
-----
Instead of overriding this method, consider overriding the `__array__` method
instead.  The object returned by the `__array__` method shall be returned by
this method as well, but a cautionary dependency check shall be done first.
"""
        if _np is None:
            raise NotImplementedError("Missing optional dependency 'numpy'.")

        return self.__array__()

    def to_pandas (self, str_index = False):
        """Returns the scores aranged into a `pandas.Series`.

This method returns a similar structure to the `scores` property, but it is
ensured that the result is a `pandas.Series`.  If the object returned by the
`scores` property is already a `numpy.ndarray`, mutating the series returned
by this method shall affect the column's in-memory object.

Please refer to the documentation for `to_numpy` method and `numpy.asanyarray`
function for more details.

Parameters
----------
str_index : boolean, default = False
    If ``True``, the resulting `pandas.Series` shall use an index of integers
    (slot index values); otherwise an index of strings (slot names) is used.

Returns
-------
pandas.Series
    Scores aranged into a `pandas.Series`.

Raises
------
NotImplementedArray
    If NumPy and Panas back-ends are unavailable (if `numpy` and/or `pandas`
    cannot be imported).

See Also
--- ----
to_numpy : Similar method but returns a `numpy.ndarray`

Notes
-----
Instead of overriding this method, consider overriding the `__array__` method
instead.  The object returned by the `__array__` method shall be returned by
this method as well, but a cautionary dependency check shall be done first.
"""
        if _np is None or _pd is None:
            raise NotImplementedError(
                "Missing optional dependencies 'numpy' and/or 'pandas'."
            )

        if self._check_input and not isinstance(str_index, _AnyBoolean):
            raise TypeError("String index flag must be a boolean value.")

        return _pd.Series(
            self.__array__(),
            index = \
                self._type.slots_str_index if str_index \
                    else self._type.slots_index,
            name = self._name
        )

    def __repr__ (self):
        return str(self._type.display(self))

    def __array__ (self):
        """Returns the scores aranged into a `numpy.ndarray`.

Please refer to the documentation for `to_numpy` method for more details.

Returns
-------
numpy.ndarray
    Scores aranged into a `numpy.ndarray`.

See Also
--- ----
to_numpy : Similar method but with a dependency availability check

Notes
-----
Unlike the `to_numpy` method, this method assumes that NumPy back-end is
available (that `numpy` is successfully imported).  This is because the method
is not intended to be called directly in any library or user code (except for
the `to_numpy` method), but only implicitly via the `numpy.array` and similar
`numpy.ndarray` initialisers.  If NumPy is unavailable, the method's behaviour
is undefined and not guaranteed.
"""
        return _np.asanyarray(self._slots)

    def __len__ (self):
        return len(self._slots)

    def __iter__ (self):
        for s in Slot:
            yield s

    def __getitem__ (self, key):
        """Returnes the score of the slot.

Parameters
----------
slot : Slot or integer or string
    Slot to check.

Raises
------
TypeError
    If `slot` is not of a supported type.

Notes
-----
Additional errors might be raised (and are not caught by the method) when
converting an integer or a string to a `Slot` via ``Slot(slot)`` and
``Slot[slot.upper()]`` calls respectively.

If method parameters are not checked (recall the `_check_inputs` instance
variable), ``slot` may not be a string and `results` must be a sequence rather
than a single integer.
"""
        if self._check_input:
            key = self._type._ensure_slot(key)

        return self._slots[key]

    def __getattr__ (self, name):
        if isinstance(name, _AnyString) and name in Slot.__members__:
            return self[Slot[name]]

        return super(Column, self).__getattribute__(name)

    @property
    def name (self):
        """Name of the column."""
        return self._name

    @property
    def scores (self):
        """The list of scores over slots.

For a slot `s`, ``scores[s]`` is the score currently filled into the slot `s`
in this column.  The index `s` may be a `Slot` value or its corresponding
integral value (e. g. `Slot.ONE` is 1).

See Also
--- ----
lambda_score : Value that represents an undefined score
is_lambda_score : Method that checks if a score is undefined
get_lambda_slots : Method that returns slots with undefined scores from a \
list of scores
to_numpy : Similar function but ensures `numpy.ndarray`

Notes
-----
This property returns the underlying in-memory object for storing the state of
scores of the column.  At worst, manually mutating the returned list might
cause unexpected behaviour of the column, and, at the very least, is a method
of cheating when done during a game.  Instead, create a copy using
``copy.copy(scores)`` or ``copy.deepcopy(scores)`` before altering it.
"""
        return self._slots

    @property
    def check_input (self):
        """Flag indicating whether or not method parameters should be \
checked."""
        return self._check_input

    @property
    def type_ (self):
        """The type used by the column for calling class methods."""
        return self._type

class OrderedColumn (Column):
    """Represents an ordered column.

Parameters
----------
order : OrderedColumn.Order or integer or { "down", "mixed", "up" }
    If ``"down"``, the column is filled from top to bottom; if ``"up"``, it is
    filled from bottom to top.  A `"mixed"` column is filled in either way, but
    does not allow random access such as the free or the announced column.

name : string, optional
    Name of the column.  If not provided, the name is inferred from the
    column's type.

check_input : boolean, default = True
    If ``True``, arguments are checked in all methods before conducting any
    business logic; otherwise all arguments are assumed to be valid (by type
    and value) and legal (by game and column rules).
"""

    @_enum.unique
    class Order (
        _enum.IntFlag
            if (_sys.version_info.major, _sys.version_info.minor) >= (3, 6)
                else _enum.IntEnum
    ):
        """Represents the order of filling an ordered column.

This enumeration behaves like an integer flag, i. e. a `MIXED` order is actually
equal to ``DOWN | UP``.  Because of this, a special order of `NONE` is defined
whose value is equal to ``DOWN & UP``, which is 0.  However, avoid using this
order since such columns shall be impossible to fill (no slot will be
considered fillable in any next round).
"""
        NONE = 0
        DOWN = 1
        UP = 2
        MIXED = 3

    def __new__ (cls, *args, **kwargs):
        instance = super(OrderedColumn, cls).__new__(cls)

        instance._order = None

        return instance

    def __init__ (self, order = Order.DOWN, name = None, check_input = True):
        super(OrderedColumn, self).__init__(
            name = name,
            check_input = check_input
        )

        if isinstance(order, self._type.Order):
            self._order = order
        elif isinstance(order, _AnyInteger):
            self._order = self._type.Order(order)
        else:
            if not isinstance(order, _AnyString):
                raise TypeError(
                    "Order must be a string or an order value."
                )
            self._order = self._type.Order[order.upper()]

        if name is None:
            self._name = "{type_name}_{order}".format(
                type_name = getattr(self._type, '__name__', 'OrderedColumn'),
                order = self._order.name
            )

    def get_next_available_slots (self):
        if self._next_available_slots is None:
            available_slots = self.get_available_slots()
            end_slots = set()

            if len(available_slots):
                if self._order & self._type.Order.DOWN:
                    end_slots.add(available_slots[0])
                if self._order & self._type.Order.UP:
                    end_slots.add(available_slots[-1])

            self._next_available_slots = list(sorted(end_slots))

        return self._next_available_slots

    def is_slot_next_available (self, slot):
        if self._check_input:
            slot = self._type._ensure_slot(slot)
            if slot not in self._type.fillable_slots:
                raise KeyError(
                    "Slot {slot} is auto-filled.".format(slot = slot)
                )

        if self._available_slots is None:
            self.get_available_slots()

        if len(self._available_slots):
            if (
                (self._order & self._type.Order.DOWN) and
                slot == self._available_slots[0]
            ):
                return True
            if (
                (self._order & self._type.Order.UP) and
                slot == self._available_slots[-1]
            ):
                return True

        return False

    @property
    def order (self):
        """Order of filling the column."""
        return self._order

class FreeColumn (Column):
    """Represents a column that may be filled freely (in no specific \
order)."""

    def __new__ (cls, *args, **kwargs):
        instance = super(FreeColumn, cls).__new__(cls)

        return instance

    def __init__ (self, name = None, check_input = True):
        super(FreeColumn, self).__init__(
            name = name,
            check_input = check_input
        )

    def get_next_available_slots (self):
        return self.get_available_slots()

    def is_slot_next_available (self, slot):
        return self.is_slot_available(slot)

class AnnouncedColumn (Column):
    """Represents a column that requires an announcement before filling.

Parameters
----------
after_roll : integer, default = 1
    One-indexed roll index after which the announcement must be made.

immediately_fill : boolean, default = False
    If ``True``, next rolls after an announcement are disallowed and the
    column must be immediately filled.

name : string, optional
    Name of the column.  If not provided, the name is inferred from the
    column's type.

check_input : boolean, default = True
    If ``True``, arguments are checked in all methods before conducting any
    business logic; otherwise all arguments are assumed to be valid (by type
    and value) and legal (by game and column rules).
"""

    def __new__ (cls, *args, **kwargs):
        instance = super(AnnouncedColumn, cls).__new__(cls)

        instance._after_roll = None
        instance._immediately_fill = None
        instance._announcement = None

        return instance

    def __init__ (
        self,
        after_roll = 1,
        immediately_fill = False,
        name = None,
        check_input = True
    ):
        super(AnnouncedColumn, self).__init__(
            name = name,
            check_input = check_input
        )

        if not isinstance(after_roll, _AnyInteger):
            raise TypeError("After roll index be an integral value.")
        if after_roll < 0:
            raise ValueError(
                "After roll index must be greater than or equal to 0."
            )
        self._after_roll = int(after_roll)

        if not isinstance(immediately_fill, _AnyBoolean):
            raise TypeError("Immediately fill flag must be a boolean value.")
        self._immediately_fill = bool(immediately_fill)

        self._announcement = None

    def disallow_rolls (self):
        if self._check_input and not self._immediately_fill:
            raise RuntimeError("Cannot disallow rolls.")

        self._rollable = False

    def allow_rolls (self):
        if self._check_input and not self._immediately_fill:
            raise RuntimeError("Cannot allow rolls.")

        self._rollable = True

    def lock (self):
        self._locked = True

    def unlock (self):
        self._locked = False

    def get_next_available_slots (self):
        if self._next_available_slots is None:
            self._next_available_slots = \
                self.get_available_slots() if self._announcement is None \
                    else [ self._announcement ]

        return self._next_available_slots

    def is_slot_next_available (self, slot):
        if self._check_input:
            slot = self._type._ensure_slot(slot)
            if slot not in self._type.fillable_slots:
                raise KeyError(
                    "Slot {slot} is auto-filled.".format(slot = slot)
                )

        return \
            self.is_slot_available(slot) if self._announcement is None \
                else (slot == self._announcement)

    def requires_pre_filling_action (self, roll):
        if self._check_input:
            roll = self._type._ensure_roll_index(roll)

        if roll == self._after_roll:
            return ((), { 'announcement': Slot })

        return None

    def announce (self, slot):
        """Announce a slot that will be filled in this column.

Calling this method also locks the column.  Also, if the column must be
immediately filled after the announcement, next rolls are disallowed.

Parameters
----------
slot : Slot or integer or string
    Slot to fill.

Raises
------
TypeError
    If `slot` is not of a supported type.

KeyError
    If `slot` is not fillable.

RuntimeError
    If a slot is already announced.

See Also
--- ----
supress : Method that suppresses an announcement
lock : Method that locks the column
unlock : Method that unlocks the column
is_locked : Method that checks if the column is locked
disallow_rols : Method that disallows next rolls
allow_rols : Method that allows next rolls
can_roll : Method that checks if next rolls are allowed

Notes
-----
Additional errors might be raised (and are not caught by the method) when
converting an integer or a string to a `Slot` via ``Slot(slot)`` and
``Slot[slot.upper()]`` calls respectively.

If method parameters are not checked (recall the `_check_inputs` instance
variable), ``slot` may not be a string and `results` must be a sequence rather
than a single integer.
"""
        if self._announcement is not None:
            raise RuntimeError("A slot is already announced.")

        if self._check_input:
            slot = self._type._ensure_slot(slot)
            if slot not in self._type.fillable_slots:
                raise KeyError(
                    "Slot {slot} is auto-filled.".format(slot = slot)
                )
            if not self._type.is_lambda(self._slots[slot]):
                raise RuntimeError(
                    "Slot {slot} is already filled.".format(slot = slot)
                )

        self._announcement = slot

        self._next_available_slots = None
        self._available_slots = None

        self.lock()
        if self._immediately_fill:
            self.disallow_rolls()

    def pre_filling_action (self, roll, *args, **kwargs):
        if roll == 1:
            self.announce(
                args[0] if (len(args) == 1 and not kwargs) \
                    else kwargs['announcement']
            )

    def requires_post_filling_action (self):
        return True

    def suppress (self):
        """Suppresses the previous announcement.

This method also unlocks the column.  Also, if the column must be immediately
filled after an announcement, next rolls are disallowed.

Raises
------
RuntimeError
    If no slot is announced yet.

See Also
--- ----
announce : Method that announces filling a column
lock : Method that locks the column
unlock : Method that unlocks the column
is_locked : Method that checks if the column is locked
disallow_rols : Method that disallows next rolls
allow_rols : Method that allows next rolls
can_roll : Method that checks if next rolls are allowed
"""
        if self._announcement is None:
            raise RuntimeError("No slot is announced yet.")

        self._announcement = None

        self._available_slots = None
        self._next_available_slots = None

        self.unlock()
        if self._immediately_fill:
            self.allow_rolls()

    def post_filling_action (self, *args, **kwargs):
        self.suppress()

    def is_announced (self):
        """Checks if a slot is announced.

Returns
-------
bool
    ``True`` if a slot is announced, ``False`` otherwise.
"""
        return self._announcement is not None

    @property
    def after_roll (self):
        """One-indexed roll index after which the announcement must be \
made."""
        return self._after_roll

    @property
    def immediately_fill (self):
        """Flag indicating if the column must be immediately filled after \
the announcement."""
        return self._immediately_fill

    @property
    def announcement (self):
        """The currently announced slot or ``None`` if no slot is \
announced."""
        return self._announcement

class Yamb (object):
    """Represents a yamb game instance.

The class provides methods for managing a yamb game, such as rolling the dice
and filling columns/* all the while making sure the rules are followed */.

Parameters
----------
random_state : Die or callable or number or random.Random or \
numpy.random.RandomState or numpy.random.Generator or \
numpy.random.BitGenerator or module[random] or module[numpy.random], optional
    Random state of the dice (for rolling results).  If a `Die` instance, it
    is used as the die for the game; otherwise please refer to the
    documentation for `Die` for more details.
"""

    @classmethod
    def _ensure_replacements (cls, n_dice, replace):
        if replace is None:
            return None

        if (
            not isinstance(replace, _AnyIterable) or
            isinstance(replace, _AnyString)
        ):
            replace = (replace, )
        if not isinstance(replace, _AnySequence):
            replace = tuple(replace)
        if len(replace) != n_dice:
            raise ValueError(
                "Replacements must be of length {n_dice:d}".format(
                    n_dice = n_dice
                )
            )
        if _np is not None and isinstance(replace, _np.ndarray):
            if replace.ndim != 1:
                raise ValueError(
                    "Replacements must be a 1-dimensional sequence."
                )
            if not _np.issubdtype(replace.dtype, _np.bool_):
                raise TypeError("Replacements must be boolean values.")
        else:
            for r in replace:
                if not isinstance(r, _AnyBoolean):
                    raise TypeError("Replacements must be boolean values.")

        return replace

    @classmethod
    def _ensure_column (cls, n_columns, column):
        if not isinstance(column, _AnyInteger):
            raise TypeError("Column index must be an integral value.")
        if column < 0:
            column += n_columns
        if not (0 <= column < n_columns):
            raise ValueError(
                "Column index is out of range [{zero:d}, " \
                    "{n_columns:d}).".format(
                        zero = 0,
                        n_columns = n_columns
                    )
            )

        return int(column)

    @classmethod
    def _replace_results (
        cls,
        old_results,
        new_results,
        replace,
        sort = False
    ):
        if replace is None:
            return list(sorted(new_results))

        j = 0
        for i, r in enumerate(replace):
            if r:
                old_results[i] = new_results[j]
                j += 1
        if sort:
            old_results.sort()

        return old_results

    @classmethod
    def _new_empty_results (cls, n):
        if isinstance(n, _AnyIterable):
            if isinstance(n, _AnyCollection):
                return list(0 for _ in _range(len(n)))
            else:
                return list(0 for _ in n)

        return list(0 for _ in _range(n))

    def __new__ (cls, *args, **kwargs):
        instance = super(Yamb, cls).__new__(cls)

        instance._type = None
        instance._check_input = None
        instance._sort_results = None
        instance._n_dice = None
        instance._n_rolls = None
        instance._dice = None
        instance._columns = None
        instance._roll_index = None
        instance._results = None

        return instance

    def __init__ (
        self,
        columns = None,
        n_dice = 5,
        n_rolls = 3,
        sort_results = False,
        check_input = True,
        random_state = None
    ):
        super(Yamb, self).__init__()

        self._type = self.__class__

        if not isinstance(check_input, _AnyBoolean):
            raise TypeError("Check input flag must be a boolean value.")
        self._check_input = bool(check_input)

        if not isinstance(sort_results, _AnyBoolean):
            raise TypeError("Sort results flag must be a boolean value.")
        self._sort_results = bool(sort_results)

        if not isinstance(n_dice, _AnyInteger):
            raise TypeError("Number of dice must be an integral value.")
        if n_dice <= 0:
            raise ValueError("Number of dice must be greater than 0.")
        self._n_dice = int(n_dice)

        if not isinstance(n_rolls, _AnyInteger):
            raise TypeError("Number of rolls must be an integral value.")
        if n_rolls <= 0:
            raise ValueError("Number of rolls must be greater than 0.")
        self._n_rolls = int(n_rolls)

        self._dice = \
            random_state if isinstance(random_state, Die) \
                else Die(random_state)

        if columns is None:
            self._columns = [
                OrderedColumn(OrderedColumn.Order.DOWN),
                OrderedColumn(OrderedColumn.Order.UP),
                FreeColumn(),
                AnnouncedColumn()
            ]
        elif isinstance(columns, Column):
            self._columns = [ columns ]
        else:
            self._columns = list(columns)
            for c in self._columns:
                if not isinstance(c, Column):
                    raise TypeError(
                        "Columns must be of type `Column`, column " \
                            "`{column}` not understood.".format(
                                column = c
                            )
                    )
        if not self._columns:
            raise TypeError(
                "Yamb game cannot be empty (at least 1 column is required)."
            )

        self._roll_index = 0
        self._results = self._type._new_empty_results(self._n_dice)

    def which_column_is_locked (self):
        return self._locked

    def can_roll (self):
        return \
            False if self._roll_index >= self._n_rolls \
                else (
                    self._locked is None or
                    self._columns[self._locked].can_roll()
                )

    def start_turn (self):
        self._locked = None

        self._roll_index = 0
        self._results = self._type._new_empty_results(self._results)

    def get_pre_filling_requirements (self, column):
        if self._check_input:
            column = self._type._ensure_column(len(self._columns), column)

        return self._columns[column].requires_pre_filling_action(
            self._roll_index
        )

    def get_all_pre_filling_requirements (self):
         return tuple(
             c.requires_pre_filling_action(self._roll_index)
                for c in self._columns
        )

    def make_pre_filling_action (self, column, *args, **kwargs):
        if self._check_input:
            column = self._type._ensure_column(len(self._columns), column)
            if not (self._locked is None or column == self._locked):
                raise RuntimeError(
                    "Cannot make action on a non-locked column if another " \
                        "column is locked."
                )

        self._columns[column].pre_filling_action(
            self._roll_index,
            *args,
            **kwargs
        )

        if self._columns[column].is_locked():
            self._locked = column

    def roll_dice (self, replace = None):
        if self._check_input:
            if not (self._roll_index or replace is None):
                raise TypeError(
                    "Replacements are not allowed in the first roll."
                )
            if not self.can_roll():
                raise RuntimeError("Cannot roll dice, proceed to filling.")
            replace = self._type._ensure_replacements(self._n_dice, replace)

        results = self._dice.roll(
            self._n_dice if replace is None else sum(replace)
        )

        if not (self._roll_index and replace is None):
            self._results = self._type._replace_results(
                self._results,
                results,
                replace,
                self._sort_results
            )

        self._roll_index += 1

    def get_post_filling_requirements (self, column):
        if self._check_input:
            if self._roll_index:
                raise RuntimeError(
                    "Cannot check post-filling requirements before the end " \
                        "of the turn."
                )
            if not (self._locked is None or column == self._locked):
                raise RuntimeError(
                    "Cannot make action on a non-locked column if another " \
                        "column is locked."
                )

        return self._columns[column].requires_post_filling_action()

    def make_post_filling_action (self, column, *args, **kwargs):
        if self._check_input:
            if self._roll_index:
                raise RuntimeError(
                    "Cannot make post-filling action before the end of the " \
                        "turn."
                )
            if not (self._locked is None or column == self._locked):
                raise RuntimeError(
                    "Cannot make action on a non-locked column if another " \
                        "column is locked."
                )

        self._columns[column].post_filling_action(*args, **kwargs)

        self._locked = None

    def end_turn (self, column, slot):
        if self._check_input:
            if not self._roll_index:
                raise RuntimeError(
                    "Cannot end turn before rolling the dice at least once."
                )
            if not (self._locked is None or column == self._locked):
                raise RuntimeError(
                    "Cannot fill a non-locked column if another column is " \
                        "locked."
                )

        self._columns[column].fill_slot(slot, self._results)

        self._roll_index = 0

        requirements = self.get_post_filling_requirements(column)

        self._locked = column if requirements else None

    def update_auto_slots (self):
        for i in _range(len(self._columns)):
            self._columns[i].update_auto_slots()

    def is_full (self, fillable = False):
        return all(c.is_full(fillable) for c in self._columns)

    def get_total_score (self, slot = Slot.TOTAL):
        return sum(
            c[slot]
                for c in self._columns if not c.type_.is_lambda(c[slot])
        )

    def to_numpy (self):
        if _np is None:
            raise NotImplementedError("Missing optional dependency 'numpy'.")

        return self.__array__()

    def to_pandas (self, str_index = False):
        if _np is None or _pd is None:
            raise NotImplementedError(
                "Missing optional dependencies 'numpy' and/or 'pandas'."
            )

        return _pd.concat(
            tuple(c.to_pandas(str_index) for c in self._columns),
            axis = 1
        )

    def __repr__ (self):
        return str(Column.display(self._columns))

    def __array__ (self):
        return _np.column_stack(self._columns)

    def __len__ (self):
        return len(self._columns)

    def __getitem__ (self, key):
        if self._check_input:
            key = self._type._ensure_column(key)

        return self._columns[key]

    @property
    def n_dice (self):
        return self._n_dice

    @property
    def n_rolls (self):
        return self._n_rolls

    @property
    def sort_results (self):
        return self._sort_results

    @property
    def dice (self):
        return self._dice

    @property
    def columns (self):
        return self._columns

    @property
    def roll_index (self):
        return self._roll_index

    @property
    def results (self):
        return self._results

    @property
    def check_input (self):
        return self._check_input

    @property
    def type_ (self):
        return self._type

class Player (object if _sys.version_info.major < 3 else _abc.ABC):
    if _sys.version_info.major < 3:
        __metaclass__ = _abc.ABCMeta

    def __new__ (cls, *args, **kwargs):
        instance = super(Player, cls).__new__(cls)

        instance._type = None
        instance._check_input = None
        instance._name = None
        instance._update_auto_slots = None

        return instance

    def __init__ (
        self,
        name = None,
        update_auto_slots = False,
        check_input = True
    ):
        super(Player, self).__init__()

        self._type = self.__class__

        if not isinstance(check_input, _AnyBoolean):
            raise TypeError("Check input flag must be a boolean value.")
        self._check_input = bool(check_input)

        if not (name is None or isinstance(name, _AnyString)):
            raise TypeError("Column name must be a string value.")
        self._name = str(
            getattr(self._type, '__name__', 'Column') if name is None
                else name
        )

        if update_auto_slots is None:
            self._update_auto_slots = 0
        elif isinstance(update_auto_slots, _AnyBoolean):
            self._update_auto_slots = int(update_auto_slots)
        elif isinstance(update_auto_slots, _AnyInteger):
            if update_auto_slots < 0:
                raise ValueError(
                    "Update auto slots checkpoint must be grater than or " \
                        "equal to 0."
                )
            self._update_auto_slots = int(update_auto_slots)
        else:
            raise TypeError(
                "Update auto slots checkpoint must be a boolean or an " \
                    "integral value."
            )

    @_abc.abstractmethod
    def observe_roll_results (
        self,
        columns,
        locked_column_index,
        roll,
        results
    ):
        pass

    @_abc.abstractmethod
    def choose_pre_filling_action_column (
        self,
        columns,
        roll,
        results,
        requirements
    ):
        pass

    @_abc.abstractmethod
    def set_pre_filling_requirements (
        self,
        columns,
        column_index,
        roll,
        results,
        requirements
    ):
        pass

    @_abc.abstractmethod
    def choose_replacements (
        self,
        columns,
        locked_column_index,
        roll,
        results
    ):
        pass

    @_abc.abstractmethod
    def choose_column_to_fill (self, columns, roll, results):
        pass

    @_abc.abstractmethod
    def choose_slot_to_fill (self, columns, column_index, roll, results):
        pass

    @_abc.abstractmethod
    def observe_turn_end (self, columns, column_index, slot):
        pass

    @_abc.abstractmethod
    def set_post_filling_requirements (
        self,
        columns,
        column_index,
        slot,
        requirements
    ):
        pass

    def play (self, game = None):
        if game is None:
            game = Yamb()
        elif self._check_input and not isinstance(game, Yamb):
            raise TypeError(
                "Game must be of type `Yamb`, game {game} is not " \
                    "understood.".format(
                        game = game
                    )
            )

        i = 0

        while not game.is_full(True):
            game.start_turn()

            i += 1

            while True:
                self.observe_roll_results(
                    game.columns,
                    game.which_column_is_locked(),
                    game.roll_index,
                    game.results
                )

                if game.which_column_is_locked() is None:
                    requirements = game.get_all_pre_filling_requirements()
                    if any(requirements):
                        column = self.choose_pre_filling_action_column(
                            game.columns,
                            game.roll_index,
                            game.results,
                            requirements
                        )
                        if column is not None:
                            requirements = requirements[column]
                            if requirements:
                                if isinstance(requirements, _AnyBoolean):
                                    game.make_pre_filling_action(column)
                                else:
                                    args, kwargs = \
                                        self.set_pre_filling_requirements(
                                            game.columns,
                                            column,
                                            game.roll_index,
                                            game.results,
                                            requirements
                                        )
                                    game.make_pre_filling_action(
                                        column,
                                        *args,
                                        **kwargs
                                    )

                if game.can_roll():
                    if game.roll_index:
                        replace = self.choose_replacements(
                            game.columns,
                            game.which_column_is_locked(),
                            game.roll_index,
                            game.results
                        )
                        if replace is None:
                            break
                        else:
                            game.roll_dice(replace)
                    else:
                        game.roll_dice()
                else:
                    break

            column = game.which_column_is_locked()
            if column is None:
                column = self.choose_column_to_fill(
                    game.columns,
                    game.roll_index,
                    game.results
                )
            slot = self.choose_slot_to_fill(
                game.columns,
                column,
                game.roll_index,
                game.results
            )
            game.end_turn(column, slot)

            requirements = game.get_post_filling_requirements(column)
            if requirements:
                if isinstance(requirements, _AnyBoolean):
                    game.make_post_filling_action(column)
                else:
                    args, kwargs = self.set_post_filling_requirements(
                        game.columns,
                        column,
                        game.roll_index,
                        slot,
                        requirements
                    )
                    game.make_post_filling_action(column, *args, **kwargs)

            if self._update_auto_slots and not i % self._update_auto_slots:
                game.update_auto_slots()

            self.observe_turn_end(game.columns, column, slot)

        return game

    @property
    def name (self):
        return self._name

    @property
    def update_auto_slots (self):
        return self._update_auto_slots

    @property
    def check_input (self):
        return self._check_input

    @property
    def type_ (self):
        return self._type

class ObservingPlayer (Player):
    def __new__ (cls, *args, **kwargs):
        instance = super(ObservingPlayer, cls).__new__(cls)

        instance._player = None
        instance._output = None
        instance._line_terminator = None

        return instance

    def __init__ (
        self,
        player,
        name = None,
        output = _sys.stdout,
        line_terminator = _os.linesep
    ):
        super(ObservingPlayer, self).__init__(
            name = name,
            update_auto_slots = False,
            check_input = False
        )

        if not isinstance(player, Player):
            raise TypeError(
                "Player must be of type `Player`, player {player} is not " \
                    "understood.".format(
                        player = player
                    )
            )
        self._player = player

        if output is None:
            self._output = _sys.stdout
        else:
            self._output = output

        if not (name is None or isinstance(name, _AnyString)):
            raise TypeError("Line terminator must be a string value.")
        self._line_terminator = str(
            _os.linesep if line_terminator is None else line_terminator
        )

    def observe_roll_results (
        self,
        columns,
        locked_column_index,
        roll,
        results
    ):
        pass

    @property
    def player (self):
        return self._player

    @property
    def output (self):
        return self._output
