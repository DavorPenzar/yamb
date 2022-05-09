# -*- coding: utf-8 -*-

import collections as _collections
import functools as _functools
import itertools as _itertools
import numbers as _numbers

import numpy as _np
import scipy as _sp
import scipy.special as _sp_special

import yamb as _engine

_nan = float('nan') # math.nan, numpy.nan
_posinf = float('inf') # math.inf, numpy.inf
_neginf = -_posinf # numpy.NINF

_max_cache_size = 0x20

def identity (x):
    return x

def relu_complete (
    x,
    max_value = _posinf,
    negative_slope = 0,
    threshold = 0
):
    x, max_value, negative_slope, threshold = _np.broadcast_arrays(
        x,
        max_value,
        negative_slope,
        threshold,
        subok = True
    )

    idx_negative = (x < threshold)
    idx_max = (x >= max_value)
    idx_x = ~(idx_negative | idx_max)

    y = _np.zeros(
        x.shape,
        dtype = _np.find_common_type(
            [ x.dtype ],
            [ max_value.dtype, negative_slope.dtype, threshold.dtype ]
        )
    )

    _np.subtract(
        x,
        threshold,
        where = idx_negative,
        out = y
    )
    _np.multiply(
        negative_slope,
        y,
        where = idx_negative,
        out = y
    )
    _np.copyto(y, max_value, where = idx_max)
    _np.copyto(y, x, where = idx_x)

    return y

def relu (x):
    return _np.maximum(x, 0)

def sigmoid (x):
    return _sp_special.expit(x)

class NeuralPlayer (_engine.Player):
    expected_scores = _np.array(
        [
            '28.1840277777777778',
             '0.8333333333333333',
             '1.6666666666666667',
             '2.5',
             '3.3333333333333333',
             '4.1666666666666667',
             '5',
            '17.5',
            '17.5',
            '17.5',
             '0',
             '6.4814814814814815',
             '1.2345679012345679',
             '1.8325617283950617',
             '1.0833333333333333',
             '0.0520833333333333',
            '10.6840277777777778'
        ],
        dtype = _np.float32
    )

    def _ensure_numerical_array (a, ndim = None):
        if ndim is None:
            a = _np.asarray(a)
        else:
            if ndim == 1:
                a = _np.atleast_1d(a)
            elif ndim == 2:
                a = _np.atleast_2d(a)
            elif ndim == 3:
                a = _np.atleast_3d(a)
            else:
                a = _np.array(a, copy = False, subok = False, ndmin = ndim)

        if not _np.issubdtype(a.dtype, _np.number):
            raise TypeError(
                f"Expected numerical array, got {a.dtype} instead."
            )
        if ndim is not None and a.ndim != ndim:
            raise ValueError(
                f"Expected {ndim} dimensions, got {a.ndim} instead."
            )

        return _np.ascontiguousarray(a)

    @classmethod
    def _get_column (cls, columns, column_index, my_column):
        return column_index if my_column is None else my_column

    @classmethod
    def _get_slot (cls, columns, column_index, my_column, my_slot):
        return \
            columns[
                cls._get_column(columns, column_index, my_column)
            ].get_next_available_slots()[0] if my_slot is None else my_slot

    @classmethod
    def _apply_affine_operator (cls, A, b, x):
        return _np.dot(A, x) + b

    @classmethod
    def _transform (cls, layers, x):
        for A, b, f in layers:
            x = f(cls._apply_affine_operator(A, b, x))

        return x

    @classmethod
    def _build_layers (
        cls,
        layers,
        out_function = identity,
        input_size = None,
        output_size = None
    ):
        if (
            not isinstance(layers, (_collections.Iterable, _np.ndarray)) or
            isinstance(layers, (str, _np.str_))
        ):
            layers = (layers, )
        if out_function is None:
            out_function = identity
        if not (
            callable(out_function) or
            isinstance(out_function, _collections.Callable)
        ):
            raise TypeError("Output activation function must be callable.")

        net = list()

        l1, l2 = _itertools.tee(layers, 2)
        try:
            next(l2)
        except StopIteration:
            pass

        hidden = True
        while hidden:
            A, b = next(l1)
            try:
                next(l2)
            except StopIteration:
                hidden = False

            A = cls._ensure_numerical_array(A, 2)
            b = cls._ensure_numerical_array(b, 1)

            if input_size is not None and A.shape[1] != input_size:
                raise ValueError(
                    f"Expected input size {input_size}, got {A.shape[1]} " \
                        "instead."
                )
            if b.size != A.shape[0]:
                raise ValueError(
                    f"Linear operator and bias size mismatch: {A.shape} " \
                        f"vs. {b.shape}."
                )

            net.append((A, b, relu if hidden else out_function))

            input_size = A.shape[0]

        if not (net or input_size is None or output_size is None):
            raise ValueError("Empty net is not expected.")

        if output_size is not None and net[-1][0].shape[0] != output_size:
            raise ValueError(
                f"Expected output size {output_size}, got " \
                    f"{net[-1][0].shape[0]} instead."
            )

        return net

    @classmethod
    @_functools.lru_cache(maxsize = _max_cache_size)
    def _get_hashable_column_representation (
        cls,
        scores,
        next_available_results
    ):
        """(2, n_fillable_slots)"""
        column = _np.array(
            [
                _np.asanyarray(scores)[
                    _engine.Column.fillable_slots_array
                ],
                _np.isin(
                    _engine.Column.fillable_slots_array,
                    next_available_results,
                    assume_unique = True
                )
            ],
            dtype = _np.float32
        )
        _np.copyto(
            column[0],
            cls.expected_scores[_engine.Column.fillable_slots_array],
            where = _np.isnan(column[0])
        )

        return column

    @classmethod
    def _get_column_representation (cls, column):
        """(2, n_fillable_slots)"""
        return cls._get_hashable_column_representation(
            tuple(column.scores),
            tuple(column.get_next_available_slots())
        )

    @classmethod
    def _get_columns_representation (
        cls,
        columns,
        roll,
        announced_columns = None
    ):
        """(n_columns, 2, n_fillable_slots)"""
        if roll is None:
            roll = _posinf

        columns_representation = _np.array(
            list(cls._get_column_representation(c) for c in columns)
        )

        if announced_columns is not None:
            for i in announced_columns:
                if roll > columns[i].after_roll:
                    columns_representation[i, 1, :] = 0

        return columns_representation

    @classmethod
    @_functools.lru_cache(maxsize = _max_cache_size)
    def _get_slot_representation (cls, slot):
        return _np.isin(
            _engine.Column.fillable_slots_array,
            slot,
            assume_unique = True
        )

    @classmethod
    @_functools.lru_cache(maxsize = _max_cache_size)
    def _get_roll_representation (cls, roll):
        """(1,)"""
        return _np.array([ roll ])

    @classmethod
    @_functools.lru_cache(maxsize = _max_cache_size)
    def _get_hashable_results_representation (cls, results):
        """(6,)"""
        bow_results = _np.zeros(len(_engine.Die.sides))
        for r, c in _collections.Counter(results).items():
            bow_results[r - 1] = c

        return bow_results

    @classmethod
    def _get_results_representation (cls, results):
        """(6,)"""
        return cls._get_hashable_results_representation(
            results if isinstance (results, _collections.abc.Hashable)
                else tuple(results)
        )

    @classmethod
    def _get_column_slot_y (
        cls,
        columns,
        layers,
        x,
        announced_columns = None,
        allow_announced_column = None,
        roll = None
    ):
        if announced_columns is None:
            announced_columns = frozenset()
        if allow_announced_column is None:
            allow_announced_column = _nan

        y = cls._transform(
            layers,
            x
        ).reshape((len(columns), len(_engine.Column.fillable_slots_array)))
        y = _np.ascontiguousarray(y)
        for i, c in enumerate(columns):
            if (
                i != allow_announced_column and
                i in announced_columns and
                (roll is None or roll > c.after_roll)
            ):
                y[i].fill(_neginf)
            else:
                y[
                    i,
                    _np.isin(
                        _engine.Column.fillable_slots_array,
                        c.get_next_available_slots(),
                        assume_unique = True,
                        invert = True
                    )
                ] = _neginf

        column, slot = _np.unravel_index(
            _np.argmax(y),
            shape = y.shape,
            order = 'C'
        )
        slot = _engine.Column.fillable_slots_array[slot]

        return (column, slot)

    @classmethod
    def _get_replace_y (cls, results, layers, x):
        y = _np.around(cls._transform(layers, x))

        replace = _np.ones(len(results), dtype = _np.bool_)
        for i, r in enumerate(results):
            r -= 1
            if y[r] >= 0.5:
                replace[i] = False
                y[r] -= 1

        return replace if _np.any(replace) else None

    @classmethod
    def calculate_column_slot_units (cls, n_columns = 4, n_dice = 5):
        return (
            n_columns * 2 * len(_engine.Column.fillable_slots_array) +
                1 +
                len(_engine.Die.sides),
            n_columns * len(_engine.Column.fillable_slots_array)
        )

    @classmethod
    def calculate_unlocked_replace_units (cls, n_columns = 4, n_dice = 5):
        return (
            n_columns * 2 * len(_engine.Column.fillable_slots_array) +
                1 +
                len(_engine.Die.sides),
            len(_engine.Die.sides)
        )

    @classmethod
    def calculate_locked_replace_units (cls, n_columns = 4, n_dice = 5):
        return (
            len(_engine.Column.fillable_slots_array) +
                1 +
                len(_engine.Die.sides),
            len(_engine.Die.sides)
        )

    def __new__ (cls, *args, **kwargs):
        instance = super(NeuralPlayer, cls).__new__(cls)

        instance._n_columns = None
        instance._n_dice = None
        instance._announced_columns = None
        instance._column_slot_layers = None
        instance._unlocked_replace_layers = None
        instance._locked_replace_layers = None
        instance._column = None
        instance._slot = None

        return instance

    def __init__ (
        self,
        column_slot_layers,
        unlocked_replace_layers,
        locked_replace_layers,
        announced_columns = 3,
        n_columns = 4,
        n_dice = 5,
        name = None,
        update_auto_slots = False,
        check_input = True
    ):
        super(NeuralPlayer, self).__init__(
            name,
            update_auto_slots,
            check_input
        )

        if not isinstance(n_columns, (_numbers.Integral, _np.integer)):
            raise TypeError("Number of columns must be an integral value.")
        if n_columns < 0:
            raise ValueError(
                "Number of columns must be greater than or equal to 0."
            )
        self._n_columns = int(n_columns)

        if not isinstance(n_dice, (_numbers.Integral, _np.integer)):
            raise TypeError("Number of dice must be an integral value.")
        if n_dice < 0:
            raise ValueError(
                "Number of dice must be greater than or equal to 0."
            )
        self._n_dice = int(n_dice)

        if announced_columns is None:
            self._announced_columns = None
        else:
            if (
                not isinstance(
                    announced_columns,
                    (_collections.abc.Iterable, _np.ndarray)
                ) or
                isinstance(announced_columns, (str, _np.str_))
            ):
                announced_columns = (announced_columns, )
            if not isinstance(
                announced_columns,
                (_collections.abc.Sequence, _np.ndarray)
            ):
                announced_columns = list(announced_columns)
            if (
                len(announced_columns) or
                hasattr(announced_columns, 'dtype') or
                hasattr(announced_columns, 'dtypes')
            ):
                announced_columns = _np.atleast_1d(announced_columns)
            else:
                announced_columns = _np.array([], dtype = _np.int32, copy = True)
            if not _np.issubdtype(announced_columns.dtype, _np.integer):
                raise TypeError("Announced columns must be integral values.")
            if announced_columns.ndim != 1:
                raise ValueError(
                    "Announced columns must be a one-dimensional array."
                )
            if _np.any(
                (announced_columns < 0) |
                (announced_columns >= self._n_columns)
            ):
                raise ValueError(
                    f"Announced columns must be in range [{0}, " \
                        f"{self._n_columns})."
                )
            self._announced_columns = \
                _np.unique(announced_columns) if len(announced_columns) \
                    else None

        self._column_slot_layers = self._type._build_layers(
            column_slot_layers,
            sigmoid,
            *self._type.calculate_column_slot_units(
                self._n_columns,
                self._n_dice
            )
        )
        self._unlocked_replace_layers = self._type._build_layers(
            unlocked_replace_layers,
            relu,
            *self._type.calculate_unlocked_replace_units(
                self._n_columns,
                self._n_dice
            )
        )
        self._locked_replace_layers = self._type._build_layers(
            locked_replace_layers,
            relu,
            *self._type.calculate_locked_replace_units(
                self._n_columns,
                self._n_dice
            )
        )

        self._column = None
        self._slot = None

    def observe_roll_results (
        self,
        columns,
        locked_column_index,
        roll,
        results
    ):
        pass

    def choose_pre_filling_action_column (
        self,
        columns,
        roll,
        results,
        requirements
    ):
        if self._announced_columns is None:
            return None

        x = _np.concatenate(
            (
                self._type._get_columns_representation(
                    columns,
                    roll,
                    self._announced_columns
                ).ravel(),
                self._type._get_roll_representation(roll).ravel(),
                self._type._get_results_representation(
                    _np.sort(results)
                ).ravel()
            )
        )

        column, slot = self._type._get_column_slot_y(
            columns,
            self._column_slot_layers,
            x,
            self._announced_columns,
            self._column,
            roll
        )

        if column not in self._announced_columns:
            return None

        self._column = column
        self._slot = slot

        return self._column

    def set_pre_filling_requirements (
        self,
        columns,
        column_index,
        roll,
        results,
        requirements
    ):
        return (
            (),
            {
                'announcement': \
                    self._type._get_slot(
                        columns,
                        column_index,
                        self._column,
                        self._slot
                    )
            }
        )

    def choose_replacements (
        self,
        columns,
        locked_column_index,
        roll,
        results
    ):
        x = None
        layers = None

        if locked_column_index is None:
            x = _np.concatenate(
                (
                    self._type._get_columns_representation(
                        columns,
                        roll,
                        self._announced_columns
                    ).ravel(),
                    self._type._get_roll_representation(roll).ravel(),
                    self._type._get_results_representation(
                        _np.sort(results)
                    ).ravel()
                )
            )
            layers = self._unlocked_replace_layers
        else:
            self._slot = self._type._get_slot(
                columns,
                locked_column_index,
                self._column,
                self._slot
            )
            x = _np.concatenate(
                (
                    self._type._get_slot_representation(self._slot).ravel(),
                    self._type._get_roll_representation(roll).ravel(),
                    self._type._get_results_representation(
                        _np.sort(results)
                    ).ravel()
                )
            )
            layers = self._locked_replace_layers

        replace = self._type._get_replace_y(results, layers, x)

        return replace

    def choose_column_to_fill (self, columns, roll, results):
        x = _np.concatenate(
            (
                self._type._get_columns_representation(
                    columns,
                    None,
                    self._announced_columns
                ).ravel(),
                self._type._get_roll_representation(roll).ravel(),
                self._type._get_results_representation(
                    _np.sort(results)
                ).ravel()
            )
        )

        self._column, self._slot = self._type._get_column_slot_y(
            columns,
            self._column_slot_layers,
            x,
            self._announced_columns,
            self._column,
            None
        )

        return self._column

    def choose_slot_to_fill (self, columns, column_index, roll, results):
        return self._type._get_slot(
            columns,
            column_index,
            self._column,
            self._slot
        )

    def set_post_filling_requirements (
        self,
        columns,
        column_index,
        slot,
        requirements
    ):
        return None

    def observe_turn_end (self, columns):
        self._column = 0
        self._slot = 0

    @property
    def n_columns (self):
        return self._n_columns

    @property
    def n_dice (self):
        return self._n_dice

    @property
    def announced_columns (self):
        return self._announced_columns

    @property
    def column_slot_layers (self):
        return self._column_slot_layers

    @property
    def unlocked_replace_layers (self):
        return self._unlocked_replace_layers

    @property
    def locked_replace_layers (self):
        return self._locked_replace_layers
