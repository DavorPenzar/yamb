# -*- coding: utf-8 -*-

import collections as _collections
import functools as _functools
import math as _math
import numbers as _numbers

import numpy as _np
import scipy as _sp
import scipy.special as _sp_special

import yamb as _engine

_nan = float('nan') # math.nan, numpy.nan
_posinf = float('inf') # math.inf, numpy.inf
_neginf = -_posinf # numpy.NINF

_max_cache_size = 0x20

def linear (x):
    return x

def relu_complete (
    x,
    max_value = _posinf,
    negative_slope = 0,
    threshold = 0
):
    return _np.where(
        x > max_value,
        max_value,
        _np.where(
            x < threshold,
            negative_slope * (x - threshold),
            x
        )
    )

def relu (x):
    return _np.maximum(x, 0)

class NeuralPlayer (_engine.Player):
    expected_results = _np.array(
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

    @classmethod
    def _apply_affine_operator (cls, A, b, x):
        return _np.dot(A, x) + b

    @classmethod
    def _transform (cls, layers, x):
        for A, b, f in layers:
            x = f(cls._apply_affine_operator(A, b, x))

        return x

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
            cls.expected_results[_engine.Column.fillable_slots_array],
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
    def _get_columns_representation (cls, columns, roll):
        """(n_columns, 2, n_fillable_slots, n_fillable_slots)"""
        columns_representation = _np.array(
            list(cls._get_column_representation(c) for c in columns)
        )

        for i, c in enumerate(columns):
            if isinstance(c, _engine.AnnouncedColumn) and c.after_roll < roll:
                columns_representation[i, 1, :] = 0

        return columns_representation

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

    def __new__ (cls, *args, **kwargs):
        instance = super(NeuralPlayer, cls).__new__(cls)

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
        announced_columns = 4,
        name = None,
        update_auto_slots = False,
        check_input = True
    ):
        super(NeuralPlayer, self).__init__(
            name,
            update_auto_slots,
            check_input
        )

        self._column_slot_layers = list(
            [ _np.array(A, ndmin = 2), _np.array(b, ndmin = 1), relu ]
                for A, b in column_slot_layers
        )
        if self._column_slot_layers:
            self._column_slot_layers[-1][2] = _sp_special.expit

        self._unlocked_replace_layers = list(
            [ _np.array(A, ndmin = 2), _np.array(b, ndmin = 1), relu ]
                for A, b in unlocked_replace_layers
        )
        if self._unlocked_replace_layers:
            self._unlocked_replace_layers[-1][2] = linear

        self._locked_replace_layers = list(
            [ _np.array(A, ndmin = 2), _np.array(b, ndmin = 1), relu ]
                for A, b in locked_replace_layers
        )
        if self._locked_replace_layers:
            self._locked_replace_layers[-1][2] = linear

        if announced_columns is None:
            self._announced_columns = None
        else:
            if not isinstance(
                announced_columns,
                (_collections.abc.Iterable, _np.ndarray)
            ):
                announced_columns = (announced_columns, )
            if not isinstance(
                announced_columns,
                (_collections.abc.Sequence, _np.ndarray)
            ):
                announced_columns = list(announced_columns)
            announced_columns = _np.array(announced_columns, copy = True)
            if not _np.issubdtype(announced_columns.dtype, _np.integer):
                raise TypeError()
            if announced_columns.ndim != 1:
                raise ValueError()
            if _np.any(announced_columns <= 0):
                raise ValueError()
            self._announced_columns = \
                _np.unique(announced_columns - 1) if len(announced_columns) \
                    else None

        self._column = None
        self._slot = None

    def observe_roll_results (
        self,
        columns,
        locked_column_index,
        roll,
        results
    ):
        self._column = None
        self._slot = None

    def choose_pre_filling_action_column (
        self,
        columns,
        roll,
        results,
        requirements
    ):
        if self._announced_columns is None:
            return None

        full = _np.row_stack(
            tuple(
                c.type_.is_lambda(
                    c.scores[_engine.Column.fillable_slots_array]
                ) for c in columns
            )
        )

        if _np.all(full[self._announced_columns]):
            return None

        X = _np.concatenate(
            (
                self._type._get_columns_representation(columns, roll).ravel(),
                self._type._get_roll_representation(roll).ravel(),
                self._type._get_results_representation(
                    _np.sort(results)
                ).ravel()
            )
        )

        y = self._type._transform(
            self._column_slot_layers,
            X
        ).reshape((len(columns), len(_engine.Column.fillable_slots_array)))
        y = _np.ascontiguousarray(y)
        for i, c in enumerate(columns):
            y[
                i,
                _np.isin(
                    _engine.Column.fillable_slots_array,
                    c.get_next_available_slots(),
                    assume_unique = True,
                    invert = True)
            ] = _neginf

        column, slot = _np.unravel_index(
            _np.argmax(y),
            shape = y.shape,
            order = 'C'
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
        return ((), { 'announcement': self._slot })

    def choose_replacements (
        self,
        columns,
        locked_column_index,
        roll,
        results
    ):
        X = None
        layers = None
        y = None

        if locked_column_index is None:
            X = _np.concatenate(
                (
                    _np.isin(
                        _engine.Column.fillable_slots_array,
                        self._slot,
                        assume_unique = True
                    ).ravel(),
                    self._type._get_roll_representation(roll).ravel(),
                    self._type._get_results_representation(
                        _np.sort(results)
                    ).ravel()
                )
            )
            layers = self._locked_replace_layers
        else:
            X = _np.concatenate(
                (
                    self._type._get_columns_representation(columns, roll).ravel(),
                    self._type._get_roll_representation(roll).ravel(),
                    self._type._get_results_representation(
                        _np.sort(results)
                    ).ravel()
                )
            )
            layers = self._unlocked_replace_layers

        y = _np.around(self._type._transform(layers, X))

        replace = _np.ones(len(results), dtype = _np.bool_)
        for i, r in enumerate(results):
            r -= 1
            if y[r] >= 0.5:
                replace[i] = False
                y[r] -= 1

        return replace if _np.any(replace) else None

    def choose_column_to_fill (self, columns, roll, results):
        X = _np.concatenate(
            (
                self._type._get_columns_representation(columns, roll).ravel(),
                self._type._get_roll_representation(roll).ravel(),
                self._type._get_results_representation(
                    _np.sort(results)
                ).ravel()
            )
        )

        y = self._type._transform(
            self._column_slot_layers,
            X
        ).reshape((len(columns), len(_engine.Column.fillable_slots_array)))
        y = _np.ascontiguousarray(y)
        if self._announced_columns is not None:
            y[self._announced_columns, :] = _neginf
        for i, c in enumerate(columns):
            y[
                i,
                _np.isin(
                    _engine.Column.fillable_slots_array,
                    c.get_next_available_slots(),
                    assume_unique = True,
                    invert = True)
            ] = _neginf

        column, slot = _np.unravel_index(
            _np.argmax(y),
            shape = y.shape,
            order = 'C'
        )

        self._column = column
        self._slot = slot

        return self._column

    def choose_slot_to_fill (self, columns, column_index, roll, results):
        return self._slot

    def set_post_filling_requirements (
        self,
        columns,
        column_index,
        slot,
        requirements
    ):
        return None
