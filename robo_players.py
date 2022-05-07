# -*- coding: utf-8 -*-

import collections as _collections
import functools as _functools
import math as _math
import types as _types

import numpy as _np

import yamb as _engine

_max_cache_size = 0x20

def linear (cls, x):
    return x

def relu_complete (
    x,
    max_value = _math.inf,
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

def relu (cls, x):
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
        return _np.dot(A, b) + x

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
        column = _np.array(
            [
                scores[_engine.Column.fillable_slots_array],
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
            cls.expected_results,
            where = _np.isna(column[0])
        )

        return column

    @classmethod
    def _get_column_representation (cls, column):
        return cls._get_hashable_column_representation(
            tuple(column.scores),
            tuple(column.get_next_available_results())
        )

    @classmethod
    def _get_columns_representation (cls, columns):
        return _np.array(
            list(cls._get_column_representation(c) for c in columns)
        )

    @classmethod
    @_functools.lru_cache(maxsize = _max_cache_size)
    def _get_roll_representation (cls, roll):
        return _np.array([ roll ])

    @classmethod
    @_functools.lru_cache(maxsize = _max_cache_size)
    def _get_hashable_results_representation (cls, results):
        bow_results = _np.zeros(len(_engine.Die.sides))
        for r, c in _collections.Counter(results).items:
            bow_results[r] = c

        return bow_results

    @classmethod
    def _get_results_representation (cls, results):
        return cls._get_hashable_results_representation(
            results if isinstance (results, _collections.abc.Hashable)
                else tuple(results)
        )

    def __init__ (
        self,
        name = None,
        update_auto_slots = False,
        check_input = True
    ):
        super().__init__(name, update_auto_slots, check_input)

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
        if columns[-1].is_full(True):
            return None

        X = _np.concatenate(
            (
                self._type._get_columns_representation(columns).ravel(),
                self._type._get_roll_representation(roll).ravel(),
                self._type._get_results_representation(
                    _np.sort(results)
                ).ravel()
            )
        )

        y = _np.around(self._type._transform([], X))

        return None if y[0] <= 0 else len(columns) - 1

    def set_pre_filling_requirements (
        self,
        columns,
        column_index,
        roll,
        results,
        requirements
    ):
        column = columns[column_index]

        X = _np.concatenate(
            (
                self._type._get_columns_representation(columns[:-1]).ravel(),
                self._type._get_column_representation(
                    columns[column]
                ).ravel(),
                self._type._get_results_representation(
                    _np.sort(results)
                ).ravel()
            )
        )

        y = self._type._transform([], X)
        for i, s in enumerate(_engine.Column.fillable_slots_array):
            if not column.type_.is_lambda(column[s]):
                y[i] = -_math.inf

        return (
            (),
            {
                'announcement': \
                    _engine.Column.fillable_slots_array[_np.argmax(y)]
            }
        )

    def choose_replacements (
        self,
        columns,
        locked_column_index,
        roll,
        results
    ):
        X = _np.concatenate(
            (
                self._type._get_columns_representation(columns).ravel(),
                self._type._get_roll_representation(roll).ravel(),
                self._type._get_results_representation(
                    _np.sort(results)
                ).ravel()
            )
        )

        y = _np.around(self._type._transform([], X))

        replace = _np.ones(len(results), dtype = _np.bool_)
        for i, r in enumerate(results):
            if y[r] >= 0.5:
                replace[i] = False
                y[r] -= 1

        return replace if _np.any(replace) else None

    def choose_column_to_fill (self, columns, results):
        X = _np.concatenate(
            (
                self._type._get_columns_representation(columns).ravel(),
                self._type._get_results_representation(
                    _np.sort(results)
                ).ravel()
            )
        )

        y = self._type._transform([], X)
        for i, c in enumerate(columns):
            if isinstance(c, _engine.AnnouncedColumn) or c.is_full(True):
                y[i] = -_math.inf

        return _np.argmax(y)

    def choose_slot_to_fill (self, columns, column_index, results):
        column = columns[column]

        X = _np.concatenate(
            (
                self._type._get_columns_representation(columns).ravel(),
                self._type._get_column_representation(
                    columns[column]
                ).ravel(),
                self._type._get_results_representation(
                    _np.sort(results)
                ).ravel()
            )
        )

        y = self._type._transform([], X)
        for i, s in enumerate(_engine.Column.fillable_slots_array):
            if not column.type_.is_lambda(column[s]):
                y[i] = -_math.inf

        return _engine.Column.fillable_slots_array[_np.argmax(y)]
