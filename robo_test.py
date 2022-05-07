#!/usr/bin/env python

# -*- coding: utf-8 -*-

import itertools as _itertools
import math as _math
import random as _random
import sys as _sys

import numpy as _np
import pandas as _pd

import yamb as _engine
import yamb.booster as _booster
import robo_players as _players

_Die = _booster.boost_die(_engine.FiniteDie)
_OrderedColumn = _booster.boost_column(_engine.OrderedColumn)
_FreeColumn = _booster.boost_column(_engine.FreeColumn)
_AnnouncedColumn = _booster.boost_column(_engine.AnnouncedColumn)
_Yamb = _booster.boost_yamb(_engine.Yamb)

def main (argv = []):
    R = _np.random.default_rng(2022)

    column_representation_size = int(
        round(_math.prod([ 2, len(_engine.Column.fillable_slots_array) ]))
    )
    columns_representation_size = int(
        round(_math.prod([ 4, column_representation_size ]))
    )
    column_size = len(_engine.Column.fillable_slots_array)
    columns_size = int(round(_math.prod([ 4, column_size ])))
    roll_size = 1
    results_size = len(_engine.Die.sides)

    game = _Yamb(
        columns = [
            _OrderedColumn(order = 'down', name = 'Down'),
            _OrderedColumn(order = 'up', name = 'Up'),
            _FreeColumn(name = 'Free'),
            _AnnouncedColumn(name = 'Announce')
        ]
    )

    player = _players.NeuralPlayer(
        column_slot_layers = [
            (
                R.normal(
                    size = (256, columns_representation_size + roll_size + results_size)
                ),
                R.normal(size = 256)
            ),
            (
                R.normal(
                    size = (64, 256)
                ),
                R.normal(size = 64)
            ),
            (
                R.normal(
                    size = (16, 64)
                ),
                R.normal(size = 16)
            ),
            (
                R.normal(
                    size = (columns_size, 16)
                ),
                R.normal(size = columns_size)
            )
        ],
        unlocked_replace_layers = [
            (
                R.normal(
                    size = (256, columns_representation_size + roll_size + results_size)
                ),
                R.normal(size = 256)
            ),
            (
                R.normal(
                    size = (64, 256)
                ),
                R.normal(size = 64)
            ),
            (
                R.normal(
                    size = (16, 64)
                ),
                R.normal(size = 16)
            ),
            (
                R.normal(
                    size = (results_size, 16)
                ),
                R.normal(size = results_size)
            )
        ],
        locked_replace_layers = [
            (
                R.normal(
                    size = (256, column_size + roll_size + results_size)
                ),
                R.normal(size = 256)
            ),
            (
                R.normal(
                    size = (64, 256)
                ),
                R.normal(size = 64)
            ),
            (
                R.normal(
                    size = (16, 64)
                ),
                R.normal(size = 16)
            ),
            (
                R.normal(
                    size = (results_size, 16)
                ),
                R.normal(size = results_size)
            )
        ]
    )

    game = player.play(game)

    print(game.to_pandas(True))
    print(f"Final score: {game.get_total_score()}")

    return 0

if __name__ == '__main__':
    exit_code = main(list(_sys.argv[1:]))

    exit(exit_code)
