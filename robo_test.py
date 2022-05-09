#!/usr/bin/env python

# -*- coding: utf-8 -*-

import math as _math
import numbers as _numbers
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

def _create_new_game (random_state = None):
    return _Yamb(
        columns = [
            _OrderedColumn(
                order = _engine.OrderedColumn.Order.DOWN,
                name = 'Down',
                check_input = False
            ),
            _OrderedColumn(
                order = _engine.OrderedColumn.Order.UP,
                name = 'Up',
                check_input = False
            ),
            _FreeColumn(name = 'Free', check_input = False),
            _AnnouncedColumn(name = 'Announce', check_input = False)
        ],
        random_state = _Die(random_state = random_state)
    )

def _create_random_layers (
    input_size,
    output_size,
    hidden_units = (),
    random_state = None
):
    if random_state is None:
        random_state = _np.random.default_rng()

    layers = list()

    for l in list(hidden_units) + [ output_size ]:
        A = _np.require(
            random_state.normal(0, 1, size = (l, input_size)),
            dtype = _np.float32,
            requirements = 'C'
        )
        b = _np.require(
            random_state.normal(0, 1, size = l),
            dtype = _np.float32,
            requirements = 'C'
        )

        layers.append((A, b))

        input_size = l

    return layers

def _create_random_player (
    column_slot_hidden_units = (),
    unlocked_replace_hidden_units = (),
    locked_replace_hidden_units = (),
    random_state = None
):
    if random_state is None:
        random_state = _np.random.default_rng()

    return _players.NeuralPlayer(
        column_slot_layers = _create_random_layers(
            *_players.NeuralPlayer.calculate_column_slot_units(),
            hidden_units = column_slot_hidden_units,
            random_state = random_state
        ),
        unlocked_replace_layers = _create_random_layers(
            *_players.NeuralPlayer.calculate_unlocked_replace_units(),
            hidden_units = unlocked_replace_hidden_units,
            random_state = random_state
        ),
        locked_replace_layers = _create_random_layers(
            *_players.NeuralPlayer.calculate_locked_replace_units(),
            hidden_units = locked_replace_hidden_units,
            random_state = random_state
        ),
        announced_columns = 3,
        update_auto_slots = False,
        check_input = False
    )

def _create_next_generation (
    players,
    scores,
    top_players = 0.1,
    skip_top = 0,
    n_players = None,
    *,
    spread = 10,
    centre = _np.mean,
    centre_kwargs = { 'axis': 0, 'keepdims': False },
    random_state = None
):
    if random_state is None:
        random_state = _np.random.default_rng()

    if n_players is None:
        n_players = len(players)
    if not isinstance(top_players, (_numbers.Integral, int, _np.integer)):
        top_players = int(round(top_players * len(players)))
    if skip_top is None:
        skip_top = 0
    if not isinstance(skip_top, (_numbers.Integral, int, _np.integer)):
        skip_top = int(round(skip_top * len(players)))

    I = _np.flip(
        _np.argsort(_np.asarray(scores))
    )[skip_top:skip_top + top_players].copy()

    players = list(players[i] for i in I)

    del I

    column_slot_layers = list()
    unlocked_replace_layers = list()
    locked_replace_layers = list()
    for i in range(top_players):
        for j, l in enumerate(players[i].column_slot_layers):
            if len(column_slot_layers) <= j:
                column_slot_layers.append([ (l[0], l[1]) ])
            else:
                column_slot_layers[j].append((l[0], l[1]))
        for j, l in enumerate(players[i].unlocked_replace_layers):
            if len(unlocked_replace_layers) <= j:
                unlocked_replace_layers.append([ (l[0], l[1]) ])
            else:
                unlocked_replace_layers[j].append((l[0], l[1]))
        for j, l in enumerate(players[i].locked_replace_layers):
            if len(locked_replace_layers) <= j:
                locked_replace_layers.append([ (l[0], l[1]) ])
            else:
                locked_replace_layers[j].append((l[0], l[1]))

    column_slot_layers_centres = list(
        (
            centre(
                list(l[i][0] for i in range(top_players)),
                **centre_kwargs
            ),
            centre(
                list(l[i][1] for i in range(top_players)),
                **centre_kwargs
            )
        ) for l in column_slot_layers
    )
    unlocked_replace_layers_centres = list(
        (
            centre(
                list(l[i][0] for i in range(top_players)),
                **centre_kwargs
            ),
            centre(
                list(l[i][1] for i in range(top_players)),
                **centre_kwargs
            )
        ) for l in unlocked_replace_layers
    )
    locked_replace_layers_centres = list(
        (
            centre(
                list(l[i][0] for i in range(top_players)),
                **centre_kwargs
            ),
            centre(
                list(l[i][1] for i in range(top_players)),
                **centre_kwargs
            )
        ) for l in locked_replace_layers
    )

    del column_slot_layers
    del unlocked_replace_layers
    del locked_replace_layers

    combo = random_state.uniform(
        0,
        1,
        size = (n_players - top_players, top_players)
    ).astype(_np.float32)
    combo /= _np.sum(combo, axis = 1, keepdims = True)

    players += list(None for _ in range(n_players - top_players))
    for i in range(top_players, n_players):
        column_slot_layers = list(
            [ _np.zeros_like(l[0]), _np.zeros_like(l[1]) ]
                for l in column_slot_layers_centres
        )
        unlocked_replace_layers = list(
            [ _np.zeros_like(l[0]), _np.zeros_like(l[1]) ]
                for l in unlocked_replace_layers_centres
        )
        locked_replace_layers = list(
            [ _np.zeros_like(l[0]), _np.zeros_like(l[1]) ]
                for l in locked_replace_layers_centres
        )

        for p, f in zip(players, combo[i - top_players]):
            for j in range(len(column_slot_layers)):
                for k in range(2):
                    column_slot_layers[j][k] += f * p.column_slot_layers[j][k]
            for j in range(len(unlocked_replace_layers_centres)):
                for k in range(2):
                    unlocked_replace_layers[j][k] += \
                        f * p.unlocked_replace_layers[j][k]
            for j in range(len(unlocked_replace_layers_centres)):
                for k in range(2):
                    locked_replace_layers[j][k] += \
                        f * p.locked_replace_layers[j][k]

            for j in range(len(column_slot_layers)):
                for k in range(2):
                    column_slot_layers[j][k] = \
                        column_slot_layers_centres[j][k] + \
                            spread * (
                                column_slot_layers[j][k] -
                                column_slot_layers_centres[j][k]
                            )
            for j in range(len(unlocked_replace_layers_centres)):
                for k in range(2):
                    unlocked_replace_layers[j][k] = \
                        unlocked_replace_layers_centres[j][k] + \
                            spread * (
                                unlocked_replace_layers[j][k] -
                                unlocked_replace_layers_centres[j][k]
                            )
            for j in range(len(locked_replace_layers_centres)):
                for k in range(2):
                    locked_replace_layers[j][k] = \
                        locked_replace_layers_centres[j][k] + \
                            spread * (
                                locked_replace_layers[j][k] -
                                locked_replace_layers_centres[j][k]
                            )

        players[i] = _players.NeuralPlayer(
            column_slot_layers = column_slot_layers,
            unlocked_replace_layers = unlocked_replace_layers_centres,
            locked_replace_layers = locked_replace_layers_centres,
            announced_columns = 3,
            update_auto_slots = False,
            check_input = False
        )

    return players

def main (argv = []):
    R = _np.random.default_rng(2022)

    n_players = 100
    n_generations = 20
    original_spread = 10

    players = list(
        _create_random_player(
            column_slot_hidden_units = (256, 1024, 32),
            unlocked_replace_hidden_units = (256, 16),
            locked_replace_hidden_units = (256, 1024, 32),
            random_state = R
        ) for i in range(n_players)
    )

    for g in range(n_generations):
        games = list(
            _create_new_game(random_state = R) for _ in range(n_players)
        )
        scores = list(0 for _ in range(n_players))

        for i in range(n_players):
            games[i] = players[i].play(games[i])
            games[i].update_auto_slots()
            scores[i] = games[i].get_total_score(_engine.Slot.TOTAL)

        scores = _pd.Series(scores, dtype = _np.float32, name = 'Score')

        print(f"Generation {g + 1:d}:")
        print(scores.describe())
        print('')

        players = _create_next_generation(
            players,
            scores,
            top_players = 0.05,
            skip_top = 0.00501,
            n_players = n_players,
            spread = original_spread / _math.sqrt(g + 1),
            centre = _np.median,
            random_state = R
        )

    return 0

if __name__ == '__main__':
    exit_code = main(list(_sys.argv[1:]))

    exit(exit_code)
