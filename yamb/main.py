#!/usr/bin/env python

# -*- coding: utf-8 -*-

import random as _random
import sys as _sys

import numpy as _np

import _types
import engine as _engine
import booster as _booster

def main (argv = []):
    Die = _booster.boost_die(_engine.Die, class_name = 'BoostedDie')
    OrderedColumn = _booster.boost_column(
        _engine.OrderedColumn,
        class_name = 'BoostedOrderedColumn'
    )

    die = Die()

    column = OrderedColumn('up', check_input = False)

    while not column.is_full():
        column.fill_slot(
            column.get_next_available_slots()[0],
            #tuple(_np.sort(die.roll(5)))
            [ 1, 2, 3, 4, 5 ]
        )
        column.update_auto_slots()

    print(column.name)
    for s in _engine.Slot:
        if s == _engine.Slot.TOTAL:
            continue
        print("{slot}: {score}".format(slot = s.name, score = column[s]))

    print('')

    print(column.TOTAL)

    return 0

if __name__ == '__main__':
    exit_code = main(list(_sys.argv[1:]))

    exit(exit_code)
