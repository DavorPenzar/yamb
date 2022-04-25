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

    n_iter = 10000

    scores = _np.zeros(n_iter, dtype = _np.float32)

    for i in range(n_iter):
        column = OrderedColumn(OrderedColumn.Order.DOWN, check_input = False)

        while True:
            available_slots = column.get_next_available_slots()
            if len(available_slots): 
                column.fill_slot(available_slots[0], die.roll(5))
            else:
                break
        column.update_auto_slots()

        scores[i] = column[_engine.Slot.TOTAL]

    print(f"Mean: {_np.mean(scores)}")
    print(f"SD:   {_np.std(scores, ddof = 1)}")
    print(f"Min:  {_np.amin(scores)}")
    print(f"25 %: {_np.percentile(scores, 25)}")
    print(f"50 %: {_np.median(scores)}")
    print(f"75 %: {_np.percentile(scores, 75)}")
    print(f"Max:  {_np.amax(scores)}")

    return 0

if __name__ == '__main__':
    exit_code = main(list(_sys.argv[1:]))

    exit(exit_code)
