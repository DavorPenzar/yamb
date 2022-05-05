#!/usr/bin/env python

# -*- coding: utf-8 -*-

import random as _random
import sys as _sys

import numpy as _np
import pandas as _pd

import yamb as _engine

def _get_requirements (requirements):
    if isinstance(requirements, (bool, _np.bool_)) and requirements == True:
        return ((), {})

    args = list(None for _ in requirements[0])
    kwargs = dict()
    for j, a in enumerate(requirements[0]):
        args[j] = a(int(input().strip()))
    for k, a in requirements[1].items():
        kwargs[k] = a(int(input(f"{k:} = ").strip()))

    args = tuple(args)

    return (args, kwargs)

def main (argv = []):
    yamb = _engine.Yamb(
        random_state = _engine.FiniteDie(
            random_state = _np.random.default_rng()
        )
    )

    while not yamb.is_full():
        replace = None

        print(yamb.to_pandas(str_index = True))

        for roll in range(4):
            if not yamb.can_roll():
                break

            results, requirements = yamb.roll_dice(roll, replace)

            if roll:
                print(f"Results: {repr(results)}")

            if any(r for r in requirements):
                print(f"Requirements: {repr(requirements)}")

                column = None
                free_columns = list(c for c in range(len(yamb)) if not yamb[c].is_full())
                if len(free_columns) == 1:
                    column = free_columns[0]
                else:
                    column = input('Require for column: ').strip()
                if column:
                    column = int(column) - 1
                    if requirements[column]:
                        args, kwargs = _get_requirements(requirements[column])
                        yamb.make_pre_filling_action(column, roll, *args, **kwargs)

            if roll and roll != 3:
                my_replace = input('Replace: ')
                if my_replace:
                    my_replace = frozenset(int(r.strip()) for r in my_replace.split(','))
                if not my_replace:
                    break

                replace = list(((r + 1) in my_replace) for r in range(yamb.n_dice))

        column = yamb.which_column_is_locked()
        if column is None:
            free_columns = list(c for c in range(len(yamb)) if not yamb[c].is_full())
            if len(free_columns) == 1:
                column = free_columns[0]
            else:
                column = int(input('Enter in column: ').strip()) - 1

        next_slots = yamb[column].get_next_available_slots()
        slot = None
        if len(next_slots) == 1:
            slot = next_slots[0]
        else:
            slot = int(input('In slot: ').strip())

        requirements = yamb.end_turn(column, slot)
        if requirements:
            args, kwargs = _get_requirements(requirements)
            yamb.make_post_filling_action(column, *args, **kwargs)

        yamb.update_auto_slots()

        print('')

    print(yamb.to_pandas(str_index = True))
    print(f"Final score: {sum(yamb[c].TOTAL for c in range(len(yamb)))}")

    return 0

if __name__ == '__main__':
    exit_code = main(list(_sys.argv[1:]))

    exit(exit_code)
