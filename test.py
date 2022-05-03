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
        args[j] = a(int(input()))
    for k, a in requirements[1].items():
        kwargs[k] = a(int(input(f"{k} = ")))

    args = tuple(args)

    return (args, kwargs)

def main (argv = []):
    yamb = _engine.Yamb()

    while not yamb.is_full():
        replace = None

        print(
            _pd.DataFrame(
                _np.column_stack(
                    list(c for c in yamb.columns)
                ),
                columns = list(c.name for c in yamb.columns),
                index = list(s.name for s in _engine.Slot)
            )
        )

        for i in range(4):
            results, requirements = yamb.roll_dice(i, replace)

            if i:
                print(f"Results: {repr(results)}")

            if any(r for r in requirements):
                print(f"Requirements: {repr(requirements)}")

                c = input('Require for column: ')
                if c:
                    c = int(c) - 1
                    if requirements[c]:
                        args, kwargs = _get_requirements(requirements[c])
                        yamb.make_pre_filling_action(c, i, *args, **kwargs)

            if i and i != 3:
                my_replace = input('Replace: ')
                if my_replace:
                    my_replace = frozenset(int(r.strip()) for r in my_replace.split(','))
                if not my_replace:
                    break

                replace = list(((r + 1) in my_replace) for r in range(5))

        column = yamb.which_column_is_locked()
        if column is None:
            column = int(input('Enter in column: ')) - 1

        next_slots = yamb.columns[column].get_next_available_slots()
        slot = None
        if len(next_slots) == 1:
            slot = next_slots[0]
        else:
            slot = int(input('In slot: '))

        requirements = yamb.end_turn(column, slot)
        if requirements:
            args, kwargs = _get_requirements(requirements)
            yamb.make_post_filling_action(column, *args, **kwargs)

        yamb.update_auto_slots()

        print('')

    return 0

if __name__ == '__main__':
    exit_code = main(list(_sys.argv[1:]))

    exit(exit_code)
