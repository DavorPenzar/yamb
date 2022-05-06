#!/usr/bin/env python

# -*- coding: utf-8 -*-

import random as _random
import sys as _sys

import numpy as _np
import pandas as _pd

import yamb as _engine

class _SimpleConsolePlayer (_engine.Player):
    @classmethod
    def _get_requirements (cls, requirements):
        if (
            isinstance(requirements, (bool, _np.bool_)) and
            requirements == True
        ):
            return ((), {})

        args = list(None for _ in requirements[0])
        kwargs = dict()
        for j, a in enumerate(requirements[0]):
            args[j] = a(int(input().strip()))
        for k, a in requirements[1].items():
            kwargs[k] = a(int(input(f"{k:s} = ").strip()))

        args = tuple(args)

        return (args, kwargs)

    def observe_roll_results (
        self,
        columns,
        locked_column_index,
        roll,
        results
    ):
        if roll:
            print(f"Results: {list(results)}")
        else:
            print(
                _pd.concat(list(c.to_pandas(True) for c in columns), axis = 1)
            )

    def choose_pre_filling_action_column (
        self,
        columns,
        roll,
        results,
        requirements
    ):
        print(f"Requirements: {requirements}")

        column = None

        free_columns = list(
            c for c in range(len(columns)) if not columns[c].is_full(True)
        )
        if len(free_columns) == 1:
            column = free_columns[0]
        else:
            column = input('Require for column: ').strip()
            if column:
                column = int(column) - 1
            else:
                column = None

        return column

    def set_pre_filling_requirements (
        self,
        columns,
        column_index,
        roll,
        results,
        requirements
    ):
        return self._type._get_requirements(requirements)

    def choose_replacements (
        self,
        columns,
        locked_column_index,
        roll,
        results
    ):
        replace = None

        my_replace = input('Replace: ')
        if my_replace:
            my_replace = frozenset(int(r.strip()) for r in my_replace.split(','))
        if my_replace:
            replace = list(((r + 1) in my_replace) for r in range(len(results)))

        return replace

    def choose_column_to_fill (self, columns, results):
        column = None

        free_columns = list(
            c for c in range(len(columns)) if not columns[c].is_full()
        )
        if len(free_columns) == 1:
            column = free_columns[0]
        else:
            column = int(input('Enter in column: ').strip()) - 1

        return column

    def choose_slot_to_fill (self, columns, column_index, results):
        slot = None

        next_slots = columns[column_index].get_next_available_slots()
        if len(next_slots) == 1:
            slot = next_slots[0]
        else:
            slot = int(input('Enter in slot: ').strip())

        return slot

    def set_post_filling_requirements (
        self,
        columns,
        column_index,
        slot,
        requirements
    ):
        return self._type._get_requirements(requirements)

def main (argv = []):
    player = _SimpleConsolePlayer(update_auto_slots = True)

    game = player.play()

    print('')

    print(f"Final score: {game.get_total_score()}")

    return 0

if __name__ == '__main__':
    exit_code = main(list(_sys.argv[1:]))

    exit(exit_code)
