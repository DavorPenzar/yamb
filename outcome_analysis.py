#!/usr/bin/env python

# -*- coding: utf-8 -*-

import fractions as _fractions
import itertools as _itertools
import sys as _sys

import numpy as _np
import pandas as _pd

import yamb as _engine
import yamb.booster as _booster

def main (argv = []):
    BoostedFreeColumn = _booster.boost_column(_engine.FreeColumn)

    n_dice = 5
    n_outcomes = len(_engine.Die.sides) ** n_dice

    ## *** TEST ALL POSSIBLE OUTCOMES ***

    scores = _np.zeros((n_outcomes, len(_engine.Slot)), dtype = _np.int32)

    for i, o in enumerate(_itertools.product(_engine.Die.sides, repeat = 5)):
        aux_column = BoostedFreeColumn(check_input = False)
        for s in _engine.Column.fillable_slots:
            aux_column.fill_slot(s, o)
        aux_column.update_auto_slots()
        _np.copyto(scores[i], aux_column, casting = 'unsafe')

    scores = _pd.DataFrame(
        scores,
        columns = _pd.Index(list(s.name for s in _engine.Slot), name = 'Slot')
    )
    total_scores = scores.sum(axis = 0)

    ## *** CALCULATE EXPECTED OUTCOMES ***

    expectations = _pd.DataFrame(
        list([ s.name, 0.0, '', 0, 0 ] for s in _engine.Slot),
        columns = [
            'Slot_Name',
            'Value',
            'Fraction',
            'Numerator',
            'Denominator'
        ],
        index = _pd.RangeIndex(len(_engine.Slot), name = 'Slot_Index')
    )

    # Auxiliary column is needed to accurately calculate auto slots based on
    # other slots' scores.
    aux_column = _engine.FreeColumn(check_input = False)
    for s in _engine.Column.fillable_slots:
        aux_column.scores[s] = _fractions.Fraction(
            int(total_scores[s]),
            n_outcomes
        )
    aux_column.update_auto_slots()

    for s in _engine.Slot:
        f = aux_column[s]

        expectations.loc[s, 'Value'] = float(f)
        expectations.loc[s, 'Fraction'] = str(f)
        expectations.loc[s, 'Numerator'] = f.numerator
        expectations.loc[s, 'Denominator'] = f.denominator

    ## *** OUTPUT RESULTS ***

    print(scores.describe().T)
    print('')
    print(expectations)

    return 0

if __name__ == '__main__':
    exit_code = main(list(_sys.argv[1:]))

    exit(exit_code)
