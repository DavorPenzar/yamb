#!/usr/bin/env python

# -*- coding: utf-8 -*-

import fractions as _fractions
import itertools as _itertools
import math as _math
import sys as _sys

import matplotlib as _mpl
import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd

import seaborn as _sns

import yamb as _engine
import yamb.booster as _booster

_BoostedFreeColumn = _booster.boost_column(_engine.FreeColumn)

_plt.style.use('seaborn')
_sns.set_theme()

n_dice = 5
n_outcomes = len(_engine.Die.sides) ** n_dice

def _nice_subplots_grid (nsubs):
    r = _math.sqrt(nsubs)

    nrows = int(_math.floor(r))
    ncols = int(_math.ceil(r))

    if nrows < nsubs / ncols:
        nrows += 1

    return { 'nrows': nrows, 'ncols': ncols }

def main (argv = []):
    ## *** TEST ALL POSSIBLE OUTCOMES ***

    scores = _np.zeros((n_outcomes, len(_engine.Slot)), dtype = _np.int32)

    for i, o in enumerate(
        _itertools.product(_engine.Die.sides, repeat = n_dice)
    ):
        aux_column = _BoostedFreeColumn(check_input = False)
        for s in _engine.Column.fillable_slots:
            aux_column.fill_slot(s, o)
        aux_column.update_auto_slots()

        _np.copyto(scores[i], aux_column, casting = 'unsafe')

    scores = _pd.DataFrame(
        scores,
        columns = _pd.Index(list(s.name for s in _engine.Slot), name = 'Slot')
    )
    scores_sum = scores.sum(axis = 0)
    scores_stats = scores.describe().T.copy()
    scores_stats['std'] = scores.std(axis = 0, ddof = 0) # <- use population
                                                         # SD since all
                                                         # possible outcomes
                                                         # are considered


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
            int(scores_sum[s]),
            n_outcomes
        )
    aux_column.update_auto_slots()

    for s in _engine.Slot:
        f = aux_column[s]

        expectations.loc[s, 'Value'] = float(f)
        expectations.loc[s, 'Fraction'] = str(f)
        expectations.loc[s, 'Numerator'] = f.numerator
        expectations.loc[s, 'Denominator'] = f.denominator


    ## *** PLOT HISTOGRAMS ***

    hist_fig, hist_ax = _plt.subplots(
        sharex = False,
        sharey = False,
        **_nice_subplots_grid(len(_engine.Slot))
    )
    ax_order = 'F' if hist_ax.ndim > 1 and _np.isfortran(hist_ax) else 'C'

    for s in _engine.Slot:
        ax_idx = _np.unravel_index(s, hist_ax.shape, order = ax_order)
        _sns.histplot(
            scores[s.name],
            stat = 'density',
            kde = True,
            ax = hist_ax[ax_idx]
        )


    ## *** OUTPUT RESULTS ***

    print('Descriptive statistics:')
    print(scores_stats)

    print('')

    print('Expectations:')
    print(expectations)

    _plt.show()

    return 0

if __name__ == '__main__':
    exit_code = main(list(_sys.argv[1:]))

    exit(exit_code)
