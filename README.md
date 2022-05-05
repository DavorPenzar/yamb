# Yamb

Light-weight [*Python*](http://docs.python.org/) implementation of the popular dice game [*yamb*](http://en.wikipedia.org/wiki/Yamb_(game)) suitable for training automated computer players.

## Usage

Currently, one can play the game by running the following command:

```bash
python3 test.py

```

This is still experimental and the main purpose is to test the engine back-end implementation. However, the game can actually already be played.

This is your score table:

```
                OrderedColumn_DOWN OrderedColumn_UP FreeColumn AnnouncedColumn
Slot                                                                          
TOTAL                         None             None       None            None
ONE                           None             None       None            None
TWO                           None             None       None            None
THREE                         None             None       None            None
FOUR                          None             None       None            None
FIVE                          None             None       None            None
SIX                           None             None       None            None
NUMBERS_SUM                   None             None       None            None
MAX                           None             None       None            None
MIN                           None             None       None            None
SUMS_DIFFERENCE               None             None       None            None
TWO_PAIRS                     None             None       None            None
STRAIGHT                      None             None       None            None
FULL_HOUSE                    None             None       None            None
CARRIAGE                      None             None       None            None
YAMB                          None             None       None            None
COLLECTIONS_SUM               None             None       None            None
```

It is automatically filled in through game play.

Below the table, the following lines shall appear:

```
Results: [5, 2, 5, 1, 2]
Requirements: [None, None, None, ((), {'announcement': <enum 'Slot'>})]
Require for column: 4
announcement = 9
Replace: 1, 3
Results: [4, 2, 6, 1, 2]
Replace: 1, 3
Results: [1, 2, 1, 1, 2]
Enter in column: 4
In slot: 9
```

Dice roll results are provided in a line as the following: `Results: [5, 2, 5, 1, 2]`. When prompted to replace a die, refer to its one-indexed position, enumerating from left to right&mdash;e. g. to replace the first result of five, enter `1`; to replace the result of one, enter `4` etc.). To replace multiple dice, delimit their positions using a comma (white space after the comma is optional)&mdash;e. g. to replace both fives, enter `1, 3`.

When prompted to enter the column, use its one-indexed position as in the table above, enumerating from left to right: `"OrderedColumn_DOWN"` is `1`, `"AnnouncedColumn"` is `4`. When prompted to enter the slot, use its zero-indexed position as in the table above, enumerating from top to bottom: `"ONE"` is `1`, `"SIX"` is `6`, `"MAX"` is `8`, `"MIN"` is `9`, `"TWO_PAIRS"` is `11`, `"YAMB"` is `15`. You will not need to use the index `0` as this is the automatically filled-in slot (`"TOTAL"`).  You shall not be asked to enter the column if there is no choice (e. g. only one column is not completely filled in yet, or you have announced a slot in the announced column), and you shall not be asked to enter a slot if there is no choice (e. g. only one slot is next available in the slot, or a slot is announced).

A special output line may look like the following: `Requirements: [None, None, None, ((), {'announcement': <enum 'Slot'>})]`. It tells you which columns have special requirements prior to filling (the same order as in the table above: leftmost is `"OrderedColumn_DOWN"`, rightmost is `"AnnouncedColumn"`). `None` or `False` indicates no requirements; otherwise the requirements are expressed through their argument names and expected types. These names and types shall not be thoroughly described here as only a single argument may be expected for now.

Below the special output line you shall be asked to enter column position (one-index position from left to right) by the following line: `Require for column: `. If no column is entered or a column with no requirements is entered, no special action is required. Otherwise (including the case where only the `"AnnouncedColumn"` is not completely filled) you shall be prompted to enter the requirements for the chosen column, in the form of an output such as `announcement = `. Simply enter the desired announced slot using its one-indexed position (top to bottom) as in the table above.

## Intended Usage

The intended usage was actually to develop an automated computer player of the game which would play using a strategy as close to the optimal as possible. To do that, a game environment framework was needed, therefore the game *engine* was implemented. The next steps would include:

1. finishing the game engine&mdash;finishing the `yamb.Yamb` class (mostly inline documentation, but also code revision and possible changes), finishing the `yamb` [*Python*](http://docs.python.org/) package,
2. developing an automated player through:
    * generation of many such players through [evolutionary algorithms (EA)](http://en.wikipedia.org/wiki/Evolutionary_algorithm) and seeing if their parameters and performance converges&mdash;these may include neural networks, decision trees, ensembles etc.
        * results from [*expected_results.csv*](expected_results.csv)/[*expected_results_us.csv*](expected_results_us.csv) may be used for filling missing (not yet filled) slots; optionally, auto-filled slots may be recalculated using actual values for filled slots and expected results for slots that are not yet filled (e. g. if slots 1 and 3 are filled with values 4 and 9 respectively, but other number slots are still missing, `NUMBERS_SUM` may be recalculated to the value of 27.1666666666666667 from the original 17,5 which assumes values 0,8333333333333333 in slot 1 and 2,5 in slot 3),
    * hyperparameter optimisation of the [EA](http://en.wikipedia.org/wiki/Evolutionary_algorithm) approach.

If needed, the game engine may be altered/optimised before the development of such players if significant bottlenecks are diagnosed during the process. Alternatively, use `yamb.booster` submodule (and optimise it if needed, as well).

Finally, when such players are developed, a user interface around the game may be implemnted to allow human players to play versus eachother and/or a computer.
