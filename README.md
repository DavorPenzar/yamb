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
    * finishing the neural network based player ([`NeuralPlayer`](robo_players.py#L46)) (mostly inline documentation),
    * generation of many such players through [evolutionary algorithms (EA)](http://en.wikipedia.org/wiki/Evolutionary_algorithm) and seeing if their parameters and performance converges
        * results from [*expected_scores.csv*](expected_scores.csv)/[*expected_scores_us.csv*](expected_scores_us.csv) are used for filling missing (not yet filled) slots in the first generation, but subsequent generations should use averages from the previous generation,
    * hyperparameter optimisation of the [EA](http://en.wikipedia.org/wiki/Evolutionary_algorithm) approach.

If needed, the game engine may be altered/optimised before the development of such players if significant bottlenecks are diagnosed during the process. Alternatively, use `yamb.booster` submodule (and optimise it if needed, as well).

Finally, when such players are developed, a user interface around the game may be implemnted to allow human players to play versus eachother and/or a computer.

## Game Rules

The game may be played solitaire or multiplayer. The rules are the same, but, when played multiplayer, the players take turns rather than one player waiting until the other player(s) are finished. The upper bound of the number of players does not exist, but keep in mind that the game does not last short even if played singleplayer.

The game is played using 5 dice. In each turn, the player may roll the dice no more than three times. Also, in all rolls except the first one, the player may choose which dice they would like to roll, and which they would like to keep from the previous rolls (a die that was not rolled in the second roll but kept from the first roll may still be rolled in the third roll). At the end of the turn, the player uses values of the dice after the final roll (all 5 dice, including the ones kept from previous rolls). These values are used for filling game columns. In each turn, only one slot in only one column may be filled. When a slot is filled, it cannot be overwritten later.

Each column has its own rules regarding the order of filling slots, but scoring is common to all columns. Slots, descriptions and scores are presented in the table below (see below table why some slots are printed in bold):

| Slot            | Description                                                           | Scoring                                              |
|-----------------|-----------------------------------------------------------------------|------------------------------------------------------|
| One             | Sum of ones                                                           | Sum of dice showing ones                             |
| Two             | Sum of twos                                                           | Sum of dice showing twos                             |
| Three           | Sum of threes                                                         | Sum of dice showing threes                           |
| Four            | Sum of fours                                                          | Sum of dice showing fours                            |
| Five            | Sum of fives                                                          | Sum of dice showing fives                            |
| Six             | Sum of sixes                                                          | Sum of dice showing sixes                            |
| **Sum**         | Sum of *one*, *two*, &hellip;, *six*                                  | [one] + [two] + &hellip; + [six]; if &ge; 60, add 30 |
| Min             | Minimum                                                               | Sum of all 5 dice                                    |
| Max             | Maximum                                                               | Sum of all 5 dice                                    |
| **Difference**  | Difference between *max* and *min* times ones                         | [one] &times; ([max] - [min])                        |
| Two pairs       | Two distinct values appearing at least twice                          | Sum of the 4 dice making the pairs + 10              |
| Straight        | 5 consecutive values (1, 2, &hellip;, 5 or 2, 3, &hellip;, 6)         | 35 if lower (to 5), 45 if upper (to 6)               |
| Full house      | One value appearing thrice, another (different) value appearing twice | Sum of all 5 dice + 30                               |
| Carriage        | One value appearing at least four times                               | Sum of the 4 dice making the carriage + 40           |
| Yamb            | All five dice showing the same value                                  | Sum of all 5 dice + 50                               |
| **Collections** | Sum of *two pairs*, *straight*, &hellip;, *yamb*                      | [two pairs] + [straight] + &hellip; + [yamb]         |
| **Total**       | Final score                                                           | [sum] + [difference] + [collections]                 |

At the end of each turn the player must choose one of the thirteen non-bold slots to fill in; the other four bold slots are only used to calculate the player's final score. If the player chooses to fill in a slot, but the dice do not show valid values for the slot, the slot is filled with a score of 0 (e. g. if the dice do not all show the same value, but the player chooses to fill in *yamb*, they fill the slot with a score of 0). A slot filled with a score of 0 is still considered filled and cannot be overwritten later.

In a standard game of yamb, the player has a table of 4 columns:

1. down &ndash; the column is filled from top to bottom as in the previous table (e. g. *six* cannot be filled until all other numbers are filled),
2. up &ndash; the column is filled from bottom to top as in the previous table (e. g. *full house* cannot be filled until *yamb* and *carriage* are filled),
3. free &ndash; the column is filled freely, in no specific order,
4. announce &ndash; the player must announce the slot after the first roll (and they must fill this column at that slot at the end of the turn).

Popular additional and/or alternative columns are:

1. up-down &ndash; the column is filled from top to bottom and from bottom to top as in the previous table (e. g. *min* cannot be filled until all numbers or *max* and all collections are filled),
4. hand &ndash; must be filled immediately after the first roll (similar to the *announce* column, but consecutive rolls in the turn are not allowed),
2. middle &ndash; the column is filled from *min* upwards and *max* downwards,
3. late announce &ndash; the slot is announced after the second roll,
5. counter-announce (only available when played multiplayer) &ndash; the player must fill the same slot that the player before him has announced in their previous turn.

The game ends when all columns are completely filled. The player's final score is then calculated as the sum of *totals* per columns. The higher the final score the better (as one can conclude from the rules of scoring slots). If played multiplayer, the player with the highest score wins.

All four standard columns are implemented in this project. Of the alternative columns, the *up-down*, *hand* and *late announce* columns are also implemented. To implement the *counter-announce* column, a custom back-channel communication amongst the players must be implemented.
