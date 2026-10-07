## Introduction to Sports Betting

Sports betting is a two-player game between a bookmaker and a bettor.
The bookmaker's job is to offer bettors betting opportunities with the goal of maximizing their own profit.
So when, for example, a basketball game between two teams is approaching, the bookmaker posts so-called odds on the possible outcomes.
The bettor can choose which games and outcomes they would like to bet on.
If the bettor bets on an outcome that happens, the bettor wins their stake multiplied by the bookmaker's odds.
If that outcome does not happen, the bettor loses their stake.

**Example:**
For the first game, Vancouver - Nashville, the bookmaker posts odds of 2.08 on a home win (Vancouver) and 1.72 on a Nashville win.
For the second game, Anaheim - New Jersey, the bookmaker posts odds of 1.19 on an Anaheim win and 4.55 on a New Jersey win.
A bettor with 1000 CZK available decides to bet 100 CZK on the home team Vancouver to win, i.e. at odds of 2.08, and 50 CZK on the visiting New Jersey to win, i.e. at odds of 4.55. After placing the stakes, they have 850 CZK in their account. Suppose both games end in home wins, i.e. for Vancouver and Anaheim. From the first game, the bettor wins 2.08 x 100 = 208 CZK. The 50 CZK bet on the second game is lost, since Anaheim won and not New Jersey. In the end, the bettor has 1058 CZK in their account.

## Assignment

Your task is to program the most successful bettor possible. In our competition, however, what matters is not how many games you have watched, but your ability to design, implement, and test a model that learns to predict the results of upcoming games from historical data.

Your bettor is represented by the [`Model`](src/model.py) class with a `place_bets` method, through which, on every day that there are betting opportunities on the market, you receive a summary, the betting opportunities, and a data increment. We expect you to return the bets you wish to place. Since individual editions of the competitions (seasons) last several months, your model will have to adapt during the season to new results and to the players' form.

## Data Types

In the `place_bets` method, you will encounter four data types in total across its arguments - `summary` containing a [Summary DataFrame](#summary-dataframe), `opps` containing an [Opps DataFrame](#opps-dataframe), and `inc` containing a [Games DataFrame](#games-dataframe). The last, fifth type is the [Bets DataFrame](#bets-dataframe) that you return.

### Summary DataFrame

The summary dataframe contains information about the current state of the betting environment. It tells you the current date and what funds you have available.

|    |   Bankroll | Date       |   Min_bet |   Max_bet |
|---:|-----------:|:-----------|----------:|----------:|
|  0 |    1000    | 1993-11-06 |    1      |    100    |

<table>
<tr>
<td>

- **Bankroll** - Your current account balance
- **Date** - The current date
</td>
<td>

- **Min_bet** - The minimum possible non-zero bet
- **Max_bet** - The maximum possible bet
</td>
</tr>
</table>

### Opps DataFrame

The betting opportunities contain games played in the coming days. Opportunities are valid up to and including the date of the game. It is therefore possible to bet on a given opportunity **multiple times**. The amounts you have previously bet are available in the `BetH`, `BetA`, and `BetD` columns. Also note that for the game with ID 21, the draw odds are `0.0`, which means that the game cannot end in a draw.

| ID | Season |       Date | HID | AID | OddsH | OddsA |OddsD | BetH | BetA | BetD |
| -: | -----: | ---------: | --: | --: | ----: | ----: | ---: | ---: | ---: | ---: |
| 15 |  1993  | 1993-11-06 |  22 |  41 |  1.93 |  1.91 | 6.25 | 10.0 |  0.0 |  0.0 |
| 16 |  1993  | 1993-11-06 |  12 |  24 |  1.41 |  3.04 | 5.60 |  0.0 |  7.0 |  0.0 |
| 17 |  1993  | 1993-11-06 |   1 |  17 |  1.52 |  2.65 | 6.20 |  0.0 |  0.0 |  0.0 |
| 18 |  1993  | 1993-11-06 |   2 |  42 |  1.49 |  2.74 | 5.30 |  5.0 |  0.0 |  0.0 |
| 19 |  1993  | 1993-11-07 |  13 |  19 |  1.47 |  2.82 | 6.40 |  0.0 |  0.0 |  0.0 |
| 20 |  1993  | 1993-11-07 |  21 |  11 |  1.50 |  2.69 | 0.00 |  0.0 |  0.0 |  0.0 |

<table>
<tr>
<td>

- **ID** - Unique game identifier (table index)
- **Season** - The game's season
- **Date** - Date of the game
- **HID** - Unique identifier of the home team
- **AID** - Unique identifier of the away team

</td>
<td>

- **OddsH** - Odds on a home team win
- **OddsA** - Odds on an away team win
- **OddsD** - Odds on a draw
- **BetH** - Your previous bets on a home team win
- **BetA** - Your previous bets on an away team win
- **BetD** - Your previous bets on a draw

</td>
</tr>
</table>


### Incremental Data
The incremental data contains results and statistics you have not seen yet. The [Games DataFrame](#games-dataframe) contains information about the parameters of games that have been played. Expect the first increment you see to also contain older games (from previous seasons).

#### Games DataFrame

|   ID | Season |       Date | HID | AID | OddsH | OddsA | OddsD |     H |     A |     D | HS | AS | Special | H_PEN | A_PEN | H_MAJ | A_MAJ | H_PPG | A_PPG | H_SHG | A_SHG | H_SV | A_SV | H_PIM | A_PIM | H_SOG | A_SOG | H_BLK_S | A_BLK_S | H_HIT | A_HIT | H_BLK | A_BLK | H_FO | A_FO | H_P1 | A_P1 | H_P2 | A_P2 | H_P3 | A_P3 | H_OT | A_OT | H_SO | A_SO |
| ---: | -----: | ---------: | --: | --: | ----: | ----: | ----: | ----: | ----: | ----: | -: | -: | ------: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ---: | ---: | ----: | ----: | ----: | ----: | ------: | ------: | ----: | ----: | ----: | ----: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 |   1993 | 1993-11-06 |  18 |   9 | 1.35 |  2.35 |   0.0 | False |  True | False |  5 |  6 |     |     3 |     5 |     0 |     0 |     1 |     3 |     0 |     1 |   23 |   41 |     6 |    10 |    46 |    28 |      15 |       2 |    14 |    17 |     1 |    12 |   50 |   28 |    2 |    2 |    1 |    3 |    2 |    0 |      |      |      |      |
| 16 |   1993 | 1993-11-06 |  22 |  20 | 1.41 |  2.14 |   0.0 |  True | False | False |  6 |  3 |         |     5 |     5 |     0 |     0 |     1 |     2 |     0 |     0 |   36 |   27 |    10 |    10 |    33 |    39 |      10 |       6 |    11 |    15 |     6 |     8 |   34 |   33 |    2 |    1 |    1 |    2 |    3 |    0 |      |      |      |      |
| 17 |   1993 | 1993-11-06 |  24 |   9 | 1.37 |  2.28 |   0.0 | False |  True | False |  3 |  4 |         |     9 |    11 |     0 |     0 |     3 |     0 |     0 |     1 |   21 |   34 |    18 |    22 |    37 |    25 |       8 |       4 |    21 |    19 |     4 |     8 |   34 |   29 |    2 |    2 |    0 |    2 |    1 |    0 |      |      |      |      |
| 18 |   1993 | 1993-11-06 |  10 |  18 | 1.43 |  2.09 |   0.0 |  True | False | False |  4 |  2 |         |     2 |     4 |     0 |     0 |     1 |     0 |     0 |     0 |   31 |   15 |     4 |     8 |    19 |    33 |      10 |      11 |     9 |    11 |    11 |    10 |   30 |   41 |    3 |    0 |    0 |    1 |    1 |    1 |      |      |      |      |
| 19 |   1993 | 1993-11-07 |   1 |   5 | 1.44 |  2.08 |   0.0 | False |  True | False |  1 |  2 |     |     7 |     7 |     2 |     1 |     1 |     1 |     0 |     0 |   26 |   33 |    34 |    24 |    34 |    27 |      14 |      11 |    13 |     7 |    10 |    14 |   35 |   31 |    0 |    1 |    0 |    0 |    1 |    0 |      |      |      |      |
| 20 |   1993 | 1993-11-07 |  15 |  19 | 1.84 |  1.55 |   0.0 | False |  True | False |  2 |  6 |         |    15 |    14 |     3 |     3 |     1 |     2 |     0 |     0 |   32 |   26 |    60 |    58 |    28 |    38 |       6 |      10 |    34 |    19 |    10 |     5 |   43 |   30 |    0 |    3 |    1 |    1 |    1 |    2 |      |      |      |      |


<table> <tr> <td>

**ID** - Unique game identifier (table index)

**Season** - The game's season

**Date** - Date of the game

**HID** - Unique identifier of the home team

**AID** - Unique identifier of the away team

**OddsH** - Odds on a home team win

**OddsA** - Odds on an away team win

**OddsD** - Odds on a draw

**H** - Flag indicating whether the home team won

**A** - Flag indicating whether the away team won

**D** - Flag indicating whether the game ended in a draw

**HS** - Goals scored by the home team

**AS** - Goals scored by the away team

**Special** - Special game flag (overtime (OT), penalty shootout (PS), forfeit (DW))

</td> <td>

**H_PEN** - Number of minor penalties of the home team

**A_PEN** - Number of minor penalties of the away team

**H_MAJ** - Number of major penalties of the home team

**A_MAJ** - Number of major penalties of the away team

**H_PPG** - Power-play goals scored by the home team

**A_PPG** - Power-play goals scored by the away team

**H_SHG** - Short-handed goals scored by the home team

**A_SHG** - Short-handed goals scored by the away team

**H_SV** - Saves by the home team's goaltender

**A_SV** - Saves by the away team's goaltender

**H_PIM** - Total penalty minutes of the home team

**A_PIM** - Total penalty minutes of the away team

**H_SOG** - Shots on goal by the home team

**A_SOG** - Shots on goal by the away team

**H_BLK_S** - Blocked shots of the home team

**A_BLK_S** - Blocked shots of the away team



</td> <td>

**H_HIT** - Hits (body checks) by the home team

**A_HIT** - Hits (body checks) by the away team

**H_BLK** - Blocks by the home team

**A_BLK** - Blocks by the away team

**H_FO** - Faceoffs won by the home team

**A_FO** - Faceoffs won by the away team

**H_P1** - Home team goals in the 1st period

**A_P1** - Away team goals in the 1st period

**H_P2** - Home team goals in the 2nd period

**A_P2** - Away team goals in the 2nd period

**H_P3** - Home team goals in the 3rd period

**A_P3** - Away team goals in the 3rd period

**H_OT** - Home team goals in overtime

**A_OT** - Away team goals in overtime

**H_SO** - Home team goals in the shootout

**A_SO** - Away team goals in the shootout

</td> </tr> </table>

### Bets DataFrame

We expect this DataFrame as your response from the `place_bets` method. Even if you don't want to place any bets, you must return a `DataFrame` (at least an empty one).

| ID | BetH | BetA | BetD |
| -: | ---: | ---: | ---: |
| 15 | 10.0 |  0.0 |  0.0 |
| 16 |  0.0 |  7.0 |  0.0 |
| 17 |  0.0 |  0.0 |  0.0 |
| 18 |  5.0 |  0.0 |  0.0 |
| 19 |  0.0 |  0.0 |  0.0 |
| 20 |  0.0 |  0.0 |  0.0 |


- **ID** - Index of the corresponding game from the [Opps DataFrame](#opps-dataframe) (table index)
- **BetH** - Your bets on a home team win
- **BetA** - Your bets on an away team win
- **BetD** - Your bets on a draw

## Technical Details

- All data passed between your model and the bookmaker is stored in a `pandas.DataFrame`.
- Communication between you and the bookmaker happens over stdin/stdout. You don't need to worry about anything, we have handled serialization for you, but **you should avoid using stdout in your submitted solution**.
- **If you send the bookmaker bets on opportunities other than the ones you received, they will be ignored.**
- **If you don't have enough funds for your bets, they will be ignored.**
- **Bets > max_bet and bets < min_bet will be ignored.**

## Common Problems

- The most common error in the upload system is that your code, inside the evaluation loop on hyperion, gets into a state it is not prepared for.
  - For example "RTE: KeyError" - check that your code is ready for an empty opportunities DataFrame, newly appearing team IDs, etc.

Send all questions, problems, and suggestions to qqh@fel.cvut.cz
