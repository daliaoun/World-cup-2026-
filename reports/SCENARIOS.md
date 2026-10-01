# Spain vs Argentina: Every Scenario

**World Cup Final — 19 July 2026, East Rutherford**

Pre-match: **Spain 61.8% — Argentina 38.2%** (90% interval 44.8%–75.4%).
Expected goals: Spain 1.15 – 0.79 Argentina.

Everything below is computed exactly from the in-play model — no simulation noise. Read it
as a companion to the match: find the scoreline and the minute, get the number.

> **One caveat that applies throughout.** These are conditional probabilities *under the
> model*. "If Argentina score first at 60', they win 85%" means: among all the futures
> where that happens, they win 85% of them. Conditioning on a goal also conditions on
> everything that goal implies about how the match is actually going — which the model
> cannot see.

---

## 1. Three ways to win a final

| | inside 90 min | in extra time | on penalties | **total** |
|---|---|---|---|---|
| **Spain** | 43.4% | 9.1% | 9.3% | **61.8%** |
| **Argentina** | 24.8% | 5.8% | 7.6% | **38.2%** |

The match reaches extra time **31.8%** of the time and penalties **17.0%**. In a shootout
Spain are 54.9% — their Elo edge is worth about five points of a coin flip, no more.

**The underdog's route runs through the clock.** 35% of Argentina's title chance requires
the match to go past 90 minutes, against 30% for Spain. Argentina do not need to be
better tonight; they need it to stay close.

## 2. The first goal

| | probability |
|---|---|
| Spain score first | 50.9% |
| Argentina score first | 35.3% |
| No goal at all in 90 minutes | 13.8% |

Median arrival: **minute 32**.

| deadlock broken by | |
|---|---|
| 15' | 21.8% |
| 30' | 40.9% |
| half-time | 58.2% |
| 60' | 69.8% |
| 75' | 78.1% |
| 90' | 86.2% |

## 3. Who wins after the opening goal

The title chance of whichever team scores first, by the minute they score it:

| minute | Argentina score → Argentina | Spain score → Spain |
|---|---|---|
| 10' | 67.5% | 84.5% |
| 20' | 70.4% | 85.7% |
| 30' | 73.2% | 86.9% |
| 45' | 78.3% | 89.2% |
| 60' | **85.1%** | 92.5% |
| 70' | 89.6% | 94.8% |
| 80' | 94.6% | 97.3% |
| 85' | 97.2% | 98.6% |

Note the asymmetry. An early Argentina lead is worth much less than an early Spain lead —
at 10 minutes, 67.5% against 84.5% — because Spain have more time and more attacking
quality to claw it back. By 80 minutes the gap has nearly closed: a lead is a lead when
there's no time left.

## 4. What a goal is worth

Starting from level, how much the next goal swings its scorer's title chance:

| minute | Spain level | Spain score | Argentina score | **Spain swing** | **Argentina swing** |
|---|---|---|---|---|---|
| 1' | 61.7% | 83.5% | 35.2% | +21.8 | +26.5 |
| 15' | 60.8% | 85.2% | 30.8% | +24.4 | +30.0 |
| 30' | 60.1% | 86.9% | 26.8% | +26.8 | +33.4 |
| 45' | 59.4% | 89.2% | 21.7% | +29.8 | +37.7 |
| 60' | 58.7% | 92.5% | 14.9% | +33.9 | +43.8 |
| 70' | 58.3% | 94.8% | 10.4% | +36.5 | +47.9 |
| 80' | 58.0% | 97.3% | 5.4% | +39.3 | +52.6 |
| 85' | 57.8% | 98.6% | 2.8% | +40.7 | +55.0 |
| 89' | 57.7% | 99.7% | 0.6% | **+42.0** | **+57.2** |

**An Argentina goal is worth more than a Spain goal at every single minute** — 26.5 vs
21.8 points in the first minute, 57.2 vs 42.0 in the 89th. The underdog gains more from
the same event, because it moves them across a bigger gap. The value of the opening goal
roughly doubles between kickoff and the final whistle.

## 5. Two-goal positions

Title chance for the team two goals up (or the team that just pulled one back):

| minute | Spain 2-0 | Spain 2-1 | Argentina 0-2 | Argentina 1-2 |
|---|---|---|---|---|
| 30' | 97.3% | 86.9% | 92.2% | 73.2% |
| 60' | 99.2% | 92.5% | 97.8% | 85.1% |
| 75' | 99.8% | 96.0% | 99.5% | 92.2% |
| 85' | 100.0% | 98.6% | 99.9% | 97.2% |

A two-goal cushion is effectively decisive from the hour mark onward. **The consolation
goal matters far more than it feels like it does**: 2-1 at 60 minutes leaves Spain at
92.5%, not 99.2% — one goal restores about 7 points of hope.

## 6. If nobody scores

| 0-0 at | Spain |
|---|---|
| kickoff | 61.8% |
| 30' | 60.1% |
| 60' | 58.7% |
| 75' | 58.1% |
| 85' | 57.8% |
| 90' | 57.7% |

**Ninety goalless minutes move Spain by four points.** This surprises people, but it's
right: a scoreless match drifts toward extra time and penalties, where Spain's advantage
shrinks to almost nothing (54.9%). Argentina's most reliable path is not to be better —
it is to make the match boring.

## 7. Extra time and penalties

| | Spain | Argentina | still level |
|---|---|---|---|
| **Level entering ET (90')** | wins in ET 28.5% | 18.3% | → penalties 53.3% |
| → title chance | **57.7%** | 42.3% | |
| **Still level at 105'** | wins in ET 17.8% | 11.9% | → penalties 70.3% |
| → title chance | **56.4%** | 43.6% | |

Once extra time starts, it's more likely than not (53.3%) to end in a shootout. Halfway
through, that rises to 70.3%. Each phase the match survives nudges Argentina closer to
even.

## 8. Will Spain go behind?

Spain have not trailed at any point in this tournament — seven matches, one goal
conceded. The model's view of tonight:

| | probability of trailing at some point |
|---|---|
| Spain | **38.7%** |
| Argentina | **55.9%** |

So the streak is more likely than not to survive, but it's far from safe: better than a
one-in-three chance Spain go behind for the first time in the tournament, in the final.

---

## The five numbers to remember

1. **Spain 61.8%**, and honestly somewhere between 45% and 75%.
2. **Argentina need 90 minutes to pass.** 35% of their chance lives in extra time and penalties.
3. **An Argentina goal is always worth more** — up to +57 points in the 89th minute.
4. **A goalless match costs Spain only 4 points** across the whole 90.
5. **Spain go behind 38.7% of the time** — the tournament-long clean run is a coin flip away from ending.
