# The Final: Spain vs Argentina

**19 July 2026 — New York/New Jersey Stadium, East Rutherford**

A follow-up to the semi-final forecast, with three additions: goal timing, score-state
dependence, and — the one that matters most — an honest error bar.

---

## The forecast

| | to lift the trophy | 90 minutes |
|---|---|---|
| **Spain** | **61.8%** | W 43.4% · D 31.8% · L 24.8% |
| **Argentina** | **38.2%** | |

Expected goals: **Spain 1.15 – 0.79 Argentina**. Most likely scorelines at 90 minutes:
1-0 (17.2%), 1-1 (14.3%), 0-0 (13.8%), 0-1 (11.5%). Extra time 31.8%, penalties 17.0%.

**And now the number that matters more than any of the above.**

Bootstrapping the whole pipeline — resample the training matches, refit Dixon-Coles,
re-forecast, 200 times — gives a **90% interval of 44.8% to 75.4%**.

That is enormous. The honest translation of "Spain 61.8%" is *somewhere between a coin
flip and a strong favourite, and the data cannot narrow it further*. On 11% of resampled
histories the model would have picked Argentina instead.

Almost nobody publishes this interval, because it makes the forecast look weak. It is
the most truthful thing in the report.

## The market agrees this time

| | model | market (de-vigged) |
|---|---|---|
| Spain | 61.8% | 57.7% – 59.2% |
| Argentina | 38.2% | 40.8% – 42.3% |

*(FanDuel −148/+129, DraftKings −164/+134.)*

For both semi-finals the model contradicted the market and was right twice. Here it
doesn't. A 3–4 point gap is well inside the noise, and inside the bootstrap interval
several times over. **There is no disagreement to be brave about**, which is worth saying
out loud, because two correct calls creates a strong temptation to go looking for a third
contrarian pick that the evidence does not support.

Worth noting what the model cannot see: Spain have conceded once in seven matches and are
37 unbeaten; Argentina have come from behind repeatedly, including two late goals against
England after trailing in the 85th minute. The model sees Spain's superiority as real but
regresses it toward the mean, which is correct on average and will look foolish if Spain
keep the clean sheet.

---

## What's new since the semi-finals

### 1. Goals do not arrive uniformly

The static model implicitly assumes a goal is as likely in minute 3 as in minute 88. It
isn't. Controlling for the strength of the teams on the pitch:

| minute | multiplier on baseline rate |
|---|---|
| 1–15 | 0.75 |
| 16–30 | 0.85 |
| 31–45 | 1.05 |
| 46–60 | 0.98 |
| 61–75 | 0.98 |
| 76–90 | 1.39 |

The last fifteen minutes produce roughly **1.9× the goals of the first fifteen**. (The
31–45 and 76–90 figures are inflated by stoppage-time goals being coded at minute 45 and
90.)

This has a useful side effect. The semi-final model needed a hand-estimated extra-time
damping factor. Now extra time falls out of the same curve — it is simply played at the
intensity of the closing stage — and the two independent estimates agree: **1.09**
measured directly on extra-time matches, **1.19** extrapolated from the within-match
hazard. The fudge factor became a prediction, and it checked out.

### 2. The scoreline changes how teams play — and the raw data lies about how

Measured naively, teams that are two goals ahead appear to score **more**:

| state | goal rate vs level |
|---|---|
| 2+ behind | 2.07× |
| 1 behind | 1.19× |
| level | 1.00× |
| 1 ahead | 0.78× |
| 2+ ahead | 0.68× |

Read carelessly, that says leading teams keep piling it on. Putting it in a model would
be a disaster — scoring would become self-reinforcing and the simulator would spit out
absurd blowouts.

It is an artefact. The teams who are two goals up are mostly **good teams beating bad
ones**. Once every minute of exposure is offset by the two teams' own Dixon-Coles rates,
the effect reverses:

| state | multiplier on the team's own baseline |
|---|---|
| 2+ behind | 1.085 |
| 1 behind | 1.068 |
| level | 1.019 |
| **1 ahead** | **0.883** |
| 2+ ahead | 1.027 |

**A team protecting a one-goal lead scores at 88% of its own normal rate.** Trailing
teams push. That is the football everybody already believes in — but the uncontrolled
number says the exact opposite, and only the control tells you which story is real.

This is the same failure mode as the extra-time bug in the last report: a quantity
measured on a sample that was selected by the very thing being measured.

### 3. A hypothesis that turned out to be wrong

The state-dependent model was built without Dixon-Coles' tau correction, on a specific
hunch: tau was invented in 1997 as an empirical patch for independent Poissons mispricing
0-0, 1-0, 0-1 and 1-1, and score-state dependence looked like the physical mechanism
tau was quietly approximating. If so, modelling the mechanism should reproduce the patch.

It does not.

| scoreline | independent | DC + tau | in-play state model |
|---|---|---|---|
| 0-0 | 14.21% | **15.47%** | **13.69%** |
| 1-0 | 15.92% | **14.66%** | **16.63%** |
| 0-1 | 11.81% | **10.55%** | **11.99%** |
| 1-1 | 13.23% | **14.49%** | **14.44%** |

Tau pushes 0-0 up and 1-0 down. State-dependence does close to the opposite. Across all
100 cells of the score matrix, the correlation between the two corrections is **−0.045** —
essentially zero. They agree on one thing only: both raise the probability of a draw
(0.309 → 0.334 for tau, → 0.320 for state dependence), which is why both fix the same
visible symptom.

And in a walk-forward backtest they are indistinguishable:

| model | RPS | log loss | accuracy |
|---|---|---|---|
| Dixon-Coles (static, with tau) | 0.1864 | 0.9663 | 55.5% |
| In-play (state-dependent, no tau) | **0.1863** | 0.9663 | 55.1% |

Difference: +0.0000, 95% CI [−0.0001, +0.0001].

So: two mechanistically unrelated corrections, both fixing the draw deficit, and **710
matches cannot tell them apart.** The clean story would have been "the 1997 patch was
secretly modelling score effects." The data says no. Reporting the tie is the point —
the in-play model earns its place by handling extra time natively and producing live win
probabilities, not by being more accurate.

### 4. Live win probability

Because the model now tracks the match minute by minute, it can answer questions the
static model structurally cannot.

**Spain's chance of lifting the trophy, given the state of the match:**

| score | kickoff | 30' | 60' | 75' | 90' |
|---|---|---|---|---|---|
| 2-0 | 95.0 | 97.3 | 99.2 | 99.8 | 100 |
| 1-0 | 83.4 | 86.9 | 92.5 | 96.0 | 100 |
| **0-0 / 1-1** | **61.8** | 60.1 | 58.7 | 58.1 | 57.7 |
| 0-1 | 35.4 | 26.8 | 14.9 | 7.8 | 0 |
| 0-2 | 14.9 | 7.8 | 2.2 | 0.5 | 0 |

Two things worth reading off this. A level match barely moves Spain's number all night
(61.8% → 57.7%) — because level at 90 means extra time and penalties, where Spain's edge
is small. And an Argentina goal is worth far more than a Spain goal: 0-1 at the hour puts
Spain at 14.9%, while 1-0 at the hour puts them at 92.5%.

---

## What would still make this better

- **Player-level data.** Still the biggest gap. Messi is a rating adjustment this model
  cannot make.
- **Historical market odds.** Without them "we beat the market twice" is an anecdote
  about two matches, not a measured edge.
- **The state effect is estimated globally.** Argentina's repeated comebacks suggest
  team-specific score effects exist; 4,841 matches is not enough to estimate them per team.
- **`2+ ahead` at 1.027 doesn't fit the pattern** and is probably small-sample noise.

---

## Bottom line

**Spain 61.8%, Argentina 38.2% — with a 90% interval from 44.8% to 75.4%.**

The market says 58%. We agree, and this time there is nothing brave to say.

The interesting results this round were both negative: a plausible hypothesis about tau
was falsified by a −0.045 correlation, and a sophisticated in-play model tied the model
it was built to beat. It shipped anyway, because it does things the old one couldn't —
just not the thing it was supposed to.
