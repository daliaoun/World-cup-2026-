# Forecasting the 2026 World Cup Semi-Finals

**France vs Spain** — 14 July 2026, AT&T Stadium, Arlington
**England vs Argentina** — 15 July 2026, Mercedes-Benz Stadium, Atlanta

A predictive model built from 49,509 international matches (1872–2026), validated by walk-forward
backtest, and simulated 50,000 times per fixture through 90 minutes, extra time and penalties.

---

## The forecast

| | **to reach the final** | 90-minute result | most likely score |
|---|---|---|---|
| **France** | **42.6%** | W 27.2% · D 32.7% · L 40.1% | — |
| **Spain** | **57.4%** | | 1-1 (14.6%), 0-0 (13.7%), 0-1 (13.3%) |
| | *xG 0.91 – 1.17* · extra time 32.7% · penalties 17.7% | | |
| **England** | **44.4%** | W 28.5% · D 33.5% · L 38.0% | |
| **Argentina** | **55.6%** | | 1-1 (14.7%), 0-0 (14.6%), 0-1 (13.4%) |
| | *xG 0.92 – 1.10* · extra time 33.5% · penalties 18.3% | | |

Two tight matches. Neither is close to a coin flip's opposite — nobody here is a 70% favourite —
and roughly **one in three** ends level after 90 minutes.

**Both picks are against the betting market.** That is the most important sentence in this report
and it is dealt with honestly in §5.

---

## 1. The data problem nobody mentions

The standard dataset (`martj42/international_results`, CC0, 49,509 matches) records **final scores
including extra time**. Fit a goals model on those and every knockout match silently poisons the
estimates: a 1-1 that became 2-1 in the 109th minute enters the likelihood as a 2-1, and the model
learns that knockout teams score more than they do.

The per-goal minute data lets us undo this. The dataset codes regulation stoppage time as minute ≤ 90,
so any goal after minute 90 is an extra-time goal. I verified the convention before trusting it: of 97
matches independently known to have gone to extra time (they reached a shootout, or had a goal after
105'), **91 are level at 90'** under this rule. The six exceptions are all two-legged aggregate ties,
where the rule still correctly identifies the extra-time goals.

**181 matches were corrected.** The effect on the semi-finalists is not cosmetic:

| | as recorded | true 90-minute record |
|---|---|---|
| France | 16-2 | 16-2 *(no ET)* |
| Spain | 11-1 | 11-1 *(no ET)* |
| England | 13-6 | **12-6** |
| Argentina | 17-6 | **13-5** |

Argentina's tournament looks considerably less imperious once the extra-time goals are stripped out.
The model penalises them for this — and *still* makes them favourites, which is worth holding onto
when we get to §5.

## 2. Three models, one referee

The brief mentioned XGBoost, so I built it — but I let a backtest decide rather than assuming.

Every model is refit **before every match date, on data strictly prior to it**. The evaluation set is
710 elite-tournament matches (World Cup, Euro, Copa América) since 2010.

The metric is **RPS (Ranked Probability Score)**, not accuracy and not log loss. Football outcomes are
*ordered* — home, draw, away — and predicting a draw when the away team wins is a smaller error than
predicting a home win. RPS is the only common metric that knows this; log loss and Brier are
ordering-blind. This is the standard finding in the forecasting literature (Constantinou & Fenton, 2012).

| model | RPS ↓ | log loss | accuracy |
|---|---|---|---|
| Uniform (⅓ each) | 0.2299 | 1.0986 | 40.4% |
| Base rates | 0.2271 | 1.0873 | 40.4% |
| Elo | 0.1911 | 1.0086 | 54.1% |
| **XGBoost** (Poisson) | 0.1891 | 0.9756 | 53.4% |
| **Dixon-Coles** (time-weighted) | **0.1864** | **0.9663** | **55.5%** |

**The 1997 statistical model beats the gradient-boosted machine.** XGBoost is real and well-built — 28
engineered features, Poisson objective, symmetric long-format training, its own low-score correction —
and it comfortably beats Elo. It still loses to Dixon-Coles.

I also tried blending them. A weighted average of the two, with the weight chosen on pre-2021 matches
and scored on everything after, produces an RPS of **0.1778 — identical to Dixon-Coles alone**, to four
decimal places. A paired bootstrap puts the ensemble's edge at +0.0000 [-0.0013, +0.0013], a 52.4%
chance of being better than nothing at all. **The ensemble was dropped.** Complexity that doesn't pay
its way should not ship.

Dixon-Coles' tuned hyperparameters: time-decay ξ = 0.0015/day (**form half-life ≈ 15 months**) and
ridge shrinkage λ₂ = 0.5.

## 3. Does it work on *this* tournament?

The backtest tunes on history, so it can flatter. The sharper test: refit before each of the **100
WC2026 matches already played** and score the predictions it would have made live.

| | RPS | accuracy |
|---|---|---|
| Elo | 0.1634 | 60.0% |
| **Dixon-Coles** | **0.1531** | **66.0%** |
| — knockout rounds only | 0.1552 | 67.9% |

It does **better** on live 2026 data than on its own tuning set (0.1864). The model is not degrading on
contact with reality. Calibration is close to the diagonal across the probability range.

## 4. The knockout machinery

A 1X2 probability is useless for a semi-final: "draw" is not an outcome. Somebody walks off having won.
So the simulator runs the full cascade, and both extra stages are **estimated, not assumed**.

**Extra time.** My first estimate said teams score at 2.27× the regulation rate in extra time — absurd
on its face. The bug was mine, and it is instructive: I had defined "went to extra time" as *a goal was
scored after minute 90*, then measured the extra-time scoring rate on that sample. Every match in it had
scored in extra time by construction. I was surveying lottery winners to find out how often people win
the lottery.

The missing population — matches that went to extra time and *stayed level* — sits in `shootouts.csv`.
Adding them back (295 matches, of which 153 reached penalties) gives a damping factor of **1.087**:
per minute, extra time is played at essentially the same intensity as regulation. Extra time only
*feels* cagey because it is short.

**Validation:** the simulator says 53.9% of extra-time matches go to penalties. Observed: **51.9%**.
Inside sampling noise.

**Penalties.** Not a coin flip, but nearly. Across 681 historical shootouts the Elo gap is a genuine
predictor (z = 2.89) — a 100-point edge is worth about +4pp. But my first fit also showed a 52.9% edge
to the "home" team, which would have quietly handed France and England a free bonus at a *neutral*
venue. Giving home advantage its own term collapsed that intercept to insignificance (z = 0.62): it was
home advantage wearing a disguise. **At a neutral venue, two equal teams are 50/50.** Who shoots first
is a coin toss and integrates out of a forecast.

## 5. The model disagrees with the market. On both matches. In both directions.

| | model | market (de-vigged) | gap |
|---|---|---|---|
| France | 42.6% | **57.8%** | −15pp |
| Spain | **57.4%** | 42.2% | |
| England | 44.4% | **54.7%** | −10pp |
| Argentina | **55.6%** | 45.3% | |

The market is the hardest benchmark in sport and the standard academic yardstick. A model that reverses
it on *both* matches should be treated as guilty until proven innocent.

The obvious suspect is the model's structural blind spot: **it sees scorelines, not teamsheets.** It
cannot know who is injured. So rather than hand-tune the model toward the market — which would be
fitting to the answer — I asked how large the blind spot would have to be to close the gap.

**France vs Spain.** Spain are missing **Nico Williams and Yeremy Pino**, both wingers. France are
essentially at full strength (Mbappé's ankle knock cleared). To reach the market's number, Spain's
attack would have to be degraded to **65% of normal** — losing two wingers would have to cost them 35%
of their attacking output. A realistic haircut of 5–15% moves the model to roughly Spain 53–55%. So the
injuries explain **part** of this gap, and the honest reading of SF1 is *a coin flip, leaning Spain*.

**England vs Argentina.** Here the same exercise fails completely. To reach the market's number,
Argentina's attack would need to be **30% worse** — but Argentina came out of the Switzerland tie with a
**clean bill of health**, Messi is fit, and it is *England* who lose a defender to suspension. Team news
does not explain this gap; if anything it points the other way. Independent xG supports the model too:
across the knockout rounds Argentina have generated **6.3 non-penalty xG to opponents' 1.8**, against
England's 4.0 to 2.5.

What might the market know that the model doesn't? One credible read, offered by analysts covering the
tie: Argentina have never replaced Ángel Di María on the wings, and their lack of width was exposed
against Egypt and Switzerland — a weakness a goals-only model cannot represent. The other credible read
is simply that England and France carry enormous betting publics at US-facing books, and lines get
shaded toward the popular side. **I cannot distinguish these, and I am not going to pretend otherwise.**

## 6. What I'd fix with more time

- **No historical market odds.** This is the real gap. The market is the benchmark that matters and I
  could only compare against it for these two fixtures, not backtest against it. Without that, "we beat
  the market" is a claim I have not earned and have not made.
- **No player-level data.** The single biggest limitation, and the direct cause of §5. Availability-
  adjusted team ratings (or squad-market-value features) would close it.
- **No xG.** Shot-quality data would sharpen the goal rates; StatsBomb's open data covers WC2022 and
  Euro 2024 but not this tournament.
- The Elo draw model uses a fixed mapping rather than a fitted one.

---

## Bottom line

**Spain to reach the final (57%), and Argentina to reach the final (56%)** — with the caveat, stated
plainly, that the market disagrees with both, and that on France–Spain the injury news is on the
market's side.

The most defensible claim in this report is not the forecast. It is that a well-built XGBoost model,
given every advantage, lost to a bivariate Poisson model published in 1997 — and that an ensemble of
the two added exactly nothing. The interesting result was the negative one.

---

### Reproducing

```
python src/data.py        # ingest + extra-time correction
python src/elo.py         # Elo ratings
python src/features.py    # 28 leak-free features
python src/backtest.py    # walk-forward evaluation
python src/forecast.py    # the two semi-final forecasts
python src/figures.py     # figures
```

Every model is refit on data strictly before each match. `evaluate.save_preds`/`load_preds` assert
that a prediction matrix is aligned to the eval frame it is scored against — after an unstable
`sort_values` silently scrambled 256 of 710 rows and made the ensemble look like it was working.
