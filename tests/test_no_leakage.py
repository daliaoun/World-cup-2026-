"""
The invariant that matters most in any forecasting model: it must not see the future.

Dixon-Coles.fit filters to matches strictly before `as_of`. This test proves it, by
fitting once, then appending an absurd match dated AFTER the cutoff and fitting again.
If the future leaked in, the ratings would move. They must not.
"""
import numpy as np
import pandas as pd

from src.dixon_coles import DixonColes


def _toy_history():
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    rows = []
    day = pd.Timestamp("2020-01-01")
    for k in range(40):
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append(dict(
            date=day + pd.Timedelta(days=7 * k),
            home_team=h, away_team=a,
            hs=float(rng.poisson(1.4)), as_=float(rng.poisson(1.1)),
            home_adv=0.0, weight_imp=1.0,
        ))
    return pd.DataFrame(rows)


def test_future_matches_do_not_leak_into_the_fit():
    df = _toy_history()
    cutoff = pd.Timestamp("2020-09-01")

    base = DixonColes(xi=0.0015, lam2=0.5).fit(df, as_of=cutoff)
    r_base = base.rates("A", "B", home_adv=0.0)

    # a ludicrous result AFTER the cutoff: if it leaked, A's attack would explode
    future = pd.DataFrame([dict(
        date=cutoff + pd.Timedelta(days=5),
        home_team="A", away_team="B", hs=15.0, as_=0.0,
        home_adv=0.0, weight_imp=1.0,
    )])
    polluted = pd.concat([df, future], ignore_index=True)
    after = DixonColes(xi=0.0015, lam2=0.5).fit(polluted, as_of=cutoff)
    r_after = after.rates("A", "B", home_adv=0.0)

    np.testing.assert_allclose(r_after, r_base, atol=1e-9)
