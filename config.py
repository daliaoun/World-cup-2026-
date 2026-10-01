"""Central configuration for the WC2026 semifinal forecasting pipeline."""
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
for _p in (RAW, PROC, OUT, FIG):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Data sources
# ----------------------------------------------------------------------------
SOURCES = {
    # Primary backbone: every men's full international since 1872 (CC0).
    "international_results": {
        "base": "https://raw.githubusercontent.com/martj42/international_results/master",
        "files": ["results.csv", "shootouts.csv", "goalscorers.csv", "former_names.csv"],
    },
    # Event data (xG) for national teams. Free, GitHub-hosted.
    "statsbomb": {
        "base": "https://raw.githubusercontent.com/statsbomb/open-data/master/data",
        # (competition_id, season_id, label)
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
            (55, 282, "UEFA Euro 2024"),
            (223, 282, "Copa America 2024"),
        ],
    },
    # Requires a free API key; used as an optional cross-check on fixtures/results.
    "football_data_org": {
        "base": "https://api.football-data.org/v4",
        "env_key": "FOOTBALL_DATA_API_KEY",
    },
}

# ----------------------------------------------------------------------------
# Match-importance weights.
# Rationale: friendlies are experimental (rotation, unlimited subs) and carry
# less information about true strength. Mirrors the FIFA/Elo weighting scheme.
# ----------------------------------------------------------------------------
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "Copa América": 0.85,
    "UEFA Euro": 0.85,
    "African Cup of Nations": 0.85,
    "AFC Asian Cup": 0.85,
    "Gold Cup": 0.80,
    "Confederations Cup": 0.80,
    "UEFA Nations League": 0.75,
    "CONCACAF Nations League": 0.70,
    "FIFA World Cup qualification": 0.70,
    "UEFA Euro qualification": 0.65,
    "African Cup of Nations qualification": 0.65,
    "AFC Asian Cup qualification": 0.65,
    "Copa América qualification": 0.65,
    "Gold Cup qualification": 0.60,
    "Friendly": 0.35,
}
DEFAULT_TOURNAMENT_WEIGHT = 0.55

# Elo K-factors (World Football Elo Ratings convention)
ELO_K = {
    "FIFA World Cup": 60,
    "Copa América": 50, "UEFA Euro": 50, "African Cup of Nations": 50,
    "AFC Asian Cup": 50, "Gold Cup": 50, "Confederations Cup": 50,
    "UEFA Nations League": 40, "CONCACAF Nations League": 40,
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 40, "African Cup of Nations qualification": 40,
    "AFC Asian Cup qualification": 40, "Copa América qualification": 40,
    "Friendly": 20,
}
ELO_K_DEFAULT = 30
ELO_START = 1500.0
ELO_HOME_ADV = 65.0   # rating points; ~ standard for international football

# ----------------------------------------------------------------------------
# Dixon-Coles
# ----------------------------------------------------------------------------
DC_TRAIN_YEARS = 8          # lookback window for fitting
DC_XI_GRID = [0.0000, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0045]  # per-day decay
DC_XI = 0.0015              # default; overwritten by tuning
DC_MIN_MATCHES = 8          # a team needs this many matches in-window to get own params

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
N_SIMS = 50_000             # the "10,000 simulations" idea, with a wider net
ET_MINUTES = 30
ET_DAMPING = None           # estimated from data in src/simulate.py
SEED = 20260714

# ----------------------------------------------------------------------------
# The two fixtures we are forecasting
# ----------------------------------------------------------------------------
FIXTURES = [
    dict(date="2026-07-14", home="France", away="Spain",
         venue="AT&T Stadium, Arlington TX", neutral=True, knockout=True,
         label="Semi-final 1"),
    dict(date="2026-07-15", home="England", away="Argentina",
         venue="Mercedes-Benz Stadium, Atlanta GA", neutral=True, knockout=True,
         label="Semi-final 2"),
]

# Market prices (DraftKings via ESPN, 2026-07-12). Two-way "to advance" market.
# Used ONLY as an out-of-sample benchmark for the model, per standard practice
# in the forecasting literature. Not betting advice.
MARKET_ODDS = {
    ("France", "Spain"):      {"France": -155, "Spain": +125},
    ("England", "Argentina"): {"England": -135, "Argentina": +110},
}
