"""
================================================================================
MicroTrack Digital Twin -- REVISION SCRIPT for IEEE DG 2026
Astana IT University / L.N. Gumilyov ENU

Covers every reviewer request:
  * comfort-band reward (Eq. 1); the old single-setpoint reward survives only
    as one row of the sensitivity table
  * two-period evaluation: in-period P1 + genuinely held-out P2
  * multi-seed statistics (mean +/- SD) with a paired Wilcoxon signed-rank test
  * sensitivity analysis: comfort band / energy weight / grounding ratio
  * one-step dynamics calibration + MAE  -> Limitations item (iii)
  * energy accounting in kWh/month and the break-even condition (Eq. 2)
  * Q-table footprint measured on TRAINING-updated states only
  * ready-to-paste LaTeX table bodies in ./results/

USAGE
  python src/microtrack_revision.py
  python src/microtrack_revision.py --help    for all input and output paths

  Paths default to the repository layout (data/*.zip -> results/) and resolve
  relative to the repository root, so the script runs from any directory.

RUNTIME
  Full config (10 seeds, stride 1, 8 episodes): ~25-45 min on one CPU core.
  Pass --fast for a structurally identical smoke test in ~2-4 min. A fast run
  does NOT reproduce the published numbers.

FIGURE
  This script writes no figure. Run make_fig.py afterwards to produce the
  compact single-column generalization.png used in the paper.
================================================================================
"""

# ==============================================================================
# CELL 1 -- imports and configuration
# ==============================================================================
import os, glob, json, time, zipfile, warnings, argparse
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("scipy not available -- Wilcoxon test will be skipped")

# ------------------------------------------------------------------ data paths
# Each period lives in its OWN archive. The archive decides which period a
# record belongs to; the dates below are only a sanity bound.
ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(
    description="MicroTrack Digital Twin revision run for IEEE DG 2026.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--p1-zip", type=Path, default=ROOT / "data" / "box_data.zip",
                    help="Period 1 (in-period) telemetry archive")
parser.add_argument("--p2-zip", type=Path, default=ROOT / "data" / "box_data_p2.zip",
                    help="Period 2 (held-out) telemetry archive")
parser.add_argument("--extract-root", type=Path, default=ROOT / "data" / "extracted",
                    help="scratch directory the archives are unpacked into")
parser.add_argument("--out-dir", type=Path, default=ROOT / "results",
                    help="destination for raw metrics, LaTeX bodies and paper_numbers.json")
parser.add_argument("--fast", action="store_true",
                    help="structurally identical smoke test; does NOT reproduce the published numbers")
args = parser.parse_args()

ZIP_PATHS = {"P1": str(args.p1_zip), "P2": str(args.p2_zip)}
EXTRACT_ROOT = str(args.extract_root)
OUT_DIR = str(args.out_dir)

# ---------------------------------------------------------------- period split
# Period 1 (in-period): training + in-period evaluation, chronological 80/20.
# Period 2 (held-out) : never seen during training.
P1_START, P1_END = "2025-05-29", "2026-04-28"
P2_START, P2_END = "2026-04-29", "2026-06-15"

# Drop records falling outside their period's declared window. Keep this True:
# the entire paper rests on P2 being genuinely unseen.
ENFORCE_DATE_BOUNDS = True

# --------------------------------------------------------------- run size knobs
FAST_MODE = args.fast

if FAST_MODE:
    N_SEEDS, SENS_SEEDS, EPISODES, TRAIN_STRIDE, EVAL_STEPS = 3, 2, 3, 20, 20000
else:
    N_SEEDS, SENS_SEEDS, EPISODES, TRAIN_STRIDE, EVAL_STEPS = 10, 5, 8, 1, 60000

SEEDS = list(range(42, 42 + N_SEEDS))          # 42 stays the first seed
SENS_SEED_LIST = SEEDS[:SENS_SEEDS]

# ------------------------------------------------------------ paper parameters
T_TARGET, RH_TARGET, CO2_MAX = 24.0, 55.0, 800.0
T_EXC_LOW, T_EXC_HIGH   = 18.0, 28.0           # excursion band
T_SAFE_LOW, T_SAFE_HIGH = 16.0, 30.0           # safety-penalty band
SAFETY_PENALTY          = -25.0

COMFORT_BAND  = (22.0, 27.0)   # Eq. 1 default
ENERGY_WEIGHT = 0.04           # Eq. 1 default
BLEND_EVAL    = 0.70           # 70/30 telemetry grounding at evaluation
BLEND_TRAIN   = 0.60           # 60/40 every 50 steps during training
GROUND_EVERY  = 50             # training re-grounding interval

ALPHA, GAMMA       = 0.15, 0.95
EPS_START, EPS_MIN = 1.0, 0.05
EPS_DECAY          = 0.9997

# actuator effects per step: (dT, dRH, dCO2)
EFFECTS = {"fan": (-0.10, -0.6, -18.0),
           "heater": (+0.15, -0.3, 0.0),
           "pump": (0.00, +0.9, 0.0)}

# power draw (W) and month length (h) for energy accounting
P_FAN, P_HEATER, P_PUMP, HOURS_PER_MONTH = 18.0, 120.0, 10.0, 730.0

# threshold-baseline rules
TH_FAN_T, TH_FAN_CO2, TH_HEATER_T = 25.5, 800.0, 20.0
# correlation orchestrator
CORR_RH_RISE, CORR_RH_LAG, CORR_T_GATE = 2.0, 8, 23.5

# state discretisation -> 7 x 7 x 5 x 2 = 490 states
T_BINS   = [0, 19, 21, 23, 25, 27, 29, 100]
RH_BINS  = [0, 25, 35, 45, 55, 65, 75, 100]
CO2_BINS = [0, 450, 600, 750, 900, 5000]
LIGHT_ON = 50.0
N_T, N_RH, N_CO2 = len(T_BINS) - 1, len(RH_BINS) - 1, len(CO2_BINS) - 1
N_STATES = N_T * N_RH * N_CO2 * 2

# 8 binary actuator combinations (fan, heater, pump)
ACTIONS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 1),
           (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 1, 1)]
N_ACTIONS = len(ACTIONS)
ACT_INDEX = {a: i for i, a in enumerate(ACTIONS)}

# mean-reversion targets: 'per_period' calibrates them from each period's own
# telemetry (correct), 'fixed' reproduces the legacy hard-coded 22.0 / 42.0
# constants, which were tuned on Period 1 and therefore biased Period 2.
REVERSION_MODE = "per_period"
FIXED_REVERSION = dict(a_T=0.03, T_ref=22.0, sd_T=0.04,
                       a_RH=0.02, RH_ref=42.0, sd_RH=0.2,
                       drift_CO2=0.8, sd_CO2=5.0)

os.makedirs(OUT_DIR, exist_ok=True)
print(f"State space: {N_STATES} | Actions: {N_ACTIONS} | Seeds: {SEEDS}")
print(f"FAST_MODE={FAST_MODE} | episodes={EPISODES} stride={TRAIN_STRIDE} "
      f"eval_steps={EVAL_STEPS}")


# ==============================================================================
# CELL 2 -- data loading and period split (two separate archives)
# ==============================================================================
def extract_archive(zip_path, dest):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Archive not found: {zip_path}")
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest


def load_data(data_dir):
    files = sorted(glob.glob(f"{data_dir}/**/*.csv", recursive=True))
    if not files:
        raise ValueError(f"No CSV files under {data_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "timestamp" in df.columns and "air_temperature" in df.columns:
                dfs.append(df)
        except Exception:
            pass
    if not dfs:
        raise ValueError(f"No usable CSV files under {data_dir}")
    data = pd.concat(dfs, ignore_index=True)
    n_raw = len(data)

    for col in ["air_temperature", "air_humidity", "co2", "light_level"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        else:
            data[col] = np.nan

    # quality filter exactly as reported in the paper (NaN rows are dropped)
    data = data[data["air_temperature"].between(15, 40)
                & data["air_humidity"].between(10, 95)
                & data["co2"].between(300, 2000)].copy()

    data["light_level"] = data["light_level"].fillna(0.0)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")
    data = data.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    print(f"  {len(files)} CSV files | {n_raw:,} raw -> {len(data):,} "
          f"quality-filtered records")
    print(f"  range {data['timestamp'].min().date()} -> "
          f"{data['timestamp'].max().date()}")
    return data


def slice_period(df, start, end):
    m = (df["timestamp"] >= pd.Timestamp(start)) & \
        (df["timestamp"] <= pd.Timestamp(end) + pd.Timedelta(days=1))
    return df.loc[m].reset_index(drop=True)


def to_arrays(df):
    """pandas -> numpy once, so the simulation loop never touches .iloc"""
    return dict(T=df["air_temperature"].to_numpy(np.float64),
                RH=df["air_humidity"].to_numpy(np.float64),
                CO2=df["co2"].to_numpy(np.float64),
                light=df["light_level"].to_numpy(np.float64),
                ts=df["timestamp"].to_numpy())


periods = {}
for tag, zpath in ZIP_PATHS.items():
    print(f"\n[{tag}] {zpath}")
    dest = extract_archive(zpath, os.path.join(EXTRACT_ROOT, tag))
    periods[tag] = load_data(dest)

p1_df, p2_df = periods["P1"], periods["P2"]

# --- overlap guard ------------------------------------------------------------
# Archives are often re-exported cumulatively, so P1 may silently contain part
# of P2. Any shared timestamp is removed from P1, never from P2.
overlap = np.intersect1d(p1_df["timestamp"].to_numpy(), p2_df["timestamp"].to_numpy())
if len(overlap):
    print(f"\n!! {len(overlap):,} timestamps appear in BOTH archives -- "
          f"removing them from P1 to keep the hold-out clean")
    p1_df = p1_df[~p1_df["timestamp"].isin(overlap)].reset_index(drop=True)

if ENFORCE_DATE_BOUNDS:
    bounds = {"P1": (P1_START, P1_END), "P2": (P2_START, P2_END)}
    trimmed = {}
    for tag, df_ref in [("P1", p1_df), ("P2", p2_df)]:
        lo, hi = bounds[tag]
        kept = slice_period(df_ref, lo, hi)
        dropped = len(df_ref) - len(kept)
        if dropped:
            print(f"!! {tag}: {dropped:,} records outside {lo}..{hi} dropped "
                  f"(archive spanned {df_ref['timestamp'].min().date()} -> "
                  f"{df_ref['timestamp'].max().date()})")
        trimmed[tag] = kept
    p1_df, p2_df = trimmed["P1"], trimmed["P2"]

if len(p1_df) == 0 or len(p2_df) == 0:
    raise ValueError("One of the periods is empty after filtering -- check the "
                     "archives and the P1/P2 date bounds.")

# hard separation check: P1 must end strictly before the hold-out begins
if p1_df["timestamp"].max() >= p2_df["timestamp"].min():
    raise ValueError(
        f"P1 ends at {p1_df['timestamp'].max()} but P2 starts at "
        f"{p2_df['timestamp'].min()} -- the hold-out is contaminated. "
        "Fix the date bounds or the archives before running.")

print(f"\nP1 {p1_df['timestamp'].min().date()} -> {p1_df['timestamp'].max().date()}"
      f"   |   P2 {p2_df['timestamp'].min().date()} -> "
      f"{p2_df['timestamp'].max().date()}")

split = int(len(p1_df) * 0.8)
train_df   = p1_df.iloc[:split].reset_index(drop=True)
p1_eval_df = p1_df.iloc[split:].reset_index(drop=True)

for name, df_ref in [("P1 eval", p1_eval_df), ("P2 held-out", p2_df)]:
    if len(df_ref) < EVAL_STEPS:
        print(f"!! {name} has only {len(df_ref):,} records, fewer than "
              f"EVAL_STEPS={EVAL_STEPS:,} -- the window will be shorter")

p1_eval_win = p1_eval_df.iloc[:EVAL_STEPS].reset_index(drop=True)
p2_eval_win = p2_df.iloc[:EVAL_STEPS].reset_index(drop=True)

TRAIN   = to_arrays(train_df)
EVAL_P1 = to_arrays(p1_eval_win)
EVAL_P2 = to_arrays(p2_eval_win)

print(f"P1 total {len(p1_df):,} | train {len(train_df):,} | "
      f"in-period eval {len(p1_eval_df):,}")
print(f"P2 held-out {len(p2_df):,}")
print(f"Evaluation windows: P1 {len(EVAL_P1['T']):,} steps "
      f"({p1_eval_win['timestamp'].min().date()} -> "
      f"{p1_eval_win['timestamp'].max().date()}) | "
      f"P2 {len(EVAL_P2['T']):,} steps "
      f"({p2_eval_win['timestamp'].min().date()} -> "
      f"{p2_eval_win['timestamp'].max().date()})")


# ==============================================================================
# CELL 3 -- one-step dynamics calibration
#           (replaces the hard-coded reversion constants and produces the
#            model-validation MAE quoted in Limitations item iii)
# ==============================================================================
def calibrate_dynamics(df, label, dt_lo=4.0, dt_hi=8.0):
    """
    Fit observed one-step dynamics  x_{t+1} - x_t = a * (x_ref - x_t) + eps
    on consecutive samples only (logging gaps excluded). Returns coefficients
    plus the one-step prediction MAE.
    """
    t = pd.to_datetime(df["timestamp"]).to_numpy()
    dt = (t[1:] - t[:-1]) / np.timedelta64(1, "s")
    ok = (dt >= dt_lo) & (dt <= dt_hi)

    out = {"n_pairs": int(ok.sum()), "label": label}
    if ok.sum() < 100:
        raise ValueError(f"Too few consecutive samples in {label} "
                         f"({ok.sum()}). Check the sampling cadence.")

    for key, col in [("T", "air_temperature"), ("RH", "air_humidity")]:
        x = df[col].to_numpy(np.float64)
        x0, dx = x[:-1][ok], (x[1:] - x[:-1])[ok]
        c1, c0 = np.polyfit(x0, dx, 1)              # dx = c1*x0 + c0
        a = -c1
        ref = c0 / a if abs(a) > 1e-9 else float(np.mean(x0))
        resid = dx - (c0 + c1 * x0)
        out[f"a_{key}"]   = float(a)
        out[f"{key}_ref"] = float(ref)
        out[f"sd_{key}"]  = float(np.std(resid, ddof=1))
        out[f"mae_{key}"] = float(np.mean(np.abs(resid)))

    c = df["co2"].to_numpy(np.float64)
    dc = (c[1:] - c[:-1])[ok]
    out["drift_CO2"] = float(np.mean(dc))
    out["sd_CO2"]    = float(np.std(dc, ddof=1))
    out["mae_CO2"]   = float(np.mean(np.abs(dc - np.mean(dc))))
    return out


DYN_TRAIN = calibrate_dynamics(train_df, "train")
DYN_P1    = calibrate_dynamics(p1_eval_win, "P1-eval")
DYN_P2    = calibrate_dynamics(p2_eval_win, "P2-eval")

print("\n--- one-step dynamics calibration ---")
for d in (DYN_TRAIN, DYN_P1, DYN_P2):
    print(f"{d['label']:<9} pairs={d['n_pairs']:>7,} | "
          f"a_T={d['a_T']:.4f} T_ref={d['T_ref']:.2f} MAE_T={d['mae_T']:.4f} C | "
          f"a_RH={d['a_RH']:.4f} RH_ref={d['RH_ref']:.2f} MAE_RH={d['mae_RH']:.4f} %")

if REVERSION_MODE == "fixed":
    DYN_TRAIN = DYN_P1 = DYN_P2 = dict(FIXED_REVERSION, label="fixed")
    print("\n!! REVERSION_MODE='fixed': using the legacy P1-tuned constants.")


def window_stats(arrs, label):
    T, RH = arrs["T"], arrs["RH"]
    return dict(window=label, n=int(len(T)),
                T_mean=float(T.mean()), T_sd=float(T.std(ddof=1)),
                RH_mean=float(RH.mean()), RH_sd=float(RH.std(ddof=1)),
                pct_above_fan_th=float(100.0 * np.mean(T > TH_FAN_T)))


WIN = [window_stats(EVAL_P1, "P1 in-period"), window_stats(EVAL_P2, "P2 held-out")]
print("\n--- evaluation-window statistics (Table: windows) ---")
for w in WIN:
    print(f"{w['window']:<13} T={w['T_mean']:.2f}+/-{w['T_sd']:.2f} C | "
          f"RH={w['RH_mean']:.2f}+/-{w['RH_sd']:.2f} % | "
          f"steps above {TH_FAN_T} C: {w['pct_above_fan_th']:.1f}%")


# ==============================================================================
# CELL 4 -- Digital Twin, reward, controllers, Q-learning agent
# ==============================================================================
def discretize(T, RH, CO2, light):
    """bisect on plain lists: much faster than np.searchsorted for scalars."""
    tb = bisect_right(T_BINS, T) - 1
    rb = bisect_right(RH_BINS, RH) - 1
    cb = bisect_right(CO2_BINS, CO2) - 1
    if tb < 0: tb = 0
    elif tb > N_T - 1: tb = N_T - 1
    if rb < 0: rb = 0
    elif rb > N_RH - 1: rb = N_RH - 1
    if cb < 0: cb = 0
    elif cb > N_CO2 - 1: cb = N_CO2 - 1
    return (tb, rb, cb, 1 if light > LIGHT_ON else 0)


def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def apply_action(T, RH, CO2, a_idx, dyn, noise_row):
    """Physics layer: actuator effects + calibrated drift + pre-drawn noise."""
    fan, heater, pump = ACTIONS[a_idx]
    dT = dRH = dCO2 = 0.0
    if fan:
        dT += EFFECTS["fan"][0]; dRH += EFFECTS["fan"][1]; dCO2 += EFFECTS["fan"][2]
    if heater:
        dT += EFFECTS["heater"][0]; dRH += EFFECTS["heater"][1]
    if pump:
        dRH += EFFECTS["pump"][1]

    dT   += dyn["a_T"] * (dyn["T_ref"] - T) + noise_row[0] * dyn["sd_T"]
    dRH  += dyn["a_RH"] * (dyn["RH_ref"] - RH) + noise_row[1] * dyn["sd_RH"]
    dCO2 += dyn["drift_CO2"] + noise_row[2] * dyn["sd_CO2"]

    return (_clip(T + dT, 15.0, 40.0),
            _clip(RH + dRH, 10.0, 95.0),
            _clip(CO2 + dCO2, 300.0, 2000.0))


def reward_fn(T, RH, CO2, a_idx, band=COMFORT_BAND, w_energy=ENERGY_WEIGHT):
    """Eq. (1): comfort-band temperature term, zero inside, quadratic outside."""
    if T > T_SAFE_HIGH or T < T_SAFE_LOW:
        return SAFETY_PENALTY
    t_low, t_high = band
    d_T = t_low - T
    if T - t_high > d_T:
        d_T = T - t_high
    if d_T < 0.0:
        d_T = 0.0
    e_RH = RH - RH_TARGET
    fan, heater, pump = ACTIONS[a_idx]
    e_energy = 1.0 * fan + 1.5 * heater + 0.3 * pump
    co2_pen = CO2 - CO2_MAX
    if co2_pen < 0.0:
        co2_pen = 0.0
    return -(d_T * d_T / 16.0
             + 0.5 * e_RH * e_RH / 400.0
             + 0.3 * co2_pen / 200.0
             + w_energy * e_energy)


def reward_single_setpoint(T, RH, CO2, a_idx, band=None, w_energy=ENERGY_WEIGHT):
    """Legacy reward (penalises every deviation from 24 C).
    Kept ONLY as a sensitivity condition -- it is the pathology the
    comfort band was introduced to avoid."""
    if T > T_SAFE_HIGH or T < T_SAFE_LOW:
        return SAFETY_PENALTY
    fan, heater, pump = ACTIONS[a_idx]
    e_energy = 1.0 * fan + 1.5 * heater + 0.3 * pump
    co2_pen = CO2 - CO2_MAX
    if co2_pen < 0.0:
        co2_pen = 0.0
    return -((T - T_TARGET) ** 2 / 16.0
             + 0.5 * (RH - RH_TARGET) ** 2 / 400.0
             + 0.3 * co2_pen / 200.0
             + w_energy * e_energy)


# ------------------------------------------------------------------ controllers
def threshold_action(T, RH, CO2, light, rh_hist):
    fan = 1 if (T > TH_FAN_T or CO2 > TH_FAN_CO2) else 0
    heater = 1 if T < TH_HEATER_T else 0
    return ACT_INDEX[(fan, heater, 1)], False


def correlation_action(T, RH, CO2, light, rh_hist):
    """Threshold rules + proactive pre-activation on a humidity rise.
    `proactive` is True only when the rule actually changes the action."""
    fan = 1 if (T > TH_FAN_T or CO2 > TH_FAN_CO2) else 0
    heater = 1 if T < TH_HEATER_T else 0
    proactive = False
    if len(rh_hist) >= CORR_RH_LAG:
        if (rh_hist[-1] - rh_hist[-CORR_RH_LAG]) > CORR_RH_RISE and T > CORR_T_GATE:
            if fan == 0:
                proactive = True
            fan = 1
    return ACT_INDEX[(fan, heater, 1)], proactive


def make_q_controller(q_table):
    """
    Frozen tabular policy. States never updated during TRAINING carry no learned
    values, so the controller falls back to the threshold rules -- this is what
    the deployed firmware must do, and it degrades gracefully in unseen regimes.
    """
    def act(T, RH, CO2, light, rh_hist):
        q = q_table.get(discretize(T, RH, CO2, light))
        if q is None:
            return threshold_action(T, RH, CO2, light, rh_hist)[0], False
        return int(np.argmax(q)), False
    return act


# --------------------------------------------------------------------- training
def train_agent(train_arrs, seed, band=COMFORT_BAND, w_energy=ENERGY_WEIGHT,
                reward=reward_fn, dyn=None, episodes=EPISODES,
                stride=TRAIN_STRIDE, verbose=False):
    """Returns (frozen Q-table, set of states actually updated)."""
    rng = np.random.default_rng(seed)
    dyn = dyn if dyn is not None else DYN_TRAIN
    Q = defaultdict(lambda: np.zeros(N_ACTIONS))
    updated = set()                 # states where a Bellman update really happened
    eps = EPS_START

    Tr, RHr, CO2r, Lr = (train_arrs["T"], train_arrs["RH"],
                         train_arrs["CO2"], train_arrs["light"])
    idx = np.arange(0, len(Tr) - 1, stride)

    for ep in range(episodes):
        j0 = int(rng.integers(0, max(1, min(1000, len(idx)))))
        T, RH, CO2 = float(Tr[idx[j0]]), float(RHr[idx[j0]]), float(CO2r[idx[j0]])
        light = float(Lr[idx[j0]])
        noise = rng.standard_normal((len(idx), 3))
        total_r = 0.0

        for k in range(len(idx)):
            i = idx[k]
            if k % GROUND_EVERY == 0:
                T   = BLEND_TRAIN * Tr[i]   + (1 - BLEND_TRAIN) * T
                RH  = BLEND_TRAIN * RHr[i]  + (1 - BLEND_TRAIN) * RH
                CO2 = BLEND_TRAIN * CO2r[i] + (1 - BLEND_TRAIN) * CO2
                light = Lr[i]

            s = discretize(T, RH, CO2, light)
            a = (int(rng.integers(N_ACTIONS)) if rng.random() < eps
                 else int(np.argmax(Q[s])))

            T2, RH2, CO2_2 = apply_action(T, RH, CO2, a, dyn, noise[k])
            r = reward(T2, RH2, CO2_2, a, band, w_energy)
            s2 = discretize(T2, RH2, CO2_2, light)

            Q[s][a] += ALPHA * (r + GAMMA * Q[s2].max() - Q[s][a])
            updated.add(s)

            eps = max(EPS_MIN, eps * EPS_DECAY)
            T, RH, CO2 = T2, RH2, CO2_2
            total_r += r

        if verbose:
            print(f"    ep {ep+1}/{episodes} avg_r={total_r/len(idx):+.4f} "
                  f"eps={eps:.3f} states={len(updated)}")

    q_frozen = {s: np.array(Q[s]) for s in updated}
    return q_frozen, updated


# ------------------------------------------------------------------- evaluation
def evaluate(arrs, control_fn, noise, dyn, blend=BLEND_EVAL):
    """
    All controllers in a given (seed, period) receive the SAME pre-drawn noise
    array, so the comparison is paired and any difference is due to the policy.
    """
    Tr, RHr, CO2r, Lr = arrs["T"], arrs["RH"], arrs["CO2"], arrs["light"]
    n = len(Tr)
    T, RH, CO2 = float(Tr[0]), float(RHr[0]), float(CO2r[0])
    rh_hist = [RH] * 15
    inv = 1.0 - blend

    se_T = se_RH = 0.0
    fan_on = heater_on = pump_on = excursions = proactive_n = 0

    for i in range(n):
        T   = blend * Tr[i]   + inv * T
        RH  = blend * RHr[i]  + inv * RH
        CO2 = blend * CO2r[i] + inv * CO2
        light = Lr[i]

        rh_hist.append(RH)
        if len(rh_hist) > 15:
            rh_hist.pop(0)

        a_idx, proactive = control_fn(T, RH, CO2, light, rh_hist)
        fan, heater, pump = ACTIONS[a_idx]
        fan_on += fan; heater_on += heater; pump_on += pump
        if proactive:
            proactive_n += 1

        T, RH, CO2 = apply_action(T, RH, CO2, a_idx, dyn, noise[i])

        se_T  += (T - T_TARGET) ** 2
        se_RH += (RH - RH_TARGET) ** 2
        if T > T_EXC_HIGH or T < T_EXC_LOW:
            excursions += 1

    f, h, p = fan_on / n, heater_on / n, pump_on / n
    power_W = P_FAN * f + P_HEATER * h + P_PUMP * p
    return dict(rmse_T=float(np.sqrt(se_T / n)),
                rmse_RH=float(np.sqrt(se_RH / n)),
                fan_pct=100 * f, heater_pct=100 * h, pump_pct=100 * p,
                excursion_pct=100 * excursions / n,
                power_W=power_W,
                energy_kwh_month=power_W * HOURS_PER_MONTH / 1000.0,
                proactive_pct=100 * proactive_n / n)


def eval_noise(seed, period_tag, n):
    """Deterministic per (seed, period); shared by all three controllers."""
    return np.random.default_rng((seed, period_tag)).standard_normal((n, 3))


PERIOD_SPECS = [("P1", EVAL_P1, DYN_P1, 1), ("P2", EVAL_P2, DYN_P2, 2)]


# ==============================================================================
# CELL 5 -- multi-seed run  (Table: multiseed + Wilcoxon)
# ==============================================================================
QTABLE_CACHE = {}          # (seed, band, w_energy, reward_name) -> frozen table


def get_qtable(seed, band, w_energy, reward):
    key = (seed, band, w_energy, reward.__name__)
    if key not in QTABLE_CACHE:
        QTABLE_CACHE[key] = train_agent(TRAIN, seed, band=band,
                                        w_energy=w_energy, reward=reward)[0]
    return QTABLE_CACHE[key]


def run_multiseed(seeds=SEEDS, band=COMFORT_BAND, w_energy=ENERGY_WEIGHT,
                  reward=reward_fn, blend=BLEND_EVAL, verbose=True):
    rows, visited_counts, qtable_sizes = [], [], []
    t0 = time.time()

    for si, seed in enumerate(seeds):
        if verbose:
            print(f"\n[seed {seed}]  ({si+1}/{len(seeds)})  training...")
        q_table = get_qtable(seed, band, w_energy, reward)
        visited_counts.append(len(q_table))
        qtable_sizes.append(len(q_table) * N_ACTIONS * 8 / 1024.0)   # KB

        controllers = {"Threshold": threshold_action,
                       "Correlation": correlation_action,
                       "Q-Learning": make_q_controller(q_table)}

        for period, arrs, dyn, tag in PERIOD_SPECS:
            noise = eval_noise(seed, tag, len(arrs["T"]))
            for name, fn in controllers.items():
                m = evaluate(arrs, fn, noise, dyn, blend=blend)
                m.update(seed=seed, period=period, algorithm=name)
                rows.append(m)
            if verbose:
                sub = {r["algorithm"]: r for r in rows[-3:]}
                print(f"    {period}: fan th={sub['Threshold']['fan_pct']:.1f}% "
                      f"ql={sub['Q-Learning']['fan_pct']:.1f}% | "
                      f"E th={sub['Threshold']['energy_kwh_month']:.1f} "
                      f"ql={sub['Q-Learning']['energy_kwh_month']:.1f} kWh/mo | "
                      f"states={len(q_table)}")

    if verbose:
        print(f"\nMulti-seed run finished in {(time.time()-t0)/60:.1f} min")
    df = pd.DataFrame(rows)
    df.attrs["visited"] = visited_counts
    df.attrs["qtable_kb"] = qtable_sizes
    return df


print("\n" + "=" * 78)
print("MULTI-SEED RUN")
print("=" * 78)
ms = run_multiseed()
ms.to_csv(os.path.join(OUT_DIR, "multiseed_raw.csv"), index=False)

REPORT_COLS = ["rmse_T", "rmse_RH", "fan_pct", "heater_pct", "pump_pct",
               "excursion_pct", "energy_kwh_month"]
agg = (ms.groupby(["period", "algorithm"])[REPORT_COLS]
         .agg(["mean", "std"]).round(3))
print("\n--- mean / SD over seeds ---")
print(agg)

stats_summary = {}


def paired(df, period, algo, metric):
    a = df[(df.period == period) & (df.algorithm == algo)] \
        .sort_values("seed")[metric].to_numpy()
    b = df[(df.period == period) & (df.algorithm == "Threshold")] \
        .sort_values("seed")[metric].to_numpy()
    return a, b


print("\n--- paired Q-Learning vs Threshold ---")
for period in ["P1", "P2"]:
    for metric in ["fan_pct", "energy_kwh_month"]:
        a, b = paired(ms, period, "Q-Learning", metric)
        rel = 100.0 * (a - b) / b
        sd = float(np.std(rel, ddof=1)) if len(rel) > 1 else 0.0
        line = (f"{period} {metric:<18} "
                f"QL={a.mean():7.2f}+/-{np.std(a, ddof=1):.2f}  "
                f"TH={b.mean():7.2f}+/-{np.std(b, ddof=1):.2f}  "
                f"rel={rel.mean():+6.2f}% +/- {sd:.2f}")
        if HAVE_SCIPY and len(a) >= 6 and np.any(a != b):
            try:
                _, pv = wilcoxon(a, b)
                line += f" | Wilcoxon p={pv:.4g}"
                stats_summary[f"{period}_{metric}_p"] = float(pv)
            except ValueError:
                pass
        stats_summary[f"{period}_{metric}_rel_mean"] = float(rel.mean())
        stats_summary[f"{period}_{metric}_rel_sd"] = sd
        print(line)

print("\n--- Correlation vs Threshold (the 'indistinguishable' claim) ---")
for period in ["P1", "P2"]:
    a, b = paired(ms, period, "Correlation", "fan_pct")
    if np.all(a == b):
        print(f"{period}: identical on every seed (delta = 0.000 pp)")
        stats_summary[f"{period}_corr_identical"] = True
    elif HAVE_SCIPY:
        try:
            _, pv = wilcoxon(a, b)
            print(f"{period}: delta={np.mean(a-b):+.3f} pp, Wilcoxon p={pv:.4g}")
            stats_summary[f"{period}_corr_p"] = float(pv)
            stats_summary[f"{period}_corr_delta_pp"] = float(np.mean(a - b))
        except ValueError:
            print(f"{period}: delta={np.mean(a-b):+.3f} pp (test not applicable)")

pro = ms[ms.algorithm == "Correlation"].groupby("period")["proactive_pct"].mean()
print(f"\nProactive pre-activation changed the action on "
      f"P1 {pro.get('P1', 0):.3f}% of steps, P2 {pro.get('P2', 0):.3f}%")


# ==============================================================================
# CELL 6 -- sensitivity analysis  (Table: sensitivity)
# ==============================================================================
SENS_GRID = [
    ("Comfort band",    "single 24 C", dict(reward=reward_single_setpoint)),
    ("Comfort band",    "[21,28]",     dict(band=(21.0, 28.0))),
    ("Comfort band",    "[22,27]*",    dict()),
    ("Comfort band",    "[23,26]",     dict(band=(23.0, 26.0))),
    ("Energy weight",   "0.02",        dict(w_energy=0.02)),
    ("Energy weight",   "0.04*",       dict()),
    ("Energy weight",   "0.08",        dict(w_energy=0.08)),
    ("Grounding ratio", "100/0",       dict(blend=1.00)),
    ("Grounding ratio", "80/20",       dict(blend=0.80)),
    ("Grounding ratio", "70/30*",      dict()),
    ("Grounding ratio", "60/40",       dict(blend=0.60)),
]


def run_sensitivity(grid=SENS_GRID, seeds=SENS_SEED_LIST):
    """Default-configuration agents are reused from QTABLE_CACHE, so only the
    genuinely new reward configurations are retrained."""
    out = []
    t0 = time.time()
    for gi, (group, label, kw) in enumerate(grid):
        band   = kw.get("band", COMFORT_BAND)
        w      = kw.get("w_energy", ENERGY_WEIGHT)
        reward = kw.get("reward", reward_fn)
        blend  = kw.get("blend", BLEND_EVAL)
        print(f"\n[sensitivity {gi+1}/{len(grid)}] {group} = {label}")

        rec = {"group": group, "value": label}
        for period, arrs, dyn, tag in PERIOD_SPECS:
            e_q, e_t, fan_q = [], [], []
            for seed in seeds:
                q_table = get_qtable(seed, band, w, reward)
                noise = eval_noise(seed, tag, len(arrs["T"]))
                mq = evaluate(arrs, make_q_controller(q_table), noise, dyn, blend=blend)
                mt = evaluate(arrs, threshold_action, noise, dyn, blend=blend)
                fan_q.append(mq["fan_pct"])
                e_q.append(mq["energy_kwh_month"]); e_t.append(mt["energy_kwh_month"])
            rec[f"{period}_fan"] = float(np.mean(fan_q))
            rec[f"{period}_dE"] = float(np.mean(
                100.0 * (np.array(e_q) - np.array(e_t)) / np.array(e_t)))
        out.append(rec)
        print(f"    P1 fan={rec['P1_fan']:.1f}% dE={rec['P1_dE']:+.1f}%  |  "
              f"P2 fan={rec['P2_fan']:.1f}% dE={rec['P2_dE']:+.1f}%")

    print(f"\nSensitivity finished in {(time.time()-t0)/60:.1f} min")
    return pd.DataFrame(out)


print("\n" + "=" * 78)
print("SENSITIVITY ANALYSIS")
print("=" * 78)
sens = run_sensitivity()
sens.to_csv(os.path.join(OUT_DIR, "sensitivity_raw.csv"), index=False)


# ==============================================================================
# CELL 7 -- edge footprint and the break-even condition (Eq. 2)
# ==============================================================================
visited = ms.attrs["visited"]
kb = ms.attrs["qtable_kb"]
print("\n--- edge deployment ---")
print(f"States with a learned policy (training only): "
      f"{np.mean(visited):.1f} +/- {np.std(visited, ddof=1):.1f} of {N_STATES} "
      f"[min {min(visited)}, max {max(visited)}]")
print(f"Q-table footprint: {np.mean(kb):.2f} +/- {np.std(kb, ddof=1):.2f} KB "
      f"({int(round(np.mean(visited)))} states x {N_ACTIONS} actions x 8 bytes)")
print(f"Remaining {N_STATES - int(round(np.mean(visited)))} states fall back "
      f"to the threshold rules on device.")


def break_even(h_ql, p_ql):
    """Eq. (2): fan-duty gap the agent must remove before it saves any energy."""
    return (P_HEATER * h_ql + P_PUMP * (p_ql - 1.0)) / P_FAN


print("\n--- break-even condition (Eq. 2) ---")
for period in ["P1", "P2"]:
    q = ms[(ms.period == period) & (ms.algorithm == "Q-Learning")]
    t = ms[(ms.period == period) & (ms.algorithm == "Threshold")]
    h, p = q["heater_pct"].mean() / 100, q["pump_pct"].mean() / 100
    gap_star = 100 * break_even(h, p)
    gap_obs = t["fan_pct"].mean() - q["fan_pct"].mean()
    verdict = "SAVES" if gap_obs > gap_star else "COSTS"
    print(f"{period}: required gap {gap_star:5.1f} pp | observed {gap_obs:5.1f} pp "
          f"-> {verdict} energy")
    stats_summary[f"{period}_breakeven_pp"] = float(gap_star)
    stats_summary[f"{period}_observed_gap_pp"] = float(gap_obs)


# ==============================================================================
# CELL 8 -- LaTeX emitters (paste the bodies straight into main.tex)
# ==============================================================================
def fmt(m, s, d=2):
    return f"${m:.{d}f}\\pm{s:.{d}f}$"


def emit_multiseed_table(df, path):
    lines = []
    for period in ["P1", "P2"]:
        for i, algo in enumerate(["Threshold", "Correlation", "Q-Learning"]):
            g = df[(df.period == period) & (df.algorithm == algo)]
            prefix = (f"\\multirow{{3}}{{*}}{{\\rotatebox{{90}}{{{period}}}}}\n & "
                      if i == 0 else " & ")
            lines.append(
                prefix + f"{algo:<12} & "
                + fmt(g.rmse_T.mean(), g.rmse_T.std()) + " & "
                + fmt(g.rmse_RH.mean(), g.rmse_RH.std()) + " & "
                + fmt(g.fan_pct.mean(), g.fan_pct.std(), 1) + " & "
                + fmt(g.heater_pct.mean(), g.heater_pct.std(), 1) + " & "
                + fmt(g.energy_kwh_month.mean(), g.energy_kwh_month.std(), 1)
                + " \\\\")
        if period == "P1":
            lines.append("\\midrule")
    body = "\n".join(lines)
    with open(path, "w") as f:
        f.write(body)
    return body


def emit_sensitivity_table(df, path):
    lines, last = [], None
    counts = df.groupby("group").size().to_dict()
    for _, r in df.iterrows():
        if r["group"] != last:
            if last is not None:
                lines.append("\\midrule")
            lines.append(f"\\multirow{{{counts[r['group']]}}}{{*}}{{{r['group']}}}")
            last = r["group"]
        val = r["value"].replace("*", "$^{\\dagger}$")
        lines.append(f" & {val} & {r['P1_fan']:.1f} & {r['P1_dE']:+.1f} & "
                     f"{r['P2_fan']:.1f} & {r['P2_dE']:+.1f} \\\\")
    body = "\n".join(lines)
    with open(path, "w") as f:
        f.write(body)
    return body


def emit_window_table(win, path):
    a, b = win
    body = (f"Temperature ($^{{\\circ}}$C) & {a['T_mean']:.1f} $\\pm$ {a['T_sd']:.1f} "
            f"& {b['T_mean']:.1f} $\\pm$ {b['T_sd']:.1f} \\\\\n"
            f"Humidity (\\%) & {a['RH_mean']:.1f} $\\pm$ {a['RH_sd']:.1f} "
            f"& {b['RH_mean']:.1f} $\\pm$ {b['RH_sd']:.1f} \\\\\n"
            f"Steps above {TH_FAN_T}\\,$^{{\\circ}}$C (\\%) "
            f"& {a['pct_above_fan_th']:.1f} & {b['pct_above_fan_th']:.1f} \\\\")
    with open(path, "w") as f:
        f.write(body)
    return body


print("\n" + "=" * 78)
print("LATEX OUTPUT")
print("=" * 78)
print("\n%--- Table: multiseed (body) ---")
print(emit_multiseed_table(ms, os.path.join(OUT_DIR, "tab_multiseed.tex")))
print("\n%--- Table: sensitivity (body) ---")
print(emit_sensitivity_table(sens, os.path.join(OUT_DIR, "tab_sensitivity.tex")))
print("\n%--- Table: windows (body) ---")
print(emit_window_table(WIN, os.path.join(OUT_DIR, "tab_windows.tex")))

numbers = dict(
    n_seeds=len(SEEDS), seeds=SEEDS,
    episodes=EPISODES, train_stride=TRAIN_STRIDE, eval_steps=EVAL_STEPS,
    n_train=int(len(TRAIN["T"])),
    visited_mean=float(np.mean(visited)),
    visited_sd=float(np.std(visited, ddof=1)),
    qtable_kb_mean=float(np.mean(kb)),
    qtable_kb_sd=float(np.std(kb, ddof=1)),
    n_states=N_STATES,
    dyn_mae_T_p1=DYN_P1["mae_T"], dyn_mae_T_p2=DYN_P2["mae_T"],
    dyn_mae_RH_p1=DYN_P1["mae_RH"], dyn_mae_RH_p2=DYN_P2["mae_RH"],
    reversion_mode=REVERSION_MODE,
    windows=WIN, **stats_summary)
with open(os.path.join(OUT_DIR, "paper_numbers.json"), "w") as f:
    json.dump(numbers, f, indent=2, default=float)
print(f"\nAll numbers written to {OUT_DIR}/paper_numbers.json")

print("\nRun make_fig.py to produce the paper figure.")
