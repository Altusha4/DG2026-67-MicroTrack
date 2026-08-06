# Digital Twin Benchmarking of Microclimate Control for IoT Hydroponic Grow-Boxes

Reproduction package for the IEEE DG 2026 paper *"Digital Twin Benchmarking of
Microclimate Control for IoT Hydroponic Grow-Boxes: A Two-Period Evaluation of
Threshold, Correlation-Aware, and Q-Learning Strategies"* by Altynay Yertay,
Kuanysh Bakirov, and Aruzhan Shoman.

This repository contains a four-layer Digital Twin of an ESP32-S3-instrumented
hydroponic grow-box, driven by ~1.3 million real telemetry records from two
non-overlapping operating periods. The Twin is used as a reproducible harness to
benchmark three microclimate controllers — a fixed-threshold baseline, a
correlation-aware orchestrator, and a tabular Q-learning agent with a
comfort-band reward — over N = 10 random seeds. Its purpose is to test whether
the energy savings routinely reported for RL controllers survive a change of
operating regime, by evaluating a policy trained on one period against a
genuinely future held-out period.

## Headline result

The reported saving reverses on held-out data. Against the threshold baseline,
the Q-learning agent:

| Regime | Total actuator energy | Fan activation |
|---|---|---|
| Period 1 (in-period, warmer) | **−26.0 ± 21.9 %** (saves), *p* = 0.010 | −49.7 ± 18.6 %, *p* = 0.002 |
| Period 2 (held-out, ≈2.5 °C cooler) | **+95.8 ± 65.9 %** (costs more), *p* = 0.002 | +4.7 ± 26.0 % (parity), *p* = 0.49 |

Means ± SD over 10 seeds; *p* from a two-sided Wilcoxon signed-rank test on the
10 paired runs. Within a seed all three controllers see an identical pre-drawn
noise stream, so the comparison is paired and any difference is attributable to
the policy.

In Period 1 the baseline ventilates on 89.5 % of steps, leaving ample headroom
for the agent to exploit. In Period 2 it ventilates on only 27.1 % of steps; the
frozen policy loses the advantage entirely and its residual heating actions
dominate the energy budget. The saving is therefore a property of the operating
regime, not of the controller.

The paper turns this into a closed-form break-even condition: the agent saves
energy only where the baseline over-ventilates by more than
Δf\* = (P_h·h_QL + P_p·(p_QL − 1)) / P_f percentage points of fan duty. That is
6.8 points in-period against an observed 44.5, but 82.7 points out-of-period
against an observed −1.5 — more than the baseline's entire fan duty of 32.3 %.

All figures above are recomputable from `results/multiseed_raw.csv` and
`results/paper_numbers.json`.

## Repository layout

```
├── src/                 experiment and figure code
│   ├── microtrack_revision.py   Digital Twin, three controllers, multi-seed run,
│   │                            sensitivity analysis, LaTeX table generation
│   └── make_fig.py              regenerates Fig. 1 from multiseed_raw.csv
├── data/                ESP32-S3 telemetry, two periods (zipped)
│   ├── box_data.zip             Period 1
│   └── box_data_p2.zip          Period 2
├── results/             outputs of the published run
├── paper/main.tex       manuscript source
├── requirements.txt
├── LICENSE              MIT — applies to src/ only
├── LICENSE-DATA.md      CC BY 4.0 — applies to data/, results/, paper/
└── CITATION.cff
```

## Installation

Requires Python 3.9 or newer.

```bash
git clone https://github.com/Altusha4/DG2026_67_MicroTrack.git
cd DG2026_67_MicroTrack
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The archives in `data/` are read directly and do not need to be unpacked
manually.

## Running

Full run — 10 seeds × 8 training episodes over the 520,100-step training set,
plus the 5-seed sensitivity sweep:

```bash
python src/microtrack_revision.py
python src/make_fig.py
```

This takes roughly **35–45 minutes on a single CPU core** and needs no GPU. The
first command rewrites the tables, raw metrics, and `paper_numbers.json` in
`results/`; the second regenerates Fig. 1 from those metrics. The experiment
script deliberately writes no figure, so the figure used in the paper is only
ever produced by `make_fig.py`.

`make_fig.py` writes three files:

| File | Purpose |
|---|---|
| `generalization.pdf` | vector; **this is what the manuscript includes**. 3.4 × 2.5 in, 8 pt STIXGeneral, fonts embedded as Type 42 (IEEE PDF eXpress rejects Type 3) |
| `generalization.svg` | vector source, for editing or the web |
| `generalization.png` | 600 dpi raster preview only |

`pdflatex` cannot `\includegraphics` an SVG, which is why the paper uses the PDF
rather than the SVG.

`paper/main.tex` includes the figure by bare filename, so building the PDF needs
it beside the source:

```bash
cp results/generalization.pdf paper/
cd paper && pdflatex main.tex && pdflatex main.tex
```

For a quick check that the pipeline runs end to end, `--fast` trades seeds and
episodes for wall-clock (about 3 minutes):

```bash
python src/microtrack_revision.py --fast --out-dir /tmp/smoke
```

**Numbers from a fast run do not match the paper** — only the full run
reproduces the published values, so direct fast runs at a scratch output
directory to avoid overwriting `results/`.

Both scripts accept `--help` and take all input and output paths as command-line
arguments, defaulting to the layout above. Paths resolve relative to the
repository root, so the scripts work from any working directory.

`scipy` is imported defensively: without it the run still completes, but the
Wilcoxon *p*-values quoted in the paper are skipped.

## Data

Telemetry was logged at an ≈5.6-second cadence by an ESP32-S3 reading fourteen
channels in four sensor groups. The two periods were archived separately and are
non-overlapping by construction; the loader additionally verifies that no
timestamp occurs in both archives, so nothing from the hold-out can leak into
training.

| | Period 1 (in-period) | Period 2 (held-out) |
|---|---|---|
| Span | 29 May 2025 – 28 Apr 2026 | 29 Apr – 15 Jun 2026 |
| Daily CSV files | 137 | 48 |
| Raw records | 808,078 | 726,781 |
| **After quality filtering** | **650,126** | **652,909** |
| Coverage | 100 of 334 days (≈12.6 % duty) | 44 of 48 days (≈89 % duty) |
| Use | 520,100 train / 130,026 eval | evaluation only |

Quality filter: records are dropped when T ∉ [15, 40] °C, RH ∉ [10, 95] %, or
CO₂ ∉ [300, 2000] ppm. Records with a missing value in any of these three
channels are also dropped, because the filter is applied before imputation.

Each row is a timestamp, a device identifier, and fourteen sensor channels:

```
timestamp, device_id, soil1..soil5, ph_level, ec, tds, turbidity, co2,
air_temperature, air_humidity, water_temperature, light_level
```

**Four channels drive the control loop:** `air_temperature`, `air_humidity`,
`co2`, and `light_level`. The remaining ten (the five soil-moisture probes and
the hydro-chemistry group — EC, TDS, pH, turbidity, water temperature) were
faulty or constant for most of the record and are excluded from the Twin; they
are shipped unmodified for completeness and transparency. Period 2 in particular
reports all five soil probes as 0 and water temperature as empty.

### Note on schema drift

The sixteen-column schema stabilised only toward the end of May 2025. Some early
Period 1 files use a shorter layout without the hydro-chemistry group, and
therefore without a `co2` column; a couple of the earliest files carry no header
row at all. All Period 2 files use the full schema.

This matters for the record counts. The loader inserts `NaN` for any of the four
control channels a file does not provide, and the quality filter drops rows whose
`co2` is `NaN` — so **rows from the short-schema files are removed in their
entirety**, not merely trimmed for outliers. The gap between 808,078 raw and
650,126 quality-filtered Period 1 records is therefore the sum of two effects:
ordinary outlier rejection, plus the wholesale loss of the pre-CO₂ files. This is
intentional and is why Period 1 covers only 100 of 334 calendar days.

To see the split on your own copy:

```bash
python -c "
import zipfile, io, pandas as pd
with zipfile.ZipFile('data/box_data.zip') as z:
    names = [n for n in z.namelist() if n.endswith('.csv')]
    short = [n for n in names
             if 'co2' not in (pd.read_csv(io.BytesIO(z.read(n)), nrows=0).columns)]
print(f'{len(names)} CSV files, {len(short)} without a co2 column')
"
```

The archives contain sensor readings and timestamps only. They carry no network
addresses, credentials, device serial numbers, or personal data.

## Reproducing each table and figure

| Paper | Produced by | Result file |
|---|---|---|
| Table I (related work) | — | not generated; hand-written literature comparison |
| Table II (environment statistics) | `microtrack_revision.py` | `results/tab_windows.tex`, `results/paper_numbers.json` (`windows`) |
| Table III (multi-seed comparison) | `microtrack_revision.py` | `results/tab_multiseed.tex`, per-seed values in `results/multiseed_raw.csv` |
| Table IV (sensitivity) | `microtrack_revision.py` | `results/tab_sensitivity.tex`, raw sweep in `results/sensitivity_raw.csv` |
| Fig. 1 (energy reversal) | `make_fig.py` | `results/generalization.pdf` (plus `.svg` and `.png`) |
| In-text statistics (Wilcoxon *p*, relative deltas, break-even, Twin fidelity) | `microtrack_revision.py` | `results/paper_numbers.json` |

Two tables in the manuscript are hand-augmented rather than pasted verbatim, so
compare values rather than diffing the files:

- **Table II** additionally reports full-period statistics and a CO₂ row;
  `tab_windows.tex` supplies only the two evaluation-window columns.
- **Table III** additionally reports the Pump % column and quotes energy to two
  decimals; `tab_multiseed.tex` omits Pump % and rounds to one. Both columns are
  present in `multiseed_raw.csv` (`pump_pct`, `energy_kwh_month`), and every
  value in the published table reproduces exactly from it.

One apparent discrepancy is expected: the default row of Table IV reports 51.4 %
fan duty for Period 1, whereas Table III reports 45.1 % for the same
configuration. The sensitivity sweep averages over the first five seeds and the
multi-seed run over all ten; averaged over the same five seeds the two agree
exactly. The paper states this in the Table IV caption.

To rebuild Table III from the raw per-seed metrics without rerunning the
experiment:

```bash
python -c "
import pandas as pd
ms = pd.read_csv('results/multiseed_raw.csv')
cols = ['rmse_T','rmse_RH','fan_pct','heater_pct','pump_pct','energy_kwh_month']
print(ms.groupby(['period','algorithm'])[cols].agg(['mean','std']).round(2))
"
```

The energy column is not an independent measurement — it is
(18 W·fan + 120 W·heater + 10 W·pump) × 730 h, so it can be checked against the
duty cycles in the same file:

```bash
python -c "
import pandas as pd, numpy as np
ms = pd.read_csv('results/multiseed_raw.csv')
calc = (18*ms.fan_pct + 120*ms.heater_pct + 10*ms.pump_pct)/100*730/1000
print('max abs error:', np.abs(calc - ms.energy_kwh_month).max())
"
```

Fig. 1 is vector output and therefore resolution-independent. Regenerating it on
a different matplotlib version reproduces the layout and every value exactly,
but the file will not be byte-identical.

## Citation

```bibtex
@inproceedings{yertay2026microtrack,
  title     = {Digital Twin Benchmarking of Microclimate Control for IoT
               Hydroponic Grow-Boxes: A Two-Period Evaluation of Threshold,
               Correlation-Aware, and Q-Learning Strategies},
  author    = {Yertay, Altynay and Bakirov, Kuanysh and Shoman, Aruzhan},
  booktitle = {TODO: full proceedings title of IEEE DG 2026},
  year      = {2026},
  pages     = {TODO--TODO},
  publisher = {IEEE},
  doi       = {TODO}
}
```

TODO: fill in `pages`, `doi`, and the full proceedings title once the
proceedings are published.

## License

| Contents | License |
|---|---|
| `src/` | MIT — see `LICENSE` |
| `data/`, `results/`, `paper/` | CC BY 4.0 — see `LICENSE-DATA.md` |

## Acknowledgment

This research was funded by the Committee of Science of the Ministry of Science
and Higher Education of the Republic of Kazakhstan (Grant No. BR24992852,
"Intelligent models and methods of the Smart City digital ecosystem for
sustainable development and the improvement of the citizens' quality of life").

## Contact

Kuanysh Bakirov — k.bakirov@enu.kz
