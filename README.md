# Classifying SETI Technosignatures vs. Radio Interference with Classical ML

## Executive Summary

This capstone project applies classical, GPU-free, interpretable machine learning techniques — **Principal Component Analysis (PCA)**, **k-Means clustering**, and three classifiers (**LASSO** logistic regression, **Random Forest**, and **XGBoost**) — to the Search for Extraterrestrial Intelligence (**SETI**). The pipeline fits and evaluates these models on **3,784,704 spectrogram snippets** extracted from **6 real Green Bank Telescope (GBT) observations (~93 GB raw)** of the **ABACAD cadence** (the standard SETI observing pattern — three on-source observations of the target interleaved with three off-source observations of nearby reference stars), recorded on 2017-06-24 around the **nearby star HIP 13375 (~100 light-years from Earth)**. The pipeline correctly surfaces anomalous frequencies — the top candidate at **1426.61 MHz** scores **16× above the noise floor** — but visual cadence-stack inspection identifies it as a continuous Doppler-drifting satellite signal present in all six observations, not an extraterrestrial source.

**Bottom line:** classical ML can detect anomalies (distinguish unusual signals from background noise for downstream analysis) in real telescope data cheaply (laptop, no GPU) and transparently, but cannot algorithmically distinguish pointing-correlated **radio-frequency interference (RFI)** from real target-locked candidates. **No extraterrestrial intelligence (ETI) candidates were found — and finding them was never the goal.** The deliverable is the pipeline itself and an honest measurement of where classical ML hits its ceiling on real telescope data. This matches the outcome of the reference paper (Ma et al. 2023, *Nature Astronomy*) — 8 candidates from 820 stars, none confirmed on re-observation — and it precisely motivates why the cadence-aware β-VAE (β-Variational Autoencoder) approaches in modern SETI exist.

---

## Rationale

The Search for Extraterrestrial Intelligence (SETI) generates millions of signals of interest per observing campaign. The Breakthrough Listen GBT corpus alone is **~120 TB**. Without efficient filtering, finding the few astronomically interesting ones is impossible.

The deep-learning state-of-the-art (Ma et al. 2023) has two practical drawbacks: it requires significant GPU compute (the paper used **~50 GB total VRAM across 3 dedicated compute nodes** — see Supplementary Figure 11), and its decisions are opaque to peer-reviewing astronomers. **If a classical pipeline can do most of the same filtering work transparently — at a fraction of the cost and with full per-feature explanations — that lowers the cost of SETI research and gives astronomers a model whose decisions they can defend.** This project measures, on real GBT data, exactly how much classical methods give up versus the deep-learning baseline and where they run out of room.

---

## Research Question

**Can classical, GPU-free, interpretable machine learning — PCA dimensionality reduction, k-Means clustering, and L1-regularized / tree-based classification — distinguish potential extraterrestrial technosignatures from radio-frequency interference in real Breakthrough Listen telescope data, without neural networks or domain-specific feature engineering?**

---

## Data Sources

**Source:** [Breakthrough Listen Open Data Archive](http://seti.berkeley.edu/opendata) — UC Berkeley SETI Research Center.

**ABACAD cadence, observed 2017-06-24 with the Green Bank Telescope:**

| Position | Time (UTC) | Target | File |
|---|---|---|---|
| A1 (ON) | 15:07:11 | HIP 13375 | `spliced_blc0001020304050607_guppi_57928_54431_HIP13375_0044.gpuspec.0000.h5` |
| B (OFF) | 15:13:09 | HIP 12678 | `spliced_blc0001020304050607_guppi_57928_54789_HIP12678_0045.gpuspec.0000.h5` |
| A2 (ON) | 15:19:05 | HIP 13375 | `spliced_blc0001020304050607_guppi_57928_55145_HIP13375_0046.gpuspec.0000.h5` |
| C (OFF) | 15:24:58 | HIP 12790 | `spliced_blc0001020304050607_guppi_57928_55498_HIP12790_0047.gpuspec.0000.h5` |
| A3 (ON) | 15:30:51 | HIP 13375 | `spliced_blc0001020304050607_guppi_57928_55851_HIP13375_0048.gpuspec.0000.h5` |
| D (OFF) | 15:36:37 | HIP 12919 | `spliced_blc0001020304050607_guppi_57928_56197_HIP12919_0049.gpuspec.0000.h5` |

All six files are available at the [Breakthrough Listen Open Data Archive](http://seti.berkeley.edu/opendata). Filenames follow the BL convention: `spliced_blc<nodes>_guppi_<MJD>_<seconds-since-midnight-UTC>_<target>_<scan>.gpuspec.0000.h5` — MJD 57928 corresponds to 2017-06-24, and the seconds-since-midnight encoding (54431 = 15:07:11) matches the observation times above.

Each file: HDF5 filterbank, ~15 GB on disk, frequency range 1.0–1.9 GHz, native resolution **2.79 Hz × 18.7 s**, **16 time bins × ~318 million frequency channels** per observation. The ABACAD cadence is the standard SETI observing pattern — three observations of the target star (A) interleaved with three different OFF-source stars (B, C, D). A genuine signal from the target appears in all three A panels but is absent from the OFFs; terrestrial RFI generally appears in all six panels.

### Why HIP 13375?

HIP 13375 is a **nearby star, approximately ~100 light-years from Earth** — one of the closer Hipparcos-catalog targets observed by Breakthrough Listen. Three practical reasons drove the choice:

1. **Astrophysical priority** — proximity matters. A hypothetical transmitter of fixed power produces a signal-to-noise ratio that falls off as 1/distance². The closer the star, the stronger any real technosignature would be — making nearby stars (typically < 50 parsecs) the highest-priority targets in any SETI survey.
2. **Complete public ABACAD cadence available** — the Breakthrough Listen Open Data Archive contains a full six-file ABACAD cadence around HIP 13375 from the 2017-06-24 observing session, with all the matching OFF-source observations needed for cadence filtering. Many other targets in the archive are missing one or more cadence panels.
3. **Reproducibility** — the data is fully public, freely downloadable, and used in prior Breakthrough Listen publications, making any result here directly comparable to existing work.

Raw data files are not redistributed here due to size (~93 GB); they are downloaded directly from the Breakthrough Listen archive.

Reference paper: Ma, P. X. et al. (2023). *A deep-learning search for technosignatures of 820 nearby stars.* **Nature Astronomy**. arXiv: [2301.12670](https://arxiv.org/abs/2301.12670).

---

## Methodology

The pipeline runs in six stages, all implemented in pure scikit-learn / numpy / h5py (no neural networks, no GPU required):

0. **Raw-data acquisition** ([`src/download_cadence.py`](src/download_cadence.py)) — fetches the six Breakthrough Listen HDF5 files for the 2017-06-24 HIP 13375 ABACAD cadence from the BL Open Data Archive into `data/raw/`. ~93 GB total; runtime ~30–60 min depending on bandwidth.

1. **Streaming ingestion** ([`src/ingest_cadence.py`](src/ingest_cadence.py)) — read each of the 6 GBT files, slice into **621,544 snippets** of 16 time × 64 frequency pixels (after 8× frequency downsampling), median-normalize per snippet, and append to a single HDF5 archive `cadence_features.h5`. **Total: 3,784,704 snippets, 15.5 GB.**

2. **PCA dimensionality reduction** (notebook 2, Steps 3–4) — fit `StandardScaler` + `PCA(n_components=15)` on a 500K-snippet random sample (~2 GB in RAM), then transform all 3.78M snippets in 100K batches. **Per-snippet reconstruction error is the unsupervised anomaly score.**

3. **ABACAD candidate ranking** (notebook 2, Steps 5–6) — aggregate reconstruction errors into 10 kHz frequency bins; per-bin score = `mean(error in A panels) − mean(error in OFF panels)`. Filter known RFI bands (Inmarsat, GPS/GNSS, Iridium, MSS uplink, GSM-1800, AWS-3 cellular) and paper-excluded ranges (< 1.1 GHz, 1.2–1.34 GHz notch filter, > 1.9 GHz).

4. **Robustness checks** (notebook 2, Steps 7–9) —
   - **Drift-rate consistency** across A1/A2/A3 panels (the paper's filter, page 8)
   - **Per-time-bin max error** vs. averaged error, to catch intermittent signals diluted by averaging
   - **Dimensionality stability** at n=8 vs. n=15 (rank-1 confirmed stable, score essentially unchanged)

5. **Visual cadence-stack inspection** of top-5 candidates — the final SETI filtering step.

**Supervised classifier comparison** (notebook 2, Step 10) — three classifiers (LASSO, Random Forest, XGBoost) trained on a 100K-snippet sample with A-vs-OFF binary pseudo-labels (1 = HIP 13375 observation, 0 = other target). Hyperparameter tuning via `GridSearchCV` with 3-fold cross-validation (CV). **Evaluation metric: Receiver Operating Characteristic — Area Under the Curve (ROC AUC)**, chosen because it is threshold-independent and the binary classes are balanced.

**k-Means clustering** is performed via `MiniBatchKMeans(k=2..6)` on the 15-d PCA features, with Adjusted Rand Index (ARI) measured against A-vs-OFF labels and silhouette score computed on a 50K-snippet sample.

### Tooling

```
Python 3.13 · scikit-learn · xgboost · numpy · h5py · hdf5plugin · matplotlib · pandas · scipy
No GPU. Runs on a laptop (~20 GB peak RAM during ingestion).
```

---

## Results

### PCA anomaly detection

15-component PCA on the 3.78M real-cadence snippets retains **43.9% of variance** — a property of the high-dimensional spectrogram data, not the sample size (verified at n ∈ {8, 15, 20, 50}). For anomaly detection, this is sufficient: the unused 56% is mostly per-snippet noise and bandpass shape that we *don't* want to model — the 15 components capture the structure typical snippets share, and anomalies don't fit that subspace.

### ABACAD ranking — top candidates after RFI/excluded-band filtering

| Rank | Frequency (MHz) | ABACAD score | mean(A) | mean(OFF) |
|---|---|---|---|---|
| **1** | **1426.610** | **20.19** | 22.72 | 2.53 |
| 2 | 1404.000 | 1.22 | 2.05 | 0.83 |
| 3 | 1176.450 | 1.05 | 1.85 | 0.80 |
| 4 | 1124.930 | 0.76 | 2.36 | 1.60 |
| 5 | 1180.020 | 0.37 | 4.11 | 3.74 |

The rank-1 candidate scores **16× above the second-ranked candidate** — a clear standalone outlier, well outside the noise floor. Drift-consistency check across A1/A2/A3 panels passes (`drift_std = 0.047`). Rank-1 stable across n=8 (score 20.53) and n=15 (score 20.19); also stable when scoring by max-time-bin error instead of mean-time error (score 53.30).

### Visual cadence-stack inspection of the top candidate

A 6-panel cadence stack at 1426.61 MHz reveals a **continuous Doppler-drifting narrowband signal** (~0.6 Hz/s drift) **across all six observations** — A panels and OFF panels alike, with similar brightness. This is the diagnostic signature of a low-Earth-orbit satellite, not an extraterrestrial source. The pipeline ranked it #1 because antenna sidelobe gain varies with telescope pointing direction (giving slightly higher amplitude on A panels), but the underlying emitter is terrestrial.

Visual inspection of ranks 2–5 (drift-consistent candidates) confirmed the same pattern: every top candidate shows signals present in all six panels — RFI in every case. **Zero of the top-5 drift-consistent candidates pass visual cadence inspection as a real ETI candidate.**

### k-Means clustering

| k | Silhouette (50K sample) | ARI vs. A-vs-OFF |
|---|---|---|
| 2 | 0.580 | 0.029 |
| 3 | 0.643 | 0.011 |
| 4 | 0.213 | 0.110 |
| 5 | 0.206 | 0.107 |
| **6** | **0.301** | **0.175** |

ARI is uniformly low — discovered clusters only weakly correlate with cadence position. The k=6 cross-tabulation reveals that clusters track *per-target bandpass signature* (HIP 12790 and HIP 12919 each form distinctive clusters) rather than ON/OFF cadence identity. k-Means on PCA features is therefore **not sufficient on its own** for SETI candidate discrimination — it surfaces per-target spectral character, which is unrelated to ETI.

### Classifier comparison (A-vs-OFF binary, GridSearchCV, 3-fold CV, sample N=100,000)

| Model | Best hyperparameters | CV best AUC | Test AUC | Test accuracy | Fit time |
|---|---|---|---|---|---|
| LASSO logistic (L1) | `C = 0.1` | 0.7716 | 0.7710 | 0.7514 | ~5 s |
| Random Forest | `n_estimators=200, max_depth=None` | 0.9217 | **0.9252** | **0.8364** | ~45 s |
| XGBoost | `n_estimators=200, max_depth=4` | 0.9163 | 0.9204 | 0.8300 | ~8 s |

**Both non-linear classifiers substantially outperform LASSO** — a ~15-AUC-point gap (Random Forest 0.9252 / XGBoost 0.9204 vs. LASSO 0.7710) — indicating the relationship between PCA features and target identity is **non-linear**. Random Forest narrowly edges out XGBoost (within ~0.5 AUC points), consistent with Ma et al.'s choice of Random Forest on β-VAE latent vectors. The fact that two independent tree-based ensembles converge on essentially the same AUC ceiling (~0.92) tells us **classifier choice is no longer the bottleneck** — the discriminative information available in the 15-d PCA representation has been substantially extracted.

**Important interpretation caveat.** High AUC reflects target-bandpass discriminability between HIP 13375 and the OFF targets (different sky regions seeing different RFI environments, different receiver-state effects); it is **not** evidence of ETI detection. To improve beyond the ~0.92 ceiling, the next lever is a richer feature representation (e.g. a cadence-aware encoder), not a more powerful classifier.

### Bottom line

**No ETI candidates found in the 2017-06-24 HIP 13375 cadence.** This matches the outcome of Ma et al.'s search across 820 stars. The classical pipeline successfully identifies anomalous frequencies but cannot algorithmically distinguish "RFI with pointing-dependent amplitude" from "real target-locked signal" — visual cadence inspection is required for the final decision. This is precisely the gap the paper's cadence-aware β-VAE training is designed to close.

---

## Next Steps

1. **Cadence-aware feature representation.** Concatenate the 6-panel feature vectors at each frequency before PCA so the components encode cadence shape rather than per-snippet appearance — the classical analog of the paper's β-VAE clustering loss (Λ_S + Λ_R).
2. **Drift-rate consistency on all top-K candidates.** Currently implemented for top-20 in notebook 2 Step 7; extending to top-1000 would auto-reject more RFI without requiring visual inspection.
3. **Compare against `turbo_seti`.** The Breakthrough Listen team's production drift-search code is the right benchmark; overlap with the classical pipeline's top candidates would validate the approach as a less-expensive screening tool.
4. **Scale to multiple cadences.** This project analyzed one campaign (6 files, 1 ABACAD cadence). Processing the full 1004-cadence Breakthrough Listen archive (Ma et al.'s dataset) would establish a corpus of validated negatives and surface cross-cadence patterns.
5. **Wider snippet windows for higher drift rates.** Current 512-channel snippet covers ±5 Hz/s drift sensitivity. Ma et al. use 4,096-channel snippets with overlapping search windows to reach ±10 Hz/s. Adopting that schema would widen the detection space at the cost of ~8× compute.

---

## Outline of Project

What ships in this repository:

```
seti_capstone_solution/
├── README.md                                         # this file (non-technical project summary)
├── Capstone Project_Final Report.md                  # 6-section technical report
├── .gitignore
│
├── 1. data_loading_and_exploration.ipynb             # notebook 1 — load cadence metadata, per-panel sanity checks
├── 2. pca_dimensionality_reduction.ipynb             # notebook 2 — full pipeline (PCA → ABACAD → drift consistency → classifiers → visual inspection)
│
├── src/
│    ├── __init__.py
│    ├── download_cadence.py                          # downloads the 6 raw GBT HDF5 files from the BL Open Data Archive
│    └── ingest_cadence.py                            # streams the 6 raw files into cadence_features.h5
│
├── data/processed/
│    ├── abacad_candidates.csv                        # top-50 raw candidates (pre-RFI-band filter)
│    └── abacad_candidates_clean.csv                  # top-50 candidates after RFI-band exclusion
│
└── images/
     └── cadence_stack_rank1_1426MHz.png              # 6-panel cadence stack of the rank-1 candidate (1426.61 MHz)
```

**Not in the repo (too large for GitHub) — three reproducibility paths:**

| Artifact | Size | How to obtain |
|---|---|---|
| `data/raw/*.h5` (6 files) | ~93 GB | `python src/download_cadence.py` (~30–60 min) |
| `data/processed/cadence_features.h5` | ~15.5 GB | `python src/ingest_cadence.py` (~40 min, after the raw files are present) |
| `data/processed/cadence_pca_features.npz` | ~295 MB | **Fast path:** `gh release download v1.0 --pattern "cadence_pca_features.npz" --dir data/processed/` — skips the 12-minute PCA stream-transform step |

### Reproduction

**Full reproduction from scratch (~1.5 hours):**

```bash
pip install numpy pandas h5py hdf5plugin scikit-learn xgboost matplotlib jupyterlab
python src/download_cadence.py    # ~30–60 min — pulls 6 raw HDF5 files into data/raw/
python src/ingest_cadence.py      # ~40 min — produces data/processed/cadence_features.h5
# Then open notebooks 1 and 2 in JupyterLab and "Run All"
```

**Fast-path reproduction (~5 min, skips raw data + PCA fit):**

```bash
pip install numpy pandas h5py hdf5plugin scikit-learn xgboost matplotlib jupyterlab
gh release download v1.0 --pattern "cadence_pca_features.npz" --dir data/processed/
# Then open notebook 2 in JupyterLab and run cells from §3 onwards
```

---

## Limitations

1. **Single-campaign scope** — analyzed one 6-file ABACAD cadence around HIP 13375. The reference paper analyzed 1004 cadences (~115M snippets); findings here are illustrative, not population-level.
2. **No real ETI detected** — by design, this is consistent with the reference paper's outcome and the prior that any genuine ETI signal would be a rare event.
3. **Linear PCA on raw spectrogram pixels** has known limits — it captures variance modes but not the cadence structure that the paper's neural-net clustering loss exploits. The "Next Steps" section addresses this directly.

---

## References

1. **Ma, P. X., Ng, C., Rizk, L., Croft, S., Siemion, A. P. V., Brzycki, B., ... & Worden, S. P.** (2023). *A deep-learning search for technosignatures of 820 nearby stars.* **Nature Astronomy**. arXiv: [2301.12670](https://arxiv.org/abs/2301.12670). — The reference β-VAE + Random Forest pipeline this project compares against.

2. **Enriquez, J. E., Siemion, A., Foster, G., et al.** (2017). *The Breakthrough Listen Search for Intelligent Life: 1.1–1.9 GHz Observations of 692 Nearby Stars.* The Astrophysical Journal, 849, 104. arXiv: [1709.03491](https://arxiv.org/abs/1709.03491). — Established the ABACAD cadence observation strategy used here.

3. **Lebofsky, M., Croft, S., Siemion, A. P. V. et al.** (2019). *The Breakthrough Listen Search for Intelligent Life: Public Data, Formats, Reduction, and Archiving.* PASP 131, 124505. — Describes the HDF5 filterbank format used here.

4. **Burgess, C. P., Higgins, I., Pal, A., et al.** (2018). *Understanding disentangling in β-VAE.* arXiv: [1804.03599](https://arxiv.org/abs/1804.03599). — Theoretical foundation for the β-VAE used in the reference paper.

5. **Higgins, I., Matthey, L., Pal, A., et al.** (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR.

6. **Breakthrough Listen Open Data Archive.** UC Berkeley SETI Research Center. [seti.berkeley.edu/opendata](http://seti.berkeley.edu/opendata). — Source of all GBT recordings used here.

---

## Citation

```bibtex
@misc{mual2026seti,
  author = {Mual, Rohit},
  title  = {Classifying SETI Technosignatures vs. Radio Interference with Classical ML:
            A PCA + k-Means + LASSO/RF/XGBoost Study on Breakthrough Listen GBT Data},
  year   = {2026},
  note   = {UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence -- Capstone Project, May 2026}
}
```

---

## Acknowledgments

Special thanks and inspiration: **[Dr. Vishal Gajjar](https://www.linkedin.com/in/vishal-gajjar-b208913a/)** — Principal Astronomer at the SETI Institute and Breakthrough Listen initiative. Co-author of the Ma et al. 2023 reference paper that this capstone benchmarks against.

---

## Contact

**Rohit Mual**
- Email: [rohit.mual@gmail.com](mailto:rohit.mual@gmail.com)
- LinkedIn: [linkedin.com/in/rohitmual](https://linkedin.com/in/rohitmual)
