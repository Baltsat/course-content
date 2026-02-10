# PM_ENV Task Specification — Steinmetz Multi-Area Neural Population Curation

## Env: Steinmetz Decision Decoding

**Skill**: neural quality filtering + cross-subject region-level feature engineering via balanced accuracy on hidden mice

**Summary**: student receives Neuropixels recordings (39 sessions, 10 mice, 42 brain areas, ~30K neurons) → filters bad neurons, engineers region-aware features with cross-session normalization → system trains fixed LogReg → scores balanced accuracy on 4 hidden sessions from mice the student has never seen

---

## Task: steinmetz-curate

**Category**: data curation | neural quality filtering + cross-subject generalization
**Metric**: balanced accuracy on left/right decision decoding, tested on held-out mice

**Prompt to student**:
```
Goal: Curate multi-area Neuropixels recordings for cross-subject decision decoding.
Constraints: 4h bash, 4 GB RAM, CPU only, no network.
Deliverables: unit_quality.npz (neuron labels), neural_features.npz (feature matrix),
              cross_session_predictions.npz (predictions for hidden mice)
Evaluation: balanced accuracy of fixed LogReg on 4 hidden sessions from unseen mice.
            Using only stimulus contrast gives ~65% — you need neural features to beat this.
```

---

## 1. Upstream & Models

### Datasets

- **Source**: Steinmetz et al. 2019, Nature 576, 266–273 — [paper](https://www.nature.com/articles/s41586-019-1787-x)
- **Data**: [Figshare](https://figshare.com/articles/dataset/Dataset_from_Steinmetz_et_al_2019/9598406) | [NMA preprocessed](https://osf.io/agvxh/)
- **Location**: `env_data/sessions/` (39 sessions as `.npz`) | **Size**: ~2 GB
- **Format**: per session — `spks` (n_neurons, n_trials, 250), `brain_area` (n_neurons), `contrast_left/right` (n_trials), `response` (n_trials), `trough_to_peak` (n_neurons), behavioral signals (wheel, pupil, face, licks)
- **Content**: 250 bins × 10ms = 2.5s per trial; bin 50 = stimulus onset; 10 mice, 42 brain areas, ~29K total neurons
- **Quality**: MESSY —
  - ~25% of neurons fire < 0.5 Hz (noise/multi-unit contamination)
  - different sessions record DIFFERENT brain regions (probe placement varies)
  - ~15% "lapse" trials (random choice regardless of stimulus)
  - cross-mouse: completely different neurons, no unit matching possible
  - neuron counts range 50–1500 per session

### Models

- **Downstream**: `LogisticRegression(C=0.1, max_iter=1000)` — FIXED, student cannot change
- Binary classification: left (response=+1) vs right (response=-1); nogo trials (response=0) EXCLUDED

### Validation

- **Student sees**: sessions 1–35 (30 train + 5 dev), full behavioral labels
- **Student also sees**: sessions 36–39 neural data in `env_data/test_unlabeled/` (spks, brain_area, contrast — but NO response labels)
- **Hidden**: `scoring_data/test_labels/` — sessions 36–39 true response labels
- **Note**: sessions 36–39 are from DIFFERENT MICE than sessions 1–35

---

## 2. Tools & Workflow

### Tools

**bash** — 600s/call, 14400s cumulative

**submit_answers** — key-value dict for exploration step

**packages**: numpy, pandas, scipy, scikit-learn, matplotlib

### Workflow

1. **Explore** (0–30 min): count neurons per area, compute firing rate distributions, check trial counts, identify low-quality units
2. **Quality filtering** (30–90 min): develop quality criteria — firing rate threshold, waveform width, stability, task modulation — assign quality labels
3. **Features** (90–150 min): build region-level features per trial — mean rates in time windows, population selectivity, cross-region features, normalization
4. **Predict** (150–240 min): cross-session normalization pipeline, generate predictions for hidden mice

---

## 3. Answer (What Student Submits)

### Deliverables

**Step 1**: `submit_answers` — 8 exploration answers (session count, total neurons, brain areas, top regions, trial types, mean firing rate, % low-firing, correct trials)

**Step 2**: `unit_quality.npz` — per-session neuron quality labels (0=bad, 1=ok, 2=good) + criteria description

**Step 3**: `neural_features.npz` — feature matrix X_train/X_val with labels y_train/y_val (nogo excluded), feature names, session IDs

**Step 4**: `cross_session_predictions.npz` — binary predictions (left/right) for sessions 36–39 + pipeline description

### Pre-checks (must pass or score = 0)
- [ ] all files exist, load correctly
- [ ] quality labels cover all sessions
- [ ] nogo trials excluded from predictions
- [ ] predictions are binary (0 or 1)

---

## 4. Evaluation (Downstream Pipeline)

### Downstream Data
Hidden test: sessions 36–39 true response labels in `scoring_data/test_labels/` + all-neuron baseline decoder pre-computed for comparison

### Pipeline

**Step 1**: ExactMatch on exploration answers → 0.0–1.0

**Step 2**: for each training session, decode using ONLY student's "good" neurons vs ALL neurons baseline → improvement in cross-validated accuracy = quality filtering score

**Step 3**: train LogReg on student's feature matrix, eval balanced accuracy on val + overfitting penalty if train–val gap > 15pp

**Step 4**: student's predictions scored against hidden truth → mean balanced accuracy across 4 test sessions

### Score

```
bal_acc ≤ 0.50 → 0.0  (chance)
bal_acc = 0.60 → 0.3  (contrast-only prediction)
bal_acc = 0.70 → 0.5  (decent neural features)
bal_acc = 0.80 → 0.7  (good cross-area features + normalization)
bal_acc = 0.85 → 0.85
bal_acc ≥ 0.90 → 1.0  (near-optimal — transformer territory)
```

### Anti-hack
- response labels removed from test_unlabeled/ → can't cheat
- contrast alone → ~65–70% bal_acc; neural features needed for higher
- wheel/reaction_time directly encode response → overfitting penalty catches label leakage
- constant predictions → check prediction variance + class balance
- overfit to training mice → step 4 tests on UNSEEN mice with different neurons

---

## 5. Progressive Difficulty

**Scale via**: ↓sessions | remove region labels | add decoy neurons

**v1**: 35 train sessions, 4 hidden test, full neural data → median 0.30
**v2**: restrict to 10 best sessions (student must SELECT which) → median 0.25
**v3**: remove brain_area labels — student must discover region identity from activity → median 0.15
**v4**: add decoy neurons (random Poisson) to 20% of units → median 0.10

---

## 6. Why This Will Work — Scientific Evidence

### choice encoding is regionally restricted (PROVEN)

[Steinmetz et al. 2019](https://www.nature.com/articles/s41586-019-1787-x) proved that choice signals are NOT uniformly distributed across the brain:

> "neurons encoding upcoming choices occupied restricted regions in neocortex, basal ganglia and midbrain. Choice signals were rare and emerged with indistinguishable timing across regions." — p.266

| region role | key areas | implication for curation |
|------------|-----------|------------------------|
| **choice-encoding** | MOs, PL, CP, SNr, SCm, MRN | INCLUDE — rare but highly informative |
| **stimulus-encoding** | VISp, VISl, LGd, SCs | encodes contrast, NOT choice — partial value |
| **noise for choice** | OLF, PIR, hippocampal | EXCLUDE — not task-relevant |

A naive student will assume VISp (largest region by neuron count) encodes choice. It does NOT — this is the CORE neuroscience insight of the paper.

### cross-mouse decoding WORKS (validated at scale)

The [IBL 2025](https://www.nature.com/articles/s41586-025-09235-0) (459 sessions, 139 mice, 241 brain regions) used the SAME task design as Steinmetz:

> "choice was decodable from many brain regions" using "regularized logistic regression" ACROSS animals.

Our evaluation design (train mice 1–8, test mice 9–10) directly mirrors the IBL analysis.

### the curation gap is calibrated from published results

| method | balanced accuracy | source |
|--------|-------------------|--------|
| chance | 50% | — |
| contrast difference only | 65–70% | Steinmetz 2019 task design |
| raw neural (all neurons, single window) | 55–65% | curse of dimensionality (p >> n) |
| region-level mean rates + z-score | 70–75% | region aggregation handles p >> n |
| multi-window + PCA + behavioral signals | 80–85% | near-optimal for LogReg |
| **LogReg ceiling** | **~85%** | limited by trial variability + lapse trials |
| neural foundation models (NEDS, MtM) | 85–90% | [NEDS](https://arxiv.org/abs/2504.08201), [MtM](https://arxiv.org/abs/2407.14668) |

LogReg on raw ~750 neurons per session = 55–65% (p >> n overfitting). LogReg on curated region-level features = 80–85%. **That's a +20pp curation gap** — entirely driven by quality filtering + region selection + feature engineering.

### the action space is HUGE and documented

~350 base features: 8 region groups × 6 time windows × 4 statistics + PCA + behavioral. With normalization (6 methods), region selection (2^7 combos), quality thresholds (600 combos) → **~33 billion configurations**, ~50–100 independent decisions.

[Zatka-Haas et al. 2021 eLife](https://elifesciences.org/articles/63163): "a stochastic circuit downstream of cortex linearly weights VIS and MOs activity to produce choice" — optogenetic validation that MOs + VIS are the key regions.

### key repos for baseline implementation

| repo | purpose |
|------|---------|
| [steinmetz-et-al-2019](https://github.com/nsteinme/steinmetz-et-al-2019) | original paper: kernel analysis, ccCP |
| [steinmetz2019_NMA](https://github.com/MouseLand/steinmetz2019_NMA) | NMA preprocessed loader + binning |
| [NEDS](https://github.com/yzhang511/NEDS) | neural encoding/decoding at scale |
| [CEBRA](https://github.com/AdaptiveMotorControlLab/CEBRA) | learnable latent embeddings |
| [IBL data portal](https://www.internationalbrainlab.com/data) | same task, 459 sessions |

---

*end of specification*
