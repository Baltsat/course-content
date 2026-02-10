# PM_ENV Task Specification — CalMS21 Mouse Behavior Feature Curation

## Env: CalMS21 Behavior Classification

**Skill**: feature engineering from pose keypoints + class imbalance handling via rare behavior F1

**Summary**: student receives mouse social behavior trajectories (7 body parts × 2 mice, 30 Hz) with extreme class imbalance (63% "other") → engineers geometric/kinematic/social features, curates balanced training set → system trains fixed LogReg → scores macro F1 on hidden test sequences (different mouse pairs)

---

## Task: calms21-curate

**Category**: data curation | feature engineering + class imbalance
**Metric**: macro F1 on {attack, investigation, mount} — "other" excluded from scoring

**Prompt to student**:
```
Goal: Engineer features from raw pose keypoints to maximize rare behavior classification.
Constraints: 4h bash, 4 GB RAM, CPU only, no network.
Deliverables: features.npz (feature matrix), curated_train.npz (balanced set), predictions.npz
Evaluation: macro F1 of fixed LogReg on hidden test sequences you cannot access.
            Predicting "other" always = score 0.0.
```

---

## 1. Upstream & Models

### Datasets

- **Source**: CalMS21 — [Sun et al. 2021 NeurIPS](https://arxiv.org/abs/2104.02710)
- **Data**: [data.caltech.edu/records/s0vdx-0k302](https://data.caltech.edu/records/s0vdx-0k302)
- **Location**: `env_data/train/` (70 sequences), `env_data/val/` (19 sequences) | **Size**: ~500 MB
- **Format**: `.npy` per sequence — keypoints `(n_frames, 2, 2, 7)`, annotations `(n_frames,)`, tracking confidence `(n_frames, 2, 7)`
- **Content**: 7 keypoints (nose, ears, neck, hips, tail) per mouse, 2 mice (resident + intruder), 30 Hz, ~507K training frames
- **Quality**: MESSY —
  - extreme class imbalance: attack 2.8%, investigation 29%, mount 5.6%, other 63%
  - keypoint tracking failures when mice overlap (~3% of frames)
  - annotation noise at behavior boundaries (±5 frames between annotators)
  - sequences vary 900–3000 frames

### Models

- **Downstream**: `LogisticRegression(C=0.1, multi_class='multinomial', class_weight='balanced')` — FIXED, student cannot change
- class_weight='balanced' is built in; the challenge is feature engineering, not just adding weights

### Validation

- **Student sees**: train (70 seq, annotated) + val (19 seq, annotated)
- **Hidden**: `scoring_data/test/` — 50 sequences from different mouse pairs
- **Note**: train/test split by animal identity (completely different mice)

---

## 2. Tools & Workflow

### Tools

**bash** — 600s/call, 14400s cumulative

**submit_answers** — key-value dict for exploration step

**packages**: numpy, pandas, scipy, scikit-learn, matplotlib

### Workflow

1. **Explore** (0–20 min): load data, count classes, examine keypoint structure
2. **Features** (20–90 min): engineer distances, velocities, angles, temporal rolling stats
3. **Curate** (90–150 min): handle imbalance — undersample "other", SMOTE rare classes, remove boundary frames
4. **Predict** (150–240 min): full pipeline, generate predictions on validation

---

## 3. Answer (What Student Submits)

### Deliverables

**Step 1**: `submit_answers` — 8 exploration answers (sequence counts, class distribution, transitions)

**Step 2**: `features.npz` — feature matrix X_train/X_val with named features, labels y_train/y_val

**Step 3**: `curated_train.npz` — balanced training set with optional sample weights

**Step 4**: `predictions.npz` — per-frame predictions + class probabilities on validation set

### Pre-checks (must pass or score = 0)
- [ ] all files exist, valid format
- [ ] labels in {0,1,2,3}
- [ ] at least 3 unique classes predicted
- [ ] no all-NaN feature columns

---

## 4. Evaluation (Downstream Pipeline)

### Downstream Data
Hidden test: 50 sequences in `scoring_data/test/` + 22 canonical reference features pre-computed for cross-evaluation

### Pipeline

**Step 1**: ExactMatch on exploration answers → 0.0–1.0

**Step 2**: train LogReg on student's features, eval macro F1 on val (excluding "other") + feature diversity bonus (rewards using distance + velocity + orientation + temporal features)

**Step 3**: train LogReg on student's curated set, eval on hidden test using reference features → macro F1

**Step 4**: student's own predictions scored against hidden truth → macro F1

### Score

```
F1 = 0.00 → 0.0  (predicts "other" always)
F1 = 0.30 → 0.3  (basic features, no imbalance handling)
F1 = 0.50 → 0.5  (decent features + resampling)
F1 = 0.65 → 0.7  (good features + temporal context)
F1 = 0.79 → 0.9  (1D CNN baseline level)
F1 ≥ 0.83 → 1.0  (LogReg ceiling)
```

### Anti-hack
- predict "other" always → F1 = 0 (excluded from metric)
- test data in scoring_data/, inaccessible
- test sequences from different mouse pairs → can't memorize labels
- constant predictions → require ≥3 classes
- random features → diversity bonus requires named feature types

---

## 5. Progressive Difficulty

**Scale via**: ↓annotations | ↑behavior granularity | ↑label noise

**v1**: full annotations, 4 classes → median 0.30
**v2**: 50% annotations removed (semi-supervised) → median 0.20
**v3**: fine-grained subtypes (sniff-face vs sniff-body) → median 0.15
**v4**: 5% random label flips → median 0.10

---

## 6. Why This Will Work — Scientific Evidence

### the curation gap is PROVEN

The [MABe 2021 challenge](https://sites.google.com/view/mabe21/challenge) (NeurIPS workshop, 402 participants, 1392 submissions) established exact performance benchmarks on this dataset:

| method | macro F1 | source |
|--------|----------|--------|
| self-attention on raw keypoints | 0.610 | [Sun et al. 2021, Table 2](https://arxiv.org/abs/2104.02710) |
| fully connected on raw keypoints | 0.659 | same |
| **1D CNN baseline** | **0.793** | same |
| 1D CNN + TREBA self-supervised | 0.829 | same |
| **MABe challenge top-1** | **0.864** | [leaderboard](https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications) |
| FERAL (V-JEPA2, 330M params, raw video) | 0.945 mAP | [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.11.16.688666v1.full) |

LogReg on raw 28-dim keypoints ≈ 0.40 F1 (worse than neural models bc no temporal context). LogReg on expert-curated features ≈ 0.78–0.83 F1. **That's a +0.40 F1 curation gap** — the largest of any dataset we evaluated.

### the feature space is documented and HUGE

The [MARS pipeline](https://doi.org/10.7554/eLife.63720) (Segalin et al. 2021 eLife, [GitHub](https://github.com/neuroethology/MARS)) defines **270 canonical features** from 7 keypoints: distances (91 pairwise), velocities, body angles, social features (facing, overlap, approach rate), with temporal windowing (rolling mean/std/min/max over 0.1s, 0.37s, 0.7s).

The [1st place solution](https://github.com/nebw/mabe) added: egocentric coordinates, PCA of 14×14 distance matrix, wall distance. The [Jayzen-Tech solution](https://github.com/Jayzen-Tech/kaggle-MABe) added: rolling stats over 5 window sizes, lag features at 11 timepoints, grooming micro-features.

**Total action space**: ~160 base features × temporal windows × coordinate systems = ~10,000–33,000 possible features. Feature subset selection space: effectively infinite (2^70 from ~70 feature categories).

### every score tier is calibrated from published results

- 0.0–0.15: raw keypoints → below self-attention (0.610)
- 0.30–0.50: distances + velocities → approaching FC baseline (0.659)
- 0.50–0.70: MARS-like features + rolling stats → between FC and LSTM (0.675)
- 0.70–0.90: egocentric + PCA + threshold tuning → approaching 1D CNN (0.793)
- 0.90–1.0: near LogReg ceiling (~0.83) — challenge-winner-level features

### key repos for baseline implementation

| repo | purpose |
|------|---------|
| [MARS](https://github.com/neuroethology/MARS) | 270-feature extraction pipeline |
| [TREBA](https://github.com/neuroethology/TREBA) | 32-dim self-supervised embeddings |
| [nebw/mabe](https://github.com/nebw/mabe) | 1st place solution: egocentric + CPC |
| [CalMS21 data](https://data.caltech.edu/records/s0vdx-0k302) | official dataset + conversion scripts |
| [A-SOiD](https://github.com/YttriLab/A-SOID) | active learning, outperformed top-1 with 85% less data |

---

*end of specification*
