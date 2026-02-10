# E01: AJILE12 Multi-Modal Neural Data Curation -- Complete pm_env Specification

> complete implementation spec for E01 pm_env: containerized RL environment for data curation
> pattern: student curates data -> system trains downstream model -> score = downstream performance
> created: 2026-02-09 | author: claude opus 4.6

---

## 1. Environment Overview

### 1.1 What Is pm_env

pm_env is a containerized RL environment for training AI agents on ML tasks.

| Property | Value |
|----------|-------|
| Container runtime | Podman |
| Visible to student | `env_data/` |
| Hidden from student | `scoring_data/` |
| Student interface | MCP tools (`bash`, `submit_answers`) |
| Network access | NONE inside container |
| Hardware | CPU only |
| Time limit | 4 hours cumulative |
| Max tool call | 10 minutes |
| Data pre-downloaded | via `setup_data.py` |

### 1.2 Core Pattern: Data Curation

The student does NOT train models. The student filters, cleans, aligns, and engineers features from messy data. The SYSTEM then takes the student's curated output and trains a fixed downstream model (e.g., logistic regression). The student's score = how well that downstream model performs.

### 1.3 Task Structure

| Step | Title | Judge Type | Weight |
|------|-------|------------|--------|
| 1 | Data Exploration & Schema Discovery | ExactMatchJudge | ~10% |
| 2 | Multi-Rate Data Alignment | ExecutableJudge | ~25% |
| 3 | Feature Engineering for Neural Decoding | ExecutableJudge | ~25% |
| 4 | Cross-Subject Generalization | ExecutableJudge | ~40% |

### 1.4 Reward Scale (continuous)

| Score | Meaning |
|-------|---------|
| 0.0 | failure |
| 0.2 | median agent performance |
| 0.8 | human-level |
| >1.0 | possible (exceptional) |

### 1.5 Security Constraints

- Student MUST NOT see test data or report own score (reward hacking)
- Student MUST NOT know exact scoring formula
- Hidden test session lives in `scoring_data/` only

---

## 2. Dataset: AJILE12

### 2.1 Source

| Resource | URL |
|----------|-----|
| DANDI Archive | https://dandiarchive.org/dandiset/000055 |
| Dataset paper | Peterson et al. 2022, *Scientific Data* 9:184 |
| Methods paper | Singh et al. 2021, *J Neurosci Methods* 358:109199 |
| Analysis paper | Peterson et al. 2021, *eNeuro* 8(3) |
| AAAI paper | Wang et al. 2018, *AAAI* 32(1) |

### 2.2 Dataset Properties

| Property | Value |
|----------|-------|
| Subjects | 12 human epilepsy patients with implanted ECoG |
| Recording duration | 55 days continuous per subject |
| ECoG sampling rate | 500 Hz |
| ECoG channels | 64+ per subject (varies) |
| ECoG dtype | float32 |
| Pose tracking rate | 30 Hz (video) |
| Pose keypoints | 9 upper-body |
| Pose dtype | float64 |
| Events | reach annotations with start/stop times, angles, magnitudes |
| Electrode metadata | MNI coordinates, quality flags, frequency band R-squared values |
| Format | NWB (HDF5-based), accessed via pynwb |

### 2.3 NWB File Structure

```
/
+-- acquisition/
|   +-- ElectricalSeries/
|       +-- data                    # [n_samples, n_channels], float32
|       +-- timestamps              # [n_samples], float64 (optional)
|       +-- starting_time           # float64 (seconds from session start)
|       +-- rate                    # float64 (500.0 Hz)
|       +-- electrodes              # DynamicTableRegion -> electrodes table
|
+-- processing/
|   +-- behavior/
|       +-- Position/
|           +-- SpatialSeries/
|               +-- data            # [n_frames, 9, 2] or separate per keypoint
|               +-- timestamps      # [n_frames], 30 Hz
|               +-- reference_frame # "video pixel coordinates"
|               +-- unit            # "pixels"
|
+-- intervals/
|   +-- reaches/
|   |   +-- start_time              # movement onset (seconds)
|   |   +-- stop_time               # movement offset (seconds)
|   |   +-- reach_magnitude         # pixels
|   |   +-- reach_duration          # seconds
|   |   +-- reach_angle             # radians (2D vertical angle)
|   |   +-- onset_speed             # pixels/second
|   |   +-- day_of_recording        # int
|   |   +-- time_of_day             # seconds from midnight
|   |   +-- speech_ratio            # float (0-1)
|   |   +-- ipsi_contra_ratio       # float
|   |   +-- temporal_overlap        # float
|   |   +-- bimanual                # bool
|   |
|   +-- epochs/
|       +-- start_time, stop_time, tags
|
+-- general/
    +-- extracellular_ephys/
    |   +-- electrodes/
    |   |   +-- x, y, z             # MNI coordinates (mm)
    |   |   +-- group_name          # clinical array label
    |   |   +-- location            # anatomical region
    |   |   +-- std                 # voltage std
    |   |   +-- kurtosis            # voltage kurtosis
    |   |   +-- mad                 # median absolute deviation
    |   |   +-- good                # bool quality flag
    |   |   +-- lfb_r2              # low-freq band R-squared (8-32 Hz)
    |   |   +-- hfb_r2              # high-freq band R-squared (76-100 Hz)
    |   |
    |   +-- [electrode_groups]/
    |
    +-- subject/
        +-- subject_id, age, sex, species
```

### 2.4 Pose Keypoints (9 upper-body, DeepLabCut-derived)

| Index | Keypoint |
|-------|----------|
| 0 | nose |
| 1 | left_ear |
| 2 | right_ear |
| 3 | left_shoulder |
| 4 | right_shoulder |
| 5 | left_elbow |
| 6 | right_elbow |
| 7 | left_wrist |
| 8 | right_wrist |

### 2.5 Data Messiness (why this is hard)

1. **Multi-rate**: ECoG at 500 Hz, pose at 30 Hz -- must align
2. **Bad electrodes**: some flagged as bad, some have drift, student must identify and exclude
3. **Missing keypoints**: video tracking failures produce NaN values in pose data
4. **Session variability**: different subjects have different electrode counts, positions
5. **Temporal drift**: over long recordings, timestamps can drift slightly
6. **Real clinical data**: artifacts from patient movement, electrical interference

### 2.6 What to Pre-Download (env_data vs scoring_data)

| Subject | Session | Destination | Duration | Notes |
|---------|---------|-------------|----------|-------|
| sub-01 | ses-1 | env_data/ | 5 min clip | training |
| sub-01 | ses-2 | env_data/ | 5 min clip | training |
| sub-01 | ses-3 | env_data/ | 5 min clip | training |
| sub-02 | ses-1 | env_data/ | 5 min clip | training |
| sub-02 | ses-2 | env_data/ | 5 min clip | training |
| sub-03 | ses-1 | env_data/ | 5 min clip | training |
| sub-03 | ses-3 | env_data/ (data only, NO reaches.csv) | 5 min clip | test features only |
| sub-03 | ses-3 | scoring_data/ (FULL, with reaches.csv) | 5 min clip | hidden test |

**Total estimated size**: ~2-3 GB

**CRITICAL**: sub-03 ses-3 in env_data has ECoG, pose, electrodes, and metadata but NO reaches.csv (movement labels). The student can compute features for this session but cannot see the true labels. The full version with labels lives in scoring_data only.

---

## 3. File Layout Inside Container

```
/env_data/
+-- sub-01/
|   +-- ses-1/
|   |   +-- ecog.npy              # [n_samples, n_channels], float32
|   |   +-- ecog_rate.npy         # scalar: 500.0
|   |   +-- pose.npy              # [n_frames, 9, 2], float64
|   |   +-- pose_rate.npy         # scalar: 30.0
|   |   +-- electrodes.csv        # x, y, z, good, lfb_r2, hfb_r2, ...
|   |   +-- reaches.csv           # start_time, stop_time, ...
|   |   +-- metadata.json         # subject_id, session_id, rates, shapes, ...
|   +-- ses-2/
|   |   +-- (same structure)
|   +-- ses-3/
|       +-- (same structure)
+-- sub-02/
|   +-- ses-1/
|   +-- ses-2/
+-- sub-03/
    +-- ses-1/
    |   +-- (same structure, WITH reaches.csv)
    +-- ses-3/
        +-- ecog.npy
        +-- ecog_rate.npy
        +-- pose.npy
        +-- pose_rate.npy
        +-- electrodes.csv
        +-- metadata.json
        +-- (NO reaches.csv -- labels hidden!)

/scoring_data/
+-- sub-03/
|   +-- ses-3/
|       +-- ecog.npy
|       +-- ecog_rate.npy
|       +-- pose.npy
|       +-- pose_rate.npy
|       +-- electrodes.csv
|       +-- reaches.csv           # TRUE labels (hidden from student)
|       +-- metadata.json
+-- reference_alignment.csv       # reference for Step 2 scoring
+-- reference_features.npz        # reference for Step 3 scoring
+-- true_test_labels.npz          # true labels for Step 4 scoring
+-- score_alignment.py            # Step 2 scoring script
+-- score_features.py             # Step 3 scoring script
+-- score_curation.py             # Step 4 scoring script

/workdir/
    (student writes output files here)
```

---

## 4. Step 1: Data Exploration & Schema Discovery

### 4.1 Instructions to Student

```
You have access to neural recording data in {env_data_dir}/. The data comes from the
AJILE12 dataset -- human intracranial ECoG recordings with simultaneous pose tracking.

The data files are organized by subject and session. Explore the data and answer the
following questions using the submit_answers tool:

1. "num_subjects": How many subjects are in the dataset?
2. "num_sessions_sub01": How many sessions does subject 01 have?
3. "ecog_rate_hz": What is the ECoG sampling rate in Hz?
4. "pose_rate_hz": What is the pose tracking rate in Hz?
5. "num_keypoints": How many pose keypoints are tracked?
6. "num_electrodes_sub01": How many electrodes does subject 01 have?
7. "num_bad_electrodes_sub01": How many electrodes are flagged as bad for subject 01?
8. "keypoint_names": Comma-separated list of keypoint names (alphabetical order)

Take time to carefully explore the file structure and data formats before answering.
```

### 4.2 Judge

**Type**: ExactMatchJudge

**Expected answers** (computed from actual data in setup_data.py; values below are representative):

```python
step1_answers = {
    "num_subjects": "3",
    "num_sessions_sub01": "3",
    "ecog_rate_hz": "500",
    "pose_rate_hz": "30",
    "num_keypoints": "9",
    "num_electrodes_sub01": "64",       # actual value from downloaded data
    "num_bad_electrodes_sub01": "8",    # actual value from downloaded data
    "keypoint_names": "left_ear,left_elbow,left_shoulder,left_wrist,nose,right_ear,right_elbow,right_shoulder,right_wrist",
}
```

**Score**: fraction of correct answers. 0/8 = 0.0, 8/8 = 1.0.

### 4.3 Anti-Gaming

- Answers are data-dependent (differ per subject configuration). Can't guess.
- `num_electrodes_sub01` and `num_bad_electrodes_sub01` require actually loading and inspecting the electrode metadata.
- `keypoint_names` must be alphabetically sorted with exact comma separation.

### 4.4 Expected Distribution

| Performance | Score | What agents typically get wrong |
|-------------|-------|---------------------------------|
| Weak | 0.0-0.4 | Wrong subject count, wrong rates |
| Median | 0.5-0.75 | Keypoint names (order), bad electrode count |
| Strong | 0.875-1.0 | Gets 7-8/8 correct |

---

## 5. Step 2: Multi-Rate Data Alignment

### 5.1 Instructions to Student

```
Now align the ECoG and pose data for subject 01, session 1.

Write a Python script that:
1. Loads the ECoG data (500 Hz) for the time window [60.0, 360.0] seconds
2. Loads the pose data (30 Hz) for the same time window
3. Downsamples ECoG to 30 Hz (matching pose rate) using proper anti-aliasing
4. Excludes electrodes flagged as bad
5. Handles any missing pose keypoints (interpolation or flagging)
6. Saves the aligned data as: /workdir/aligned_data.csv

The CSV must have columns:
- timestamp: float64, seconds, at 30 Hz
- ecog_0, ecog_1, ...: float32, one per good electrode
- nose_x, nose_y, left_ear_x, left_ear_y, ...: float64
- is_movement: bool, True during annotated reach events
```

### 5.2 Judge

**Type**: ExecutableJudge

**continue_threshold**: 0.2 (student proceeds to Step 3 even with mediocre alignment)

**Subprocess args**:
```python
[
    sys.executable,
    f"{scoring_dir}/score_alignment.py",
    "/workdir/aligned_data.csv",
    f"{scoring_dir}/reference_alignment.csv",
    "alignment_results.json",
]
```

### 5.3 Scoring Script: score_alignment.py

```python
import numpy as np
import pandas as pd
import json
import sys

def score_alignment(submitted_path, reference_path, output_path):
    try:
        sub = pd.read_csv(submitted_path)
        ref = pd.read_csv(reference_path)
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": str(e)}}, open(output_path, "w"))
        return

    scores = {}

    # 1. timestamp alignment (35% weight)
    if "timestamp" not in sub.columns:
        scores["timestamp"] = 0.0
    else:
        ref_ts = ref["timestamp"].values
        sub_ts = sub["timestamp"].values
        if len(sub_ts) == 0:
            scores["timestamp"] = 0.0
        else:
            diffs = []
            for t in ref_ts:
                idx = np.argmin(np.abs(sub_ts - t))
                diffs.append(abs(sub_ts[idx] - t))
            diffs = np.array(diffs)
            scores["timestamp"] = float(np.mean(diffs < 0.0334))

    # 2. ECoG value correlation (30% weight)
    ecog_cols_ref = [c for c in ref.columns if c.startswith("ecog_")]
    ecog_cols_sub = [c for c in sub.columns if c.startswith("ecog_")]
    if len(ecog_cols_sub) == 0 or len(ecog_cols_ref) == 0:
        scores["ecog"] = 0.0
    else:
        common = set(ecog_cols_ref) & set(ecog_cols_sub)
        if len(common) == 0:
            scores["ecog"] = 0.0
        else:
            corrs = []
            min_len = min(len(sub), len(ref))
            for col in common:
                r = np.corrcoef(ref[col].values[:min_len], sub[col].values[:min_len])[0, 1]
                if np.isnan(r):
                    r = 0.0
                corrs.append(max(0, r))
            scores["ecog"] = float(np.mean(corrs))
            n_extra = len(ecog_cols_sub) - len(ecog_cols_ref)
            if n_extra > 0:
                scores["ecog"] *= 0.8

    # 3. pose value accuracy (20% weight)
    pose_cols_ref = [c for c in ref.columns if any(c.endswith(s) for s in ["_x", "_y"])]
    pose_cols_sub = [c for c in sub.columns if any(c.endswith(s) for s in ["_x", "_y"])]
    common_pose = set(pose_cols_ref) & set(pose_cols_sub)
    if len(common_pose) == 0:
        scores["pose"] = 0.0
    else:
        corrs = []
        min_len = min(len(sub), len(ref))
        for col in common_pose:
            sub_vals = sub[col].values[:min_len]
            ref_vals = ref[col].values[:min_len]
            mask = ~(np.isnan(sub_vals) | np.isnan(ref_vals))
            if mask.sum() < 10:
                corrs.append(0.0)
            else:
                r = np.corrcoef(ref_vals[mask], sub_vals[mask])[0, 1]
                corrs.append(max(0, float(r)) if not np.isnan(r) else 0.0)
        scores["pose"] = float(np.mean(corrs))

    # 4. movement label accuracy (15% weight)
    if "is_movement" not in sub.columns:
        scores["movement"] = 0.0
    else:
        min_len = min(len(sub), len(ref))
        sub_mov = sub["is_movement"].values[:min_len].astype(bool)
        ref_mov = ref["is_movement"].values[:min_len].astype(bool)
        scores["movement"] = float(np.mean(sub_mov == ref_mov))

    # final weighted score
    weights = {"timestamp": 0.35, "ecog": 0.30, "pose": 0.20, "movement": 0.15}
    final = sum(scores[k] * weights[k] for k in weights)

    result = {"score": round(final, 4), "metadata": {
        "component_scores": scores,
        "num_rows_submitted": len(sub),
        "num_rows_reference": len(ref),
        "ecog_cols_found": len(ecog_cols_sub),
        "ecog_cols_expected": len(ecog_cols_ref),
    }}
    json.dump(result, open(output_path, "w"))

if __name__ == "__main__":
    score_alignment(sys.argv[1], sys.argv[2], sys.argv[3])
```

### 5.4 Anti-Gaming

- Reference alignment uses a proven implementation (scipy.signal.decimate with FIR anti-aliasing)
- ECoG correlation catches wrong downsampling method (naive slicing vs proper anti-alias)
- Bad electrode inclusion is penalized (0.8x multiplier for extra ecog columns)
- Data-dependent: different subjects have different electrode counts

### 5.5 Expected Distribution

| Performance | Score | Common failure |
|-------------|-------|----------------|
| Weak | 0.2-0.35 | Wrong downsampling (no anti-alias), wrong time window |
| Median | 0.4-0.6 | Includes bad electrodes, imprecise timestamps |
| Strong | 0.7-0.95 | Correct anti-alias, correct electrode filtering, good NaN handling |

---

## 6. Step 3: Feature Engineering for Neural Decoding

### 6.1 Instructions to Student

```
Engineer features from the neural + pose data for predicting movement.

Using all subjects and sessions in {env_data_dir}/, create a feature matrix.
Save to /workdir/features.npz containing:
- "X": numpy array [n_samples, n_features]
- "y": numpy array [n_samples], binary (1=movement, 0=no movement)
- "feature_names": array of feature name strings
- "subject_ids": numpy array [n_samples], subject ID per sample

Feature ideas:
- ECoG band power: delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz),
  beta (13-30 Hz), gamma (30-100 Hz)
- Pose velocity and acceleration
- Inter-keypoint distances
- Rolling statistics

IMPORTANT: Think carefully about the Nyquist theorem. If your aligned data is at
30 Hz, what is the maximum frequency you can represent? For higher frequency bands,
you may need to use the original 500 Hz ECoG data.
```

### 6.2 Judge

**Type**: ExecutableJudge

**continue_threshold**: 0.1

**Subprocess args**:
```python
[
    sys.executable,
    f"{scoring_dir}/score_features.py",
    "/workdir/features.npz",
    f"{scoring_dir}/reference_features.npz",
    "features_results.json",
]
```

### 6.3 Scoring Script: score_features.py

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import json
import sys

def score_features(features_path, reference_features_path, output_path):
    try:
        data = np.load(features_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        subject_ids = data["subject_ids"]
        feature_names = list(data["feature_names"])
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": f"Failed to load: {e}"}},
                  open(output_path, "w"))
        return

    if X.shape[0] != len(y) or X.shape[0] != len(subject_ids):
        json.dump({"score": 0.0, "metadata": {"error": "Shape mismatch"}},
                  open(output_path, "w"))
        return

    if X.shape[0] < 100:
        json.dump({"score": 0.0, "metadata": {"error": "Too few samples"}},
                  open(output_path, "w"))
        return

    # check for constant or NaN features
    valid_features = []
    for i in range(X.shape[1]):
        col = X[:, i]
        if np.all(np.isnan(col)) or np.std(col[~np.isnan(col)]) < 1e-10:
            continue
        valid_features.append(i)

    if len(valid_features) == 0:
        json.dump({"score": 0.0, "metadata": {"error": "No valid features"}},
                  open(output_path, "w"))
        return

    X = X[:, valid_features]
    X = np.nan_to_num(X, nan=0.0)

    # split: train on subjects 1,2; test on subject 3
    train_mask = np.isin(subject_ids, [1, 2])
    test_mask = subject_ids == 3

    if test_mask.sum() < 10:
        n = len(y)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:int(0.8 * n)] = True
        test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    try:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5

    # map AUC to score: 0.5 -> 0.0, 0.7 -> 0.3, 0.85 -> 0.7, 0.95 -> 1.0
    if auc <= 0.5:
        auc_score = 0.0
    elif auc <= 0.7:
        auc_score = (auc - 0.5) / 0.2 * 0.3
    elif auc <= 0.85:
        auc_score = 0.3 + (auc - 0.7) / 0.15 * 0.4
    else:
        auc_score = 0.7 + (auc - 0.85) / 0.10 * 0.3
    auc_score = min(auc_score, 1.0)

    # feature quality check
    ref_data = np.load(reference_features_path, allow_pickle=True)
    student_names = set(feature_names)

    has_band_power = any(
        "band" in n or "power" in n or "delta" in n or "theta" in n
        or "alpha" in n or "beta" in n or "gamma" in n
        for n in student_names
    )
    has_velocity = any("velocity" in n or "speed" in n for n in student_names)
    has_distance = any("distance" in n or "dist" in n for n in student_names)
    has_high_freq = any("gamma" in n or "beta" in n for n in student_names)

    feature_type_score = (
        has_band_power * 0.3 +
        has_velocity * 0.2 +
        has_distance * 0.2 +
        has_high_freq * 0.3
    )

    final_score = 0.6 * auc_score + 0.4 * feature_type_score

    result = {"score": round(final_score, 4), "metadata": {
        "auc": round(auc, 4),
        "auc_score": round(auc_score, 4),
        "feature_type_score": round(feature_type_score, 4),
        "num_features": X.shape[1],
        "num_valid_features": len(valid_features),
        "num_train": int(train_mask.sum()),
        "num_test": int(test_mask.sum()),
        "has_band_power": has_band_power,
        "has_velocity": has_velocity,
        "has_distance": has_distance,
        "has_high_freq": has_high_freq,
    }}
    json.dump(result, open(output_path, "w"))

if __name__ == "__main__":
    score_features(sys.argv[1], sys.argv[2], sys.argv[3])
```

### 6.4 KEY TRAP: Nyquist Theorem

The instructions hint that beta (13-30 Hz) and gamma (30-100 Hz) CANNOT be computed from 30 Hz downsampled data (Nyquist limit = 15 Hz). The student MUST go back to the raw 500 Hz ECoG data for high-frequency band power features.

This tests whether the agent REASONS about signal processing constraints rather than blindly applying a recipe.

**Scoring impact**:
- `has_high_freq` check rewards agents who claim to compute beta/gamma features (0.3 weight in feature_type_score)
- But the downstream AUC will be low if those features are garbage (computed from 30 Hz data, yielding aliased nonsense)
- Agents that correctly use raw 500 Hz data for high-freq bands get BOTH the feature_type_score bonus AND higher downstream AUC

### 6.5 Anti-Gaming

- Downstream model is FIXED (logistic regression, C=1.0) -- student can't choose model
- Test subject (03) is partially hidden (session 3 labels in scoring_data only)
- Constant predictions yield AUC = 0.5, mapped to score = 0.0
- Feature quality checked independently of downstream performance
- Both AUC and feature type contribute to final score

### 6.6 Expected Distribution

| Performance | Score | What differentiates |
|-------------|-------|---------------------|
| Weak | 0.1-0.25 | Only raw features, no band power |
| Median | 0.3-0.5 | Has band power but misses Nyquist issue |
| Strong | 0.6-0.85 | Uses 500 Hz for beta/gamma, good pose features |

---

## 7. Step 4: Cross-Subject Generalization (FINAL)

### 7.1 Instructions to Student

```
Final challenge: build the best data curation pipeline for cross-subject neural decoding.

You have data from subjects 01, 02, and 03 (session 1). Your curated data trains a
decoder evaluated on a HIDDEN test session.

Save to /workdir/curated_output.npz:
- "X_train": [n_train, n_features] -- curated training features
- "y_train": [n_train] -- training labels
- "X_test": [n_test, n_features] -- features for test session
  (Test data: {env_data_dir}/sub-03/ses-3/)
- "y_test": [n_test] -- your PREDICTED labels
- "feature_pipeline": string describing your approach

Strategies to consider:
- Cross-subject normalization (z-score per subject)
- Feature selection / dimensionality reduction
- Outlier removal
- Oversampling rare movement events
- Band power from raw 500 Hz for high-frequency features

HINT: Cross-subject generalization is hard because electrode placement differs.
Focus on features robust across subjects.
```

### 7.2 Judge

**Type**: ExecutableJudge

**continue_threshold**: 0.0 (final step)

**Subprocess args**:
```python
[
    sys.executable,
    f"{scoring_dir}/score_curation.py",
    "/workdir/curated_output.npz",
    f"{scoring_dir}/true_test_labels.npz",
    "curation_results.json",
]
```

### 7.3 Scoring Script: score_curation.py

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import json
import sys

def score_curation(curated_path, true_labels_path, output_path):
    try:
        data = np.load(curated_path, allow_pickle=True)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test_student = data["y_test"]
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": str(e)}},
                  open(output_path, "w"))
        return

    true_labels = np.load(true_labels_path)["y"]

    # allow slight shape mismatch, truncate to shorter
    if X_test.shape[0] != len(true_labels):
        min_n = min(X_test.shape[0], len(true_labels))
        X_test = X_test[:min_n]
        true_labels = true_labels[:min_n]
        y_test_student = y_test_student[:min_n]

    # anti-gaming: check for degenerate submissions
    if np.std(y_test_student) < 1e-6:
        json.dump({"score": 0.0, "metadata": {"error": "Constant predictions"}},
                  open(output_path, "w"))
        return

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # remove constant features
    stds = np.std(X_train, axis=0)
    valid = stds > 1e-10
    X_train = X_train[:, valid]
    X_test = X_test[:, valid]

    if X_train.shape[1] == 0:
        json.dump({"score": 0.0, "metadata": {"error": "No valid features after filtering"}},
                  open(output_path, "w"))
        return

    # part 1: train downstream model on student's curated data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    try:
        model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        model.fit(X_train_s, y_train)
        y_prob = model.predict_proba(X_test_s)[:, 1]
        downstream_auc = roc_auc_score(true_labels, y_prob)
    except Exception:
        downstream_auc = 0.5

    # part 2: evaluate student's own predictions
    try:
        student_auc = roc_auc_score(true_labels, y_test_student)
    except Exception:
        student_auc = 0.5

    # map AUCs to scores
    def map_auc(auc):
        if auc <= 0.5:
            return 0.0
        if auc <= 0.7:
            return (auc - 0.5) / 0.2 * 0.3
        if auc <= 0.85:
            return 0.3 + (auc - 0.7) / 0.15 * 0.4
        return min(0.7 + (auc - 0.85) / 0.10 * 0.3, 1.0)

    downstream_score = map_auc(downstream_auc)
    student_score = map_auc(student_auc)

    final = 0.5 * downstream_score + 0.5 * student_score

    result = {"score": round(final, 4), "metadata": {
        "downstream_auc": round(downstream_auc, 4),
        "student_auc": round(student_auc, 4),
        "downstream_score": round(downstream_score, 4),
        "student_score": round(student_score, 4),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X_train.shape[1],
    }}
    json.dump(result, open(output_path, "w"))

if __name__ == "__main__":
    score_curation(sys.argv[1], sys.argv[2], sys.argv[3])
```

### 7.4 Anti-Gaming

- Test session (sub-03 ses-3) is hidden in scoring_data
- Student must provide features for test data but doesn't know true labels
- Constant predictions = score 0.0 (explicit check)
- Downstream model is FIXED (logistic regression, C=1.0)
- Two independent evaluation paths:
  - downstream model trained on student's curated features
  - student's own predictions compared to true labels

### 7.5 Expected Distribution

| Performance | Score | What differentiates |
|-------------|-------|---------------------|
| Weak | 0.1-0.2 | No cross-subject normalization, poor features |
| Median | 0.2-0.4 | Some features work, but generalization is poor |
| Strong | 0.5-0.7 | Nyquist-aware features + cross-subject normalization + movement oversampling |

---

## 8. Complete tasks.py

```python
from functools import partial
from textwrap import dedent
import sys

from pm_env.config import EvaluationRunConfig
from pm_env.judges.exact_match_judge import ExactMatchJudge
from pm_env.judges.executable_judge import ExecutableJudge
from pm_env.get_data_dir import get_env_data_dir, get_scoring_data_dir
from pm_env.schemas.task import Step, Task


def configure_tools(task_id: str):
    if task_id == "ajile12-curation":
        import pm_env.configs.tool_config as tool_config
        tool_config.BashTimedConfig.CUMULATIVE_TIME_LIMIT = 14400  # 4 hours
        tool_config.BashTimedConfig.TIMEOUT_S = 600  # 10 min per call


def pre_hook(task: Task, random_seed: int) -> dict:
    return {"task": task.id, "random_seed": random_seed}


def get_tasks(config: EvaluationRunConfig) -> list[Task]:
    env_dir = get_env_data_dir()
    scoring_dir = get_scoring_data_dir()

    # expected answers for Step 1 (values computed from actual data in setup_data.py)
    step1_answers = {
        "num_subjects": "3",
        "num_sessions_sub01": "3",
        "ecog_rate_hz": "500",
        "pose_rate_hz": "30",
        "num_keypoints": "9",
        "num_electrodes_sub01": "64",
        "num_bad_electrodes_sub01": "8",
        "keypoint_names": "left_ear,left_elbow,left_shoulder,left_wrist,nose,right_ear,right_elbow,right_shoulder,right_wrist",
    }

    return [
        Task(
            id="ajile12-curation",
            tools=["bash", "submit_answers"],
            pre_hook=partial(pre_hook, random_seed=config.random_seed),
            steps=[
                # Step 1: Data Exploration & Schema Discovery
                Step(
                    instructions=dedent(f"""\
                    You have access to neural recording data in {env_dir}/. The data comes from the AJILE12 dataset -- human intracranial ECoG recordings with simultaneous pose tracking.

                    The data files are organized by subject and session. Explore the data and answer the following questions using the submit_answers tool:

                    1. "num_subjects": How many subjects are in the dataset?
                    2. "num_sessions_sub01": How many sessions does subject 01 have?
                    3. "ecog_rate_hz": What is the ECoG sampling rate in Hz?
                    4. "pose_rate_hz": What is the pose tracking rate in Hz?
                    5. "num_keypoints": How many pose keypoints are tracked?
                    6. "num_electrodes_sub01": How many electrodes does subject 01 have?
                    7. "num_bad_electrodes_sub01": How many electrodes are flagged as bad for subject 01?
                    8. "keypoint_names": Comma-separated list of keypoint names (alphabetical order)

                    Take time to carefully explore the file structure and data formats before answering.
                    """),
                    judge=ExactMatchJudge(expected=step1_answers),
                ),

                # Step 2: Multi-Rate Data Alignment
                Step(
                    instructions=dedent(f"""\
                    Now align the ECoG and pose data for subject 01, session 1.

                    Write a Python script that:
                    1. Loads the ECoG data (500 Hz) for the time window [60.0, 360.0] seconds
                    2. Loads the pose data (30 Hz) for the same time window
                    3. Downsamples ECoG to 30 Hz (matching pose rate) using proper anti-aliasing
                    4. Excludes electrodes flagged as bad
                    5. Handles any missing pose keypoints (interpolation or flagging)
                    6. Saves the aligned data as: /workdir/aligned_data.csv

                    The CSV must have columns:
                    - timestamp: float64, seconds, at 30 Hz
                    - ecog_0, ecog_1, ...: float32, one per good electrode
                    - nose_x, nose_y, left_ear_x, left_ear_y, ...: float64
                    - is_movement: bool, True during annotated reach events
                    """),
                    judge=ExecutableJudge(
                        subprocess_run_args=[
                            sys.executable,
                            f"{scoring_dir}/score_alignment.py",
                            "/workdir/aligned_data.csv",
                            f"{scoring_dir}/reference_alignment.csv",
                            "alignment_results.json",
                        ],
                        continue_threshold=0.2,
                    ),
                ),

                # Step 3: Feature Engineering for Neural Decoding
                Step(
                    instructions=dedent(f"""\
                    Engineer features from the neural + pose data for predicting movement.

                    Using all subjects and sessions in {env_dir}/, create a feature matrix.
                    Save to /workdir/features.npz containing:
                    - "X": numpy array [n_samples, n_features]
                    - "y": numpy array [n_samples], binary (1=movement, 0=no movement)
                    - "feature_names": array of feature name strings
                    - "subject_ids": numpy array [n_samples], subject ID per sample

                    Feature ideas:
                    - ECoG band power: delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-100 Hz)
                    - Pose velocity and acceleration
                    - Inter-keypoint distances
                    - Rolling statistics

                    IMPORTANT: Think carefully about the Nyquist theorem. If your aligned data is at 30 Hz, what is the maximum frequency you can represent? For higher frequency bands, you may need to use the original 500 Hz ECoG data.
                    """),
                    judge=ExecutableJudge(
                        subprocess_run_args=[
                            sys.executable,
                            f"{scoring_dir}/score_features.py",
                            "/workdir/features.npz",
                            f"{scoring_dir}/reference_features.npz",
                            "features_results.json",
                        ],
                        continue_threshold=0.1,
                    ),
                ),

                # Step 4: Cross-Subject Generalization (FINAL)
                Step(
                    instructions=dedent(f"""\
                    Final challenge: build the best data curation pipeline for cross-subject neural decoding.

                    You have data from subjects 01, 02, and 03 (session 1). Your curated data trains a decoder evaluated on a HIDDEN test session.

                    Save to /workdir/curated_output.npz:
                    - "X_train": [n_train, n_features] -- curated training features
                    - "y_train": [n_train] -- training labels
                    - "X_test": [n_test, n_features] -- features for test session
                      (Test data: {env_dir}/sub-03/ses-3/)
                    - "y_test": [n_test] -- your PREDICTED labels
                    - "feature_pipeline": string describing your approach

                    Strategies to consider:
                    - Cross-subject normalization (z-score per subject)
                    - Feature selection / dimensionality reduction
                    - Outlier removal
                    - Oversampling rare movement events
                    - Band power from raw 500 Hz for high-frequency features

                    HINT: Cross-subject generalization is hard because electrode placement differs. Focus on features robust across subjects.
                    """),
                    judge=ExecutableJudge(
                        subprocess_run_args=[
                            sys.executable,
                            f"{scoring_dir}/score_curation.py",
                            "/workdir/curated_output.npz",
                            f"{scoring_dir}/true_test_labels.npz",
                            "curation_results.json",
                        ],
                        continue_threshold=0.0,
                    ),
                ),
            ],
        ),
    ]
```

---

## 9. Complete setup_data.py

```python
# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "dandi>=0.60",
#     "pynwb>=2.6",
#     "h5py>=3.10",
#     "remfile>=0.1",
#     "numpy>=1.26",
#     "pandas>=2.1",
#     "scipy>=1.12",
# ]
# ///

"""
Download AJILE12 subset from DANDI and prepare env_data + scoring_data.

Subjects: 01, 02, 03
Sessions: 3 per sub-01, 2 per sub-02, 3 per sub-03
Subject 03 session 3: scoring_data ONLY (hidden test)

All data is pre-extracted from NWB into numpy arrays + CSV for reliability
inside the container (no network access at runtime).
"""

import os
import shutil
import json
import numpy as np
import pandas as pd
from dandi.dandiapi import DandiAPIClient
import remfile
import h5py
from scipy.signal import decimate

DANDISET_ID = "000055"
SUBJECTS = ["01", "02", "03"]
SESSIONS_PER_SUBJECT = {"01": [1, 2, 3], "02": [1, 2], "03": [1, 3]}
CLIP_DURATION = 300.0  # 5 minutes
CLIP_START = 60.0  # skip first minute (warmup/settling)

KEYPOINTS = [
    "nose", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]


def get_s3_url(subject_id: str, session_id: int) -> str:
    filename = f"sub-{subject_id}/sub-{subject_id}_ses-{session_id}_behavior+ecephys.nwb"
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(DANDISET_ID)
        asset = dandiset.get_asset_by_path(filename)
        return asset.get_content_url(follow_redirects=1, strip_query=True)


def download_and_extract(subject_id: str, session_id: int, output_dir: str) -> dict:
    """Download NWB via streaming and extract to numpy arrays + CSV."""

    url = get_s3_url(subject_id, session_id)
    rf = remfile.File(url)
    f = h5py.File(rf, "r")

    # --- ECoG ---
    ecog_data = f["acquisition/ElectricalSeries/data"]
    ecog_rate = float(f["acquisition/ElectricalSeries/rate"][()])

    start_idx = int(CLIP_START * ecog_rate)
    end_idx = int((CLIP_START + CLIP_DURATION) * ecog_rate)
    ecog_clip = ecog_data[start_idx:end_idx, :]

    # --- electrode metadata ---
    electrodes = {}
    elec_group = f["general/extracellular_ephys/electrodes"]
    for key in ["x", "y", "z", "good", "std", "kurtosis", "mad", "lfb_r2", "hfb_r2"]:
        if key in elec_group:
            electrodes[key] = elec_group[key][:]

    # --- pose ---
    pose_group = f["processing/behavior/Position"]
    pose_keys = list(pose_group.keys())
    pose_series = pose_group[pose_keys[0]]
    pose_data = pose_series["data"]
    pose_rate = float(pose_series["rate"][()] if "rate" in pose_series else 30.0)

    pose_start = int(CLIP_START * pose_rate)
    pose_end = int((CLIP_START + CLIP_DURATION) * pose_rate)
    pose_clip = pose_data[pose_start:pose_end]

    # --- reaches ---
    reaches = {}
    if "intervals/reaches" in f:
        reaches_group = f["intervals/reaches"]
        reaches["start_time"] = reaches_group["start_time"][:]
        reaches["stop_time"] = reaches_group["stop_time"][:]
        for col in ["reach_magnitude", "reach_duration", "reach_angle", "onset_speed"]:
            if col in reaches_group:
                reaches[col] = reaches_group[col][:]

    f.close()

    # --- save to output directory ---
    subdir = os.path.join(output_dir, f"sub-{subject_id}", f"ses-{session_id}")
    os.makedirs(subdir, exist_ok=True)

    np.save(os.path.join(subdir, "ecog.npy"), ecog_clip.astype(np.float32))
    np.save(os.path.join(subdir, "ecog_rate.npy"), np.array(ecog_rate))
    np.save(os.path.join(subdir, "pose.npy"), pose_clip.astype(np.float64))
    np.save(os.path.join(subdir, "pose_rate.npy"), np.array(pose_rate))
    pd.DataFrame(electrodes).to_csv(os.path.join(subdir, "electrodes.csv"), index=False)

    if reaches:
        pd.DataFrame(reaches).to_csv(os.path.join(subdir, "reaches.csv"), index=False)

    metadata = {
        "subject_id": subject_id,
        "session_id": session_id,
        "ecog_rate": ecog_rate,
        "pose_rate": pose_rate,
        "ecog_shape": list(ecog_clip.shape),
        "pose_shape": list(pose_clip.shape),
        "n_electrodes": ecog_clip.shape[1],
        "n_good_electrodes": int(np.array(electrodes.get("good", [])).sum()),
        "n_bad_electrodes": int((~np.array(electrodes.get("good", []))).sum()),
        "clip_start_sec": CLIP_START,
        "clip_duration_sec": CLIP_DURATION,
        "keypoints": KEYPOINTS,
    }
    with open(os.path.join(subdir, "metadata.json"), "w") as mf:
        json.dump(metadata, mf, indent=2)

    return metadata


def compute_reference_alignment(env_data_dir: str, scoring_data_dir: str):
    """Compute reference alignment for scoring Step 2."""

    subdir = os.path.join(env_data_dir, "sub-01", "ses-1")
    ecog = np.load(os.path.join(subdir, "ecog.npy"))
    ecog_rate = float(np.load(os.path.join(subdir, "ecog_rate.npy")))
    pose = np.load(os.path.join(subdir, "pose.npy"))
    pose_rate = float(np.load(os.path.join(subdir, "pose_rate.npy")))
    electrodes = pd.read_csv(os.path.join(subdir, "electrodes.csv"))

    good_mask = electrodes["good"].astype(bool).values
    ecog_good = ecog[:, good_mask]

    # downsample ECoG: 500 Hz -> 30 Hz
    # scipy.signal.decimate with FIR anti-aliasing filter
    factor = int(ecog_rate / pose_rate)
    ecog_down = decimate(ecog_good, factor, axis=0, ftype="fir")

    n_samples = min(len(ecog_down), len(pose))
    ecog_down = ecog_down[:n_samples]
    pose = pose[:n_samples]

    timestamps = np.arange(n_samples) / pose_rate + CLIP_START

    df = pd.DataFrame({"timestamp": timestamps})
    for i in range(ecog_down.shape[1]):
        df[f"ecog_{i}"] = ecog_down[:, i].astype(np.float32)

    for j, kp in enumerate(KEYPOINTS):
        df[f"{kp}_x"] = pose[:, j, 0]
        df[f"{kp}_y"] = pose[:, j, 1]

    # movement labels from reaches
    reaches_file = os.path.join(subdir, "reaches.csv")
    is_movement = np.zeros(n_samples, dtype=bool)
    if os.path.exists(reaches_file):
        reaches = pd.read_csv(reaches_file)
        for _, row in reaches.iterrows():
            mask = (timestamps >= row["start_time"]) & (timestamps <= row["stop_time"])
            is_movement[mask] = True
    df["is_movement"] = is_movement

    df.to_csv(os.path.join(scoring_data_dir, "reference_alignment.csv"), index=False)


def compute_reference_features(env_data_dir: str, scoring_data_dir: str):
    """Compute reference features for scoring Step 3."""
    from scipy.signal import welch

    all_X = []
    all_y = []
    all_subjects = []
    feature_names = []

    for subject_id in SUBJECTS:
        subject_dir = os.path.join(env_data_dir, f"sub-{subject_id}")
        if not os.path.exists(subject_dir):
            continue

        for ses_name in sorted(os.listdir(subject_dir)):
            ses_dir = os.path.join(subject_dir, ses_name)
            if not os.path.isdir(ses_dir):
                continue
            reaches_file = os.path.join(ses_dir, "reaches.csv")
            if not os.path.exists(reaches_file):
                continue

            ecog = np.load(os.path.join(ses_dir, "ecog.npy"))
            ecog_rate = float(np.load(os.path.join(ses_dir, "ecog_rate.npy")))
            pose = np.load(os.path.join(ses_dir, "pose.npy"))
            pose_rate = float(np.load(os.path.join(ses_dir, "pose_rate.npy")))
            electrodes = pd.read_csv(os.path.join(ses_dir, "electrodes.csv"))
            reaches = pd.read_csv(reaches_file)

            good_mask = electrodes["good"].astype(bool).values
            ecog_good = ecog[:, good_mask]

            # compute features per 1-second window
            window_size = int(pose_rate)  # 30 samples at 30 Hz
            ecog_window_size = int(ecog_rate)  # 500 samples at 500 Hz

            n_windows = min(
                len(pose) // window_size,
                len(ecog_good) // ecog_window_size,
            )

            for w in range(n_windows):
                features = []
                names = []

                # band power from 500 Hz data (correct Nyquist handling)
                ecog_win = ecog_good[w * ecog_window_size:(w + 1) * ecog_window_size]
                bands = {
                    "delta": (1, 4),
                    "theta": (4, 8),
                    "alpha": (8, 13),
                    "beta": (13, 30),
                    "gamma": (30, 100),
                }
                for band_name, (lo, hi) in bands.items():
                    freqs, psd = welch(
                        ecog_win.mean(axis=1),
                        fs=ecog_rate,
                        nperseg=min(256, len(ecog_win)),
                    )
                    band_mask = (freqs >= lo) & (freqs <= hi)
                    power = np.mean(psd[band_mask]) if band_mask.sum() > 0 else 0.0
                    features.append(np.log1p(power))
                    if w == 0:
                        names.append(f"band_power_{band_name}")

                # pose velocity
                pose_win = pose[w * window_size:(w + 1) * window_size]
                for kp_idx in range(pose_win.shape[1]):
                    vel = np.sqrt(
                        np.sum(np.diff(pose_win[:, kp_idx, :], axis=0) ** 2, axis=1)
                    )
                    features.append(np.nanmean(vel))
                    if w == 0:
                        names.append(f"velocity_kp{kp_idx}")

                # inter-keypoint distances
                pairs = [(7, 3), (8, 4), (7, 8)]  # LW-LS, RW-RS, LW-RW
                for p0, p1 in pairs:
                    dist = np.sqrt(
                        np.sum((pose_win[:, p0, :] - pose_win[:, p1, :]) ** 2, axis=1)
                    )
                    features.append(np.nanmean(dist))
                    if w == 0:
                        names.append(f"distance_{p0}_{p1}")

                all_X.append(features)

                # movement label for this window
                t_start = w / pose_rate + CLIP_START
                t_end = (w + 1) / pose_rate + CLIP_START
                is_mov = False
                for _, row in reaches.iterrows():
                    if row["start_time"] <= t_end and row["stop_time"] >= t_start:
                        is_mov = True
                        break
                all_y.append(int(is_mov))
                all_subjects.append(int(subject_id))

                if w == 0 and len(feature_names) == 0:
                    feature_names = names

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    subjects = np.array(all_subjects, dtype=np.int32)

    np.savez(
        os.path.join(scoring_data_dir, "reference_features.npz"),
        X=X, y=y, subject_ids=subjects,
        feature_names=np.array(feature_names),
    )


def compute_true_test_labels(scoring_data_dir: str):
    """Compute true test labels for Step 4 scoring."""

    test_dir = os.path.join(scoring_data_dir, "sub-03", "ses-3")
    if not os.path.exists(test_dir):
        print("WARNING: test dir not found, skipping true_test_labels")
        return

    reaches = pd.read_csv(os.path.join(test_dir, "reaches.csv"))
    pose_rate = float(np.load(os.path.join(test_dir, "pose_rate.npy")))
    pose = np.load(os.path.join(test_dir, "pose.npy"))

    n_windows = len(pose) // int(pose_rate)
    y_true = np.zeros(n_windows, dtype=np.int32)

    for w in range(n_windows):
        t_start = w / pose_rate + CLIP_START
        t_end = (w + 1) / pose_rate + CLIP_START
        for _, row in reaches.iterrows():
            if row["start_time"] <= t_end and row["stop_time"] >= t_start:
                y_true[w] = 1
                break

    np.savez(os.path.join(scoring_data_dir, "true_test_labels.npz"), y=y_true)


def main():
    os.makedirs("env_data", exist_ok=True)
    os.makedirs("scoring_data", exist_ok=True)

    for subj in SUBJECTS:
        for ses in SESSIONS_PER_SUBJECT[subj]:
            # subject 03 session 3 goes to scoring_data ONLY (full version with labels)
            if subj == "03" and ses == 3:
                output_dir = "scoring_data"
            else:
                output_dir = "env_data"

            print(f"Downloading sub-{subj} ses-{ses} -> {output_dir}/")
            try:
                meta = download_and_extract(subj, ses, output_dir)
                print(f"  OK: ecog {meta['ecog_shape']}, pose {meta['pose_shape']}, "
                      f"good electrodes: {meta['n_good_electrodes']}/{meta['n_electrodes']}")
            except Exception as e:
                print(f"  ERROR: {e}")

    # copy sub-03 ses-3 to env_data (data only, NO labels)
    src = "scoring_data/sub-03/ses-3"
    dst = "env_data/sub-03/ses-3"
    if os.path.exists(src):
        os.makedirs(dst, exist_ok=True)
        for fname in ["ecog.npy", "ecog_rate.npy", "pose.npy", "pose_rate.npy",
                       "electrodes.csv", "metadata.json"]:
            src_f = os.path.join(src, fname)
            if os.path.exists(src_f):
                shutil.copy2(src_f, os.path.join(dst, fname))
        # DO NOT copy reaches.csv (contains labels!)
        print("Copied sub-03 ses-3 to env_data (without reaches.csv)")

    print("\nComputing reference alignment...")
    compute_reference_alignment("env_data", "scoring_data")

    print("Computing reference features...")
    compute_reference_features("env_data", "scoring_data")

    print("Computing true test labels...")
    compute_true_test_labels("scoring_data")

    # copy scoring scripts into scoring_data
    # (in production these would be packaged with the container)
    print("\nDone! Container data ready.")
    print(f"  env_data/: {sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk('env_data') for f in fn) / 1e9:.2f} GB")
    print(f"  scoring_data/: {sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk('scoring_data') for f in fn) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
```

---

## 10. Container Requirements

### 10.1 env_requirements.txt

```
numpy>=1.26
pandas>=2.1
scipy>=1.12
scikit-learn>=1.4
h5py>=3.10
pynwb>=2.6
matplotlib>=3.8
```

### 10.2 Container Spec

```yaml
base_image: python:3.13-slim
memory_limit: 8GB
cpu_limit: 4 cores
network: NONE (all data pre-downloaded)
storage: 10GB ephemeral
timeout: 4 hours
```

---

## 11. Anti-Gaming Summary

| Step | Anti-gaming measure |
|------|---------------------|
| 1 | Data-dependent answers, can't guess without exploring files |
| 2 | Reference alignment computed independently; correlation-based scoring catches wrong downsampling |
| 3 | Nyquist trap (high-freq bands need 500 Hz); downstream model is fixed; feature quality checked independently |
| 4 | Hidden test session in scoring_data; constant prediction check; dual evaluation paths (downstream + student predictions) |

### 11.1 Reward Hacking Prevention

1. **No test data leakage**: sub-03 ses-3 reaches.csv is in scoring_data only
2. **No score visibility**: student never sees scoring script output
3. **Fixed downstream model**: logistic regression with C=1.0, solver=lbfgs -- student can't game model choice
4. **Constant prediction detection**: explicit check in Step 4 scoring
5. **Correlation-based metrics**: measures signal quality, not just format compliance
6. **Multiple scoring dimensions**: even if one component is gamed, others keep the score honest

---

## 12. Expected Score Distributions

| Step | Score range | Median | What differentiates good from great |
|------|------------|--------|-------------------------------------|
| 1 | 0.0-1.0 | 0.75 | keypoint names (alphabetical order), bad electrode count |
| 2 | 0.2-0.95 | 0.55 | anti-alias filter choice, bad electrode exclusion |
| 3 | 0.1-0.85 | 0.35 | Nyquist awareness (use 500 Hz for beta/gamma) |
| 4 | 0.1-0.70 | 0.25 | cross-subject normalization, feature selection, movement oversampling |

### 12.1 AUC-to-Score Mapping (used in Steps 3 and 4)

```
AUC     -> Score
<= 0.5  -> 0.0
0.5-0.7 -> 0.0-0.3  (linear)
0.7-0.85-> 0.3-0.7  (linear)
0.85-0.95-> 0.7-1.0 (linear)
> 0.95  -> capped at 1.0
```

This creates a smooth reward curve where:
- Random performance (AUC=0.5) = 0.0
- Moderate performance (AUC=0.7) = 0.3
- Good performance (AUC=0.85) = 0.7
- Excellent performance (AUC=0.95) = 1.0

---

## 13. Why This Environment Is Good

1. **No known pattern**: multi-format alignment + Nyquist reasoning = genuine thinking required, not pattern matching
2. **Many failure points**: bad downsampling, wrong electrodes, Nyquist violation, poor features, no cross-subject normalization
3. **Variability**: continuous scores at every step, correlation-based metrics produce smooth gradients
4. **Constant improvement**: can add more subjects, harder test sessions, time/memory constraints
5. **Real data**: DANDI #000055, publicly available, neuroscience domain
6. **Data curation pattern**: student curates, system evaluates downstream -- exactly what the customer wants
7. **Progressive difficulty**: Step 1 is easy (exploration), Step 4 is genuinely hard (cross-subject generalization)
8. **Signal processing literacy test**: the Nyquist trap in Step 3 separates agents that reason about constraints from those that blindly execute recipes

---

## Appendix A: NWB File Access Reference

### A.1 Streaming from DANDI (used only in setup_data.py)

```python
from dandi.dandiapi import DandiAPIClient
import remfile
import h5py

DANDISET_ID = "000055"

def get_nwb_s3_url(subject_id: str, session_id: int) -> str:
    filename = f"sub-{subject_id}/sub-{subject_id}_ses-{session_id}_behavior+ecephys.nwb"
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(DANDISET_ID)
        asset = dandiset.get_asset_by_path(filename)
        return asset.get_content_url(follow_redirects=1, strip_query=True)

url = get_nwb_s3_url("01", 1)
rf = remfile.File(url)
f = h5py.File(rf, "r")

# lazy slicing -- only fetches requested data
ecog_chunk = f["acquisition/ElectricalSeries/data"][0:1000, 0:10]
```

### A.2 Electrode Quality Assessment

From the AJILE12 paper, bad electrode criteria:
- voltage std > 5 IQRs above median across electrodes
- voltage kurtosis > 10 IQRs above median across electrodes
- pre-computed and stored in the `good` column of the electrodes table

### A.3 Reference Time Alignment

```python
from scipy.signal import decimate

# ECoG at 500 Hz -> 30 Hz via decimate with FIR anti-aliasing
factor = int(500 / 30)  # = 16 (closest integer factor)
ecog_30hz = decimate(ecog_500hz, factor, axis=0, ftype="fir")
```

Note: 500/30 = 16.667, so `decimate(x, 16)` produces ~31.25 Hz, which is close enough. The reference alignment uses this approach and truncates to match pose length.

---

## Appendix B: References

### Primary Dataset Papers

1. Peterson, S. M., et al. (2022). AJILE12: Long-term naturalistic human intracranial neural recordings and pose. *Scientific Data*, 9, 184. https://doi.org/10.1038/s41597-022-01280-y

2. Singh, S. H., Peterson, S. M., Rao, R. P., & Brunton, B. W. (2021). Mining naturalistic human behaviors in long-term video and neural recordings. *Journal of Neuroscience Methods*, 358, 109199. https://doi.org/10.1016/j.jneumeth.2021.109199

3. Peterson, S. M., Singh, S. H., Wang, N. X., Rao, R. P., & Brunton, B. W. (2021). Behavioral and Neural Variability of Naturalistic Arm Movements. *eNeuro*, 8(3). https://doi.org/10.1523/ENEURO.0007-21.2021

4. Wang, N. X., Farhadi, A., Rao, R. P., & Brunton, B. W. (2018). AJILE Movement Prediction: Multimodal Deep Learning for Natural Human Neural Recordings and Video. *AAAI*, 32(1). https://arxiv.org/abs/1709.05939

### Code Repositories

5. BruntonUWBio/ajile12-nwb-data -- Official AJILE12 NWB tools. https://github.com/BruntonUWBio/ajile12-nwb-data

6. neurovium/Neuromatch-AJILE12 -- NMA educational materials. https://github.com/neurovium/Neuromatch-AJILE12

7. stepeter/naturalistic_arm_movements_ecog -- Analysis code for Peterson et al. 2021. https://github.com/stepeter/naturalistic_arm_movements_ecog

### Technical References

8. DANDI Archive -- Dandiset 000055. https://dandiarchive.org/dandiset/000055

9. PyNWB Documentation -- Streaming NWB Files. https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/streaming.html

10. NWB Overview -- File Structure. https://nwb-overview.readthedocs.io/en/latest/intro_to_nwb/2_file_structure.html

---

*end of specification*
