# E01: Messy Multi-Format Data Science (AJILE12) -- pm_env Specification

> production-ready pm_env specification for E01
> dataset: AJILE12 ECoG + Pose (DANDI #000055)
> created: 2026-02-09 | author: claude opus 4.6

---

## Table of Contents

1. [Environment Overview](#1-environment-overview)
2. [Data Layout](#2-data-layout)
3. [Task Definition: 4 Progressive Steps](#3-task-definition)
4. [Step 1: Data Exploration and Schema Discovery](#4-step-1)
5. [Step 2: Load and Align Multi-Rate Data](#5-step-2)
6. [Step 3: Feature Engineering](#6-step-3)
7. [Step 4: Prediction](#7-step-4)
8. [tasks.py Skeleton](#8-taskspy)
9. [setup_data.py (Pre-Download)](#9-setup_datapy)
10. [env_requirements.txt](#10-env_requirementstxt)
11. [pre_hook Implementation](#11-pre_hook)
12. [Expected Score Distributions](#12-expected-score-distributions)
13. [References](#13-references)

---

## 1. Environment Overview

### 1.1 What This Environment Tests

An AI agent receives pre-downloaded NWB files from the AJILE12 dataset containing human ECoG neural recordings and pose tracking data. The agent must:

1. Explore unfamiliar HDF5/NWB files with no documentation
2. Align multi-rate signals (500 Hz ECoG vs 30 Hz pose)
3. Engineer neuroscience features (spectral band power, kinematics)
4. Build a cross-subject movement classifier

This maps directly to the customer requirement: "Tasks where the agent receives data that's spread across multiple formats or modalities... and must collect, clean, and make predictions or inferences on the data."

### 1.2 Why AJILE12

- **Real data** (not synthetic) -- 12 human subjects with ECoG implants
- **Genuinely messy** -- bad electrodes, missing keypoints, variable session lengths, different sampling rates
- **Continuous metrics** -- alignment accuracy, feature correlation, AUC-ROC
- **Open-ended** -- many valid approaches at each step
- **Well-documented failure modes** from NMA student projects

### 1.3 Hardware and Constraints

| Constraint | Value |
|------------|-------|
| Compute | CPU only (no GPU/TPU) |
| Container memory | 8 GB |
| Container disk | 10 GB |
| Network | DISABLED (all data pre-downloaded) |
| Time limit per task | 4 hours |
| Max tool call duration | 10 minutes |
| Grading step timeout | 4 hours |

### 1.4 Container Image

```
base_image: python:3.11-slim
```

All Python dependencies are pre-installed. The NWB files are pre-downloaded into `env_data/`. The student never sees `scoring_data/`.

---

## 2. Data Layout

### 2.1 What Gets Pre-Downloaded

We download data for **3 subjects** (sub-01, sub-02, sub-03), **3 sessions each** (ses-1, ses-2, ses-3). Each session is a 24-hour continuous recording, but we extract **1-hour windows** to keep file sizes manageable (~2-5 GB per full session, ~100-200 MB per 1-hour extract).

Extraction is done by `setup_data.py` (see Section 9) which reads from DANDI, slices the NWB files, and writes trimmed local copies.

### 2.2 Directory Structure

```
/workspace/
├── env_data/                          # VISIBLE to student
│   ├── sub-01/
│   │   ├── sub-01_ses-1_slice.nwb     # 1-hour extract, all modalities
│   │   ├── sub-01_ses-2_slice.nwb
│   │   └── sub-01_ses-3_slice.nwb
│   ├── sub-02/
│   │   ├── sub-02_ses-1_slice.nwb
│   │   ├── sub-02_ses-2_slice.nwb
│   │   └── sub-02_ses-3_slice.nwb
│   └── sub-03/
│       ├── sub-03_ses-1_slice.nwb     # ONLY ses-1 visible (training)
│       └── sub-03_ses-2_slice.nwb     # ONLY ses-2 visible (training)
│                                      # ses-3 is HIDDEN in scoring_data
│
├── scoring_data/                      # HIDDEN from student
│   ├── sub-03/
│   │   └── sub-03_ses-3_slice.nwb     # hidden test session
│   ├── reference/
│   │   ├── step1_answers.json         # ground truth for Step 1
│   │   ├── step2_aligned.parquet      # reference aligned DataFrame
│   │   ├── step3_features.parquet     # reference features
│   │   └── step4_labels.parquet       # test labels for Step 4
│   └── scoring_scripts/
│       ├── judge_step1.py
│       ├── judge_step2.py
│       ├── judge_step3.py
│       └── judge_step4.py
│
└── submission/                        # student writes answers here
    ├── step1_answers.json
    ├── step2_aligned.parquet
    ├── step3_features.parquet
    └── step4_predictions.parquet
```

### 2.3 NWB Slice Structure

Each `*_slice.nwb` file contains a 1-hour window with the same internal structure as the full AJILE12 NWB files:

```
/
├── acquisition/
│   └── ElectricalSeries/
│       ├── data          [n_samples, n_channels]  float32  (500 Hz)
│       ├── starting_time float64
│       └── rate          500.0
│
├── processing/
│   └── behavior/
│       └── Position/
│           └── SpatialSeries/
│               ├── data       [n_frames, 9, 2]  float64  (30 Hz)
│               └── timestamps [n_frames]         float64
│
├── intervals/
│   ├── reaches/           DynamicTable
│   │   ├── start_time
│   │   ├── stop_time
│   │   ├── reach_magnitude
│   │   ├── reach_duration
│   │   └── reach_angle
│   └── epochs/            DynamicTable (if available)
│       ├── start_time
│       ├── stop_time
│       └── tags
│
└── general/
    └── extracellular_ephys/
        └── electrodes/    DynamicTable
            ├── x, y, z            float64 (MNI coords)
            ├── group_name         str
            ├── location           str
            ├── good               bool
            ├── std                float64
            ├── kurtosis           float64
            ├── mad                float64
            ├── lfb_r2             float64
            └── hfb_r2             float64
```

### 2.4 Data Volume Estimates

| Item | Size |
|------|------|
| 1-hour ECoG (64 ch, 500 Hz, float32) | ~460 MB |
| 1-hour pose (9 kp, 30 Hz, float64) | ~6 MB |
| Electrode table | ~10 KB |
| Reaches table | ~50-200 KB |
| Single slice NWB (with compression) | ~150-250 MB |
| Total env_data (8 files) | ~1.5-2.0 GB |
| Total scoring_data (1 file + refs) | ~300 MB |
| **Total container disk** | **~2.5 GB** |

---

## 3. Task Definition

### 3.1 Single Task, Four Progressive Steps

The environment contains ONE task (`ajile12_wrangling`) with 4 sequential Steps. Each Step builds on the previous but provides fallback reference data so a student who fails Step 2 can still attempt Step 3.

| Step | Name | Judge Type | Reward Range |
|------|------|------------|--------------|
| 1 | Data Exploration & Schema Discovery | ExactMatchJudge | 0.0 or 1.0 (binary) |
| 2 | Load & Align Multi-Rate Data | ExecutableJudge (continuous) | 0.0 -- 1.0 |
| 3 | Feature Engineering | ExecutableJudge (continuous) | 0.0 -- 1.0 |
| 4 | Prediction | ExecutableJudge (continuous) | 0.0 -- 1.0 |

### 3.2 Step Dependencies

- Step 2 references env_data/ files explored in Step 1
- Step 3 can use the student's Step 2 output OR a reference aligned DataFrame (provided in env_data after Step 2 grading)
- Step 4 can use the student's Step 3 features OR reference features

This ensures partial credit: a student who nails Steps 1 and 4 but botches Step 2 still earns meaningful reward.

---

## 4. Step 1: Data Exploration and Schema Discovery

### 4.1 Task Instructions (shown to student)

```
STEP 1: Data Exploration and Schema Discovery
==============================================

You have been given a set of NWB files in /workspace/env_data/.
These files contain neuroscience data, but you have NO documentation
about their internal structure.

Your task: explore the files and answer the following questions.

Write your answers to /workspace/submission/step1_answers.json
in the following format:

{
    "num_subjects": <int>,
    "sessions_per_subject": {"sub-01": <int>, "sub-02": <int>, "sub-03": <int>},
    "ecog_sampling_rate_hz": <number>,
    "num_pose_keypoints": <int>,
    "electrode_table_columns": [<list of column name strings, sorted alphabetically>],
    "num_good_electrodes_sub01_ses1": <int>,
    "pose_data_unit": <string>,
    "num_reach_events_sub01_ses1": <int>
}

Hints:
- The files are in NWB format (HDF5-based). Use h5py or pynwb to open them.
- Start by listing what's inside one file, then generalize.
- Be precise: column names must match exactly (case-sensitive, sorted).

Time limit: 30 minutes recommended (of the 4 hour total).
```

### 4.2 Ground Truth (scoring_data/reference/step1_answers.json)

This file is generated by `setup_data.py` during the data preparation phase. Example:

```json
{
    "num_subjects": 3,
    "sessions_per_subject": {"sub-01": 3, "sub-02": 3, "sub-03": 2},
    "ecog_sampling_rate_hz": 500.0,
    "num_pose_keypoints": 9,
    "electrode_table_columns": [
        "good", "group", "group_name", "hfb_r2", "kurtosis",
        "lfb_r2", "location", "mad", "std", "x", "y", "z"
    ],
    "num_good_electrodes_sub01_ses1": 52,
    "pose_data_unit": "pixels",
    "num_reach_events_sub01_ses1": 87
}
```

NOTE: The actual values depend on the specific 1-hour windows extracted. `setup_data.py` generates this file from the actual data.

### 4.3 Judge: ExactMatchJudge

```python
# scoring_data/scoring_scripts/judge_step1.py
"""
Step 1 Judge: ExactMatchJudge
Binary scoring -- 1.0 if all answers match, 0.0 otherwise.
Partial credit for getting most answers right is encoded
as: score = (num_correct / num_total).
"""

import json
import sys


def judge_step1(submission_path: str, reference_path: str) -> float:
    try:
        with open(submission_path, "r") as f:
            submitted = json.load(f)
        with open(reference_path, "r") as f:
            reference = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: Could not read submission: {e}", file=sys.stderr)
        return 0.0

    total_checks = 0
    correct_checks = 0

    # Check 1: num_subjects
    total_checks += 1
    if submitted.get("num_subjects") == reference["num_subjects"]:
        correct_checks += 1

    # Check 2: sessions_per_subject (dict match)
    total_checks += 1
    ref_sessions = reference["sessions_per_subject"]
    sub_sessions = submitted.get("sessions_per_subject", {})
    if isinstance(sub_sessions, dict) and sub_sessions == ref_sessions:
        correct_checks += 1

    # Check 3: ecog_sampling_rate_hz
    total_checks += 1
    ref_rate = reference["ecog_sampling_rate_hz"]
    sub_rate = submitted.get("ecog_sampling_rate_hz")
    if sub_rate is not None and abs(float(sub_rate) - ref_rate) < 1.0:
        correct_checks += 1

    # Check 4: num_pose_keypoints
    total_checks += 1
    if submitted.get("num_pose_keypoints") == reference["num_pose_keypoints"]:
        correct_checks += 1

    # Check 5: electrode_table_columns (sorted list match)
    total_checks += 1
    ref_cols = sorted(reference["electrode_table_columns"])
    sub_cols = submitted.get("electrode_table_columns", [])
    if isinstance(sub_cols, list) and sorted(sub_cols) == ref_cols:
        correct_checks += 1

    # Check 6: num_good_electrodes_sub01_ses1
    total_checks += 1
    if submitted.get("num_good_electrodes_sub01_ses1") == reference["num_good_electrodes_sub01_ses1"]:
        correct_checks += 1

    # Check 7: pose_data_unit
    total_checks += 1
    ref_unit = reference["pose_data_unit"].lower().strip()
    sub_unit = str(submitted.get("pose_data_unit", "")).lower().strip()
    if sub_unit == ref_unit:
        correct_checks += 1

    # Check 8: num_reach_events_sub01_ses1
    total_checks += 1
    ref_reaches = reference["num_reach_events_sub01_ses1"]
    sub_reaches = submitted.get("num_reach_events_sub01_ses1")
    if sub_reaches is not None and int(sub_reaches) == ref_reaches:
        correct_checks += 1

    score = correct_checks / total_checks
    print(f"Step 1: {correct_checks}/{total_checks} correct ({score:.2f})")
    return score


if __name__ == "__main__":
    submission_path = "/workspace/submission/step1_answers.json"
    reference_path = "/workspace/scoring_data/reference/step1_answers.json"
    score = judge_step1(submission_path, reference_path)
    print(f"SCORE:{score:.4f}")
```

### 4.4 What's in env_data vs scoring_data

| Location | Contents |
|----------|----------|
| env_data | All 8 NWB slice files (sub-01 x3, sub-02 x3, sub-03 x2) |
| scoring_data | `reference/step1_answers.json` with ground truth |

### 4.5 Anti-Gaming Measures

1. **No documentation provided** -- student must genuinely explore the HDF5 structure
2. **Electrode column list requires exact match** -- cannot guess column names
3. **num_good_electrodes and num_reach_events are data-dependent** -- cannot hardcode
4. **Sessions-per-subject is asymmetric** (sub-03 has only 2 visible) -- catches pattern-matchers

### 4.6 Expected Score Distribution

| Score | Interpretation | Expected % of agents |
|-------|---------------|---------------------|
| 1.0 (8/8) | All correct | 40% |
| 0.875 (7/8) | One tricky field wrong | 25% |
| 0.75 (6/8) | Two fields wrong | 15% |
| 0.5-0.625 | Partial understanding | 10% |
| < 0.5 | Major confusion | 10% |

---

## 5. Step 2: Load and Align Multi-Rate Data

### 5.1 Task Instructions (shown to student)

```
STEP 2: Load and Align Multi-Rate Data
========================================

Given the NWB files you explored in Step 1, produce an aligned
DataFrame combining ECoG and pose data at a common 30 Hz timebase.

SPECIFIC REQUEST:
- Subject: sub-01, Session: ses-1
- Time window: the FIRST 5 MINUTES (0 to 300 seconds from session start)
- Include ALL good electrodes (where the 'good' column is True)
- Include ALL 9 pose keypoints

REQUIREMENTS:
1. Downsample ECoG from 500 Hz to 30 Hz by averaging samples within
   each 33.3ms window centered on each pose timestamp.
2. Use pose timestamps as the reference timebase.
3. Exclude bad electrodes (good == False).
4. Handle any missing pose data with NaN.
5. Include movement labels from the reaches interval table.

OUTPUT FORMAT:
Save a Parquet file to /workspace/submission/step2_aligned.parquet

The DataFrame must have these columns:
- timestamp: float64 (seconds, from pose timestamps)
- ecog_{electrode_index}: float32 (one column per good electrode,
  e.g., ecog_0, ecog_1, ..., ecog_51)
- {keypoint}_x: float64 (e.g., nose_x, left_ear_x, ...)
- {keypoint}_y: float64 (e.g., nose_y, left_ear_y, ...)
- is_reach: bool (True if timestamp falls within a reach interval)

Keypoint names (in order): nose, left_ear, right_ear, left_shoulder,
right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist

Expected row count: ~9000 rows (5 minutes x 60 sec x 30 Hz)

Save as: /workspace/submission/step2_aligned.parquet

Time limit: 45 minutes recommended.
```

### 5.2 Reference Data Generation

The reference aligned DataFrame is generated during `setup_data.py` execution using the reference alignment algorithm:

```python
# Reference alignment algorithm (used in setup_data.py)
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO


def generate_reference_alignment(nwb_path, start_sec, end_sec):
    with NWBHDF5IO(nwb_path, "r") as io:
        nwb = io.read()

        ecog = nwb.acquisition["ElectricalSeries"]
        ecog_rate = ecog.rate
        ecog_start = ecog.starting_time if ecog.starting_time else 0.0

        position = nwb.processing["behavior"]["Position"]
        spatial_series = list(position.spatial_series.values())[0]
        pose_ts_all = spatial_series.timestamps[:]
        pose_data_all = spatial_series.data[:]

        electrodes_df = nwb.electrodes.to_dataframe()
        good_mask = electrodes_df["good"].values
        good_indices = np.where(good_mask)[0].tolist()

        reaches_df = nwb.intervals["reaches"].to_dataframe()

        pose_mask = (pose_ts_all >= start_sec) & (pose_ts_all < end_sec)
        pose_ts = pose_ts_all[pose_mask]
        pose_data = pose_data_all[pose_mask]

        ecog_start_sample = int((start_sec - ecog_start) * ecog_rate)
        ecog_end_sample = int((end_sec - ecog_start) * ecog_rate)
        ecog_data = ecog.data[ecog_start_sample:ecog_end_sample, :]
        ecog_ts = ecog_start + np.arange(ecog_start_sample, ecog_end_sample) / ecog_rate

        half_window = 0.5 / 30.0
        n_pose = len(pose_ts)
        n_good = len(good_indices)
        aligned_ecog = np.full((n_pose, n_good), np.nan, dtype=np.float32)

        for i, t in enumerate(pose_ts):
            mask = (ecog_ts >= t - half_window) & (ecog_ts < t + half_window)
            if mask.sum() > 0:
                aligned_ecog[i] = ecog_data[mask][:, good_indices].mean(axis=0)

        df = pd.DataFrame()
        df["timestamp"] = pose_ts

        for j, idx in enumerate(good_indices):
            df[f"ecog_{idx}"] = aligned_ecog[:, j]

        keypoint_names = [
            "nose", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        ]
        for k, name in enumerate(keypoint_names):
            df[f"{name}_x"] = pose_data[:, k, 0]
            df[f"{name}_y"] = pose_data[:, k, 1]

        df["is_reach"] = False
        for _, reach in reaches_df.iterrows():
            if reach["stop_time"] < start_sec or reach["start_time"] >= end_sec:
                continue
            rmask = (df["timestamp"] >= reach["start_time"]) & (
                df["timestamp"] <= reach["stop_time"]
            )
            df.loc[rmask, "is_reach"] = True

        return df
```

### 5.3 Judge: ExecutableJudge (Continuous)

```python
# scoring_data/scoring_scripts/judge_step2.py
"""
Step 2 Judge: Continuous scoring based on alignment accuracy.
Score = fraction of pose timestamps where the student's aligned ECoG
values are within tolerance of the reference values.
"""

import sys
import numpy as np
import pandas as pd


def judge_step2(submission_path: str, reference_path: str) -> float:
    try:
        submitted = pd.read_parquet(submission_path)
    except Exception as e:
        print(f"ERROR: Could not read submission: {e}", file=sys.stderr)
        return 0.0

    try:
        reference = pd.read_parquet(reference_path)
    except Exception as e:
        print(f"ERROR: Could not read reference: {e}", file=sys.stderr)
        return 0.0

    scores = []
    weights = []

    # --- Component 1: Timestamp alignment (weight 0.35) ---
    score_ts = score_timestamp_alignment(submitted, reference)
    scores.append(score_ts)
    weights.append(0.35)

    # --- Component 2: ECoG value accuracy (weight 0.30) ---
    score_ecog = score_ecog_values(submitted, reference)
    scores.append(score_ecog)
    weights.append(0.30)

    # --- Component 3: Pose value accuracy (weight 0.15) ---
    score_pose = score_pose_values(submitted, reference)
    scores.append(score_pose)
    weights.append(0.15)

    # --- Component 4: Reach labels (weight 0.10) ---
    score_reach = score_reach_labels(submitted, reference)
    scores.append(score_reach)
    weights.append(0.10)

    # --- Component 5: Schema correctness (weight 0.10) ---
    score_schema = score_schema_correctness(submitted, reference)
    scores.append(score_schema)
    weights.append(0.10)

    final_score = sum(s * w for s, w in zip(scores, weights))

    print(f"Step 2 breakdown:")
    print(f"  Timestamp alignment: {scores[0]:.4f} (weight {weights[0]})")
    print(f"  ECoG values:         {scores[1]:.4f} (weight {weights[1]})")
    print(f"  Pose values:         {scores[2]:.4f} (weight {weights[2]})")
    print(f"  Reach labels:        {scores[3]:.4f} (weight {weights[3]})")
    print(f"  Schema correctness:  {scores[4]:.4f} (weight {weights[4]})")
    print(f"  TOTAL:               {final_score:.4f}")

    return float(np.clip(final_score, 0.0, 1.0))


def score_timestamp_alignment(submitted, reference, tolerance_ms=33.33):
    if "timestamp" not in submitted.columns:
        return 0.0

    sub_ts = submitted["timestamp"].values
    ref_ts = reference["timestamp"].values

    if len(sub_ts) == 0:
        return 0.0

    n_ref = len(ref_ts)
    matched = 0

    for t in ref_ts:
        idx = np.searchsorted(sub_ts, t)
        candidates = []
        if idx > 0:
            candidates.append(abs(sub_ts[idx - 1] - t))
        if idx < len(sub_ts):
            candidates.append(abs(sub_ts[idx] - t))
        if candidates and min(candidates) * 1000 <= tolerance_ms:
            matched += 1

    return matched / n_ref


def score_ecog_values(submitted, reference):
    ref_ecog_cols = sorted([c for c in reference.columns if c.startswith("ecog_")])
    sub_ecog_cols = sorted([c for c in submitted.columns if c.startswith("ecog_")])

    if not sub_ecog_cols:
        return 0.0

    common_cols = sorted(set(ref_ecog_cols) & set(sub_ecog_cols))
    if not common_cols:
        return 0.0

    col_coverage = len(common_cols) / len(ref_ecog_cols)

    sub_ts = submitted["timestamp"].values
    ref_ts = reference["timestamp"].values

    min_len = min(len(sub_ts), len(ref_ts))
    if min_len == 0:
        return 0.0

    correlations = []
    for col in common_cols[:20]:
        sub_vals = submitted[col].values[:min_len]
        ref_vals = reference[col].values[:min_len]

        valid = ~(np.isnan(sub_vals) | np.isnan(ref_vals))
        if valid.sum() < 10:
            continue

        corr = np.corrcoef(sub_vals[valid], ref_vals[valid])[0, 1]
        if not np.isnan(corr):
            correlations.append(max(0.0, corr))

    if not correlations:
        return 0.0

    mean_corr = np.mean(correlations)
    return col_coverage * mean_corr


def score_pose_values(submitted, reference):
    keypoint_names = [
        "nose", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
    ]
    expected_cols = []
    for kp in keypoint_names:
        expected_cols.extend([f"{kp}_x", f"{kp}_y"])

    present = sum(1 for c in expected_cols if c in submitted.columns)
    if present == 0:
        return 0.0

    col_score = present / len(expected_cols)

    min_len = min(len(submitted), len(reference))
    correlations = []
    for col in expected_cols:
        if col not in submitted.columns or col not in reference.columns:
            continue
        sub_vals = submitted[col].values[:min_len]
        ref_vals = reference[col].values[:min_len]
        valid = ~(np.isnan(sub_vals) | np.isnan(ref_vals))
        if valid.sum() < 10:
            continue
        corr = np.corrcoef(sub_vals[valid], ref_vals[valid])[0, 1]
        if not np.isnan(corr):
            correlations.append(max(0.0, corr))

    if not correlations:
        return col_score * 0.5

    return col_score * np.mean(correlations)


def score_reach_labels(submitted, reference):
    if "is_reach" not in submitted.columns:
        return 0.0
    if "is_reach" not in reference.columns:
        return 0.0

    min_len = min(len(submitted), len(reference))
    sub_r = submitted["is_reach"].values[:min_len].astype(bool)
    ref_r = reference["is_reach"].values[:min_len].astype(bool)

    agreement = (sub_r == ref_r).mean()
    return float(agreement)


def score_schema_correctness(submitted, reference):
    ref_cols = set(reference.columns)
    sub_cols = set(submitted.columns)

    required = {"timestamp", "is_reach"}
    ecog_cols = {c for c in ref_cols if c.startswith("ecog_")}
    pose_cols = {c for c in ref_cols if "_x" in c or "_y" in c}

    score = 0.0
    if required.issubset(sub_cols):
        score += 0.3
    sub_ecog = {c for c in sub_cols if c.startswith("ecog_")}
    if len(sub_ecog) > 0:
        score += 0.3 * min(1.0, len(sub_ecog & ecog_cols) / max(len(ecog_cols), 1))
    sub_pose = {c for c in sub_cols if "_x" in c or "_y" in c}
    if len(sub_pose) > 0:
        score += 0.2 * min(1.0, len(sub_pose & pose_cols) / max(len(pose_cols), 1))

    ref_rows = len(reference)
    sub_rows = len(submitted)
    if sub_rows > 0:
        ratio = min(sub_rows, ref_rows) / max(sub_rows, ref_rows)
        score += 0.2 * ratio

    return score


if __name__ == "__main__":
    submission_path = "/workspace/submission/step2_aligned.parquet"
    reference_path = "/workspace/scoring_data/reference/step2_aligned.parquet"
    score = judge_step2(submission_path, reference_path)
    print(f"SCORE:{score:.4f}")
```

### 5.4 What's in env_data vs scoring_data

| Location | Contents |
|----------|----------|
| env_data | All NWB slices (student reads sub-01_ses-1 for this step) |
| scoring_data | `reference/step2_aligned.parquet` (reference output) |

### 5.5 Anti-Gaming Measures

1. **ECoG correlation check** -- cannot just copy pose timestamps and fill ECoG with zeros; the actual ECoG values must correlate with reference (r > 0.9)
2. **Timestamp precision** -- using `np.searchsorted` for efficient nearest-neighbor matching; detects off-by-one errors, wrong sampling rate, or inverted time axes
3. **Schema validation** -- must produce the correct column names, not just any DataFrame
4. **Row count** -- must be close to expected (~9000 rows); too few or too many means incorrect windowing
5. **Reach label accuracy** -- independent check that movement intervals are correctly assigned

### 5.6 Expected Score Distribution

| Score Range | Interpretation | Expected % |
|-------------|---------------|------------|
| 0.90 -- 1.0 | Excellent alignment, all components correct | 15% |
| 0.70 -- 0.89 | Good alignment, minor issues (few bad cols, slight drift) | 30% |
| 0.50 -- 0.69 | Partial success (timestamps OK, but ECoG poorly aligned) | 25% |
| 0.20 -- 0.49 | Major issues (wrong schema, large timestamp errors) | 20% |
| 0.00 -- 0.19 | Fundamentally broken (crash, empty output, wrong format) | 10% |

---

## 6. Step 3: Feature Engineering

### 6.1 Task Instructions (shown to student)

```
STEP 3: Feature Engineering
============================

Starting from aligned neural + pose data, compute neuroscience features.

INPUT:
If your Step 2 output exists at /workspace/submission/step2_aligned.parquet
and is valid, use it. Otherwise, use the reference alignment provided at
/workspace/env_data/step2_reference_aligned.parquet (this file will be
placed after Step 2 grading completes).

REQUIRED FEATURES (compute all of these):

1. ECoG Band Power (per good electrode):
   - delta:  1-4 Hz
   - theta:  4-8 Hz
   - alpha:  8-13 Hz
   - beta:   13-30 Hz
   - gamma:  30-100 Hz

   Use a 1-second sliding window. The ECoG data in the aligned file
   is at 30 Hz (downsampled), so each window has 30 samples.
   Compute power spectral density and integrate within each band.

   Column names: bandpower_{electrode_index}_{band}
   Example: bandpower_0_alpha, bandpower_0_beta, ...

   NOTE: Since ECoG was downsampled to 30 Hz, Nyquist is 15 Hz.
   You can ONLY compute delta, theta, and alpha directly.
   For beta and gamma, you must use the ORIGINAL NWB files
   (env_data/) to extract these from the raw 500 Hz signal,
   then align to the 30 Hz timestamps.

2. Movement Velocity (per keypoint):
   - Compute frame-to-frame displacement: sqrt(dx^2 + dy^2)
   - Divide by dt (1/30 sec) to get pixels/second
   Column names: velocity_{keypoint}
   Example: velocity_nose, velocity_left_wrist, ...

3. Movement Acceleration (per keypoint):
   - First derivative of velocity
   Column names: acceleration_{keypoint}

4. Inter-Keypoint Distances (selected pairs):
   - left_wrist to right_wrist
   - nose to left_wrist
   - nose to right_wrist
   - left_shoulder to left_wrist (arm length proxy)
   - right_shoulder to right_wrist
   Column names: dist_{kp1}_{kp2}
   Example: dist_left_wrist_right_wrist

OUTPUT FORMAT:
Save a Parquet file to /workspace/submission/step3_features.parquet

The DataFrame must include:
- timestamp: float64 (same as input)
- All band power features (for up to 10 electrodes to keep manageable)
- All velocity features (9 keypoints)
- All acceleration features (9 keypoints)
- All inter-keypoint distance features (5 pairs)
- is_reach: bool (preserved from input)

Total expected feature columns: ~80-100 (depending on electrode count)

Time limit: 60 minutes recommended.
```

### 6.2 Reference Features Generation

Reference features are computed during `setup_data.py` using the canonical methods:

```python
# Reference feature computation (used in setup_data.py)
import numpy as np
import pandas as pd
from scipy import signal


def compute_reference_features(aligned_df, nwb_path, start_sec, end_sec):
    from pynwb import NWBHDF5IO

    df = aligned_df.copy()
    timestamps = df["timestamp"].values
    dt = 1.0 / 30.0

    # --- Velocity and Acceleration ---
    keypoints = [
        "nose", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
    ]

    for kp in keypoints:
        x = df[f"{kp}_x"].values
        y = df[f"{kp}_y"].values
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        vel = np.sqrt(dx**2 + dy**2) / dt
        df[f"velocity_{kp}"] = vel

        dvel = np.diff(vel, prepend=vel[0])
        df[f"acceleration_{kp}"] = dvel / dt

    # --- Inter-Keypoint Distances ---
    pairs = [
        ("left_wrist", "right_wrist"),
        ("nose", "left_wrist"),
        ("nose", "right_wrist"),
        ("left_shoulder", "left_wrist"),
        ("right_shoulder", "right_wrist"),
    ]
    for kp1, kp2 in pairs:
        dx = df[f"{kp1}_x"].values - df[f"{kp2}_x"].values
        dy = df[f"{kp1}_y"].values - df[f"{kp2}_y"].values
        df[f"dist_{kp1}_{kp2}"] = np.sqrt(dx**2 + dy**2)

    # --- Band Power from raw ECoG (500 Hz) ---
    with NWBHDF5IO(nwb_path, "r") as io:
        nwb = io.read()
        ecog_obj = nwb.acquisition["ElectricalSeries"]
        ecog_rate = ecog_obj.rate
        ecog_start = ecog_obj.starting_time if ecog_obj.starting_time else 0.0

        electrodes_df = nwb.electrodes.to_dataframe()
        good_indices = np.where(electrodes_df["good"].values)[0].tolist()
        selected_electrodes = good_indices[:10]

        s0 = int((start_sec - ecog_start) * ecog_rate)
        s1 = int((end_sec - ecog_start) * ecog_rate)
        raw_ecog = ecog_obj.data[s0:s1, :][:, selected_electrodes]

        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 100),
        }

        window_samples = int(ecog_rate)
        hop_samples = int(ecog_rate / 30.0)

        for band_name, (flo, fhi) in bands.items():
            sos = signal.butter(4, [flo, fhi], btype="band", fs=ecog_rate, output="sos")
            filtered = signal.sosfiltfilt(sos, raw_ecog, axis=0)
            power = filtered ** 2

            n_windows = len(timestamps)
            bp = np.zeros((n_windows, len(selected_electrodes)), dtype=np.float32)

            for i in range(n_windows):
                center = int((timestamps[i] - ecog_start - start_sec + ecog_start) * ecog_rate)
                center = int((timestamps[i] - start_sec) * ecog_rate)
                w_start = max(0, center - window_samples // 2)
                w_end = min(len(power), center + window_samples // 2)
                if w_end > w_start:
                    bp[i] = power[w_start:w_end].mean(axis=0)

            for j, elec_idx in enumerate(selected_electrodes):
                df[f"bandpower_{elec_idx}_{band_name}"] = bp[:, j]

    return df
```

### 6.3 Judge: ExecutableJudge (Continuous)

```python
# scoring_data/scoring_scripts/judge_step3.py
"""
Step 3 Judge: Feature engineering quality.
Score = weighted average of feature correlation with reference.
"""

import sys
import numpy as np
import pandas as pd


def judge_step3(submission_path: str, reference_path: str) -> float:
    try:
        submitted = pd.read_parquet(submission_path)
    except Exception as e:
        print(f"ERROR: Could not read submission: {e}", file=sys.stderr)
        return 0.0

    try:
        reference = pd.read_parquet(reference_path)
    except Exception as e:
        print(f"ERROR: Could not read reference: {e}", file=sys.stderr)
        return 0.0

    scores = []
    weights = []

    # --- Component 1: Velocity features (weight 0.20) ---
    score_vel = score_feature_group(submitted, reference, "velocity_")
    scores.append(score_vel)
    weights.append(0.20)

    # --- Component 2: Acceleration features (weight 0.15) ---
    score_acc = score_feature_group(submitted, reference, "acceleration_")
    scores.append(score_acc)
    weights.append(0.15)

    # --- Component 3: Inter-keypoint distances (weight 0.15) ---
    score_dist = score_feature_group(submitted, reference, "dist_")
    scores.append(score_dist)
    weights.append(0.15)

    # --- Component 4: Band power features (weight 0.40) ---
    score_bp = score_bandpower(submitted, reference)
    scores.append(score_bp)
    weights.append(0.40)

    # --- Component 5: Feature completeness (weight 0.10) ---
    score_complete = score_completeness(submitted, reference)
    scores.append(score_complete)
    weights.append(0.10)

    final_score = sum(s * w for s, w in zip(scores, weights))

    print(f"Step 3 breakdown:")
    print(f"  Velocity features:      {scores[0]:.4f} (weight {weights[0]})")
    print(f"  Acceleration features:  {scores[1]:.4f} (weight {weights[1]})")
    print(f"  Distance features:      {scores[2]:.4f} (weight {weights[2]})")
    print(f"  Band power features:    {scores[3]:.4f} (weight {weights[3]})")
    print(f"  Feature completeness:   {scores[4]:.4f} (weight {weights[4]})")
    print(f"  TOTAL:                  {final_score:.4f}")

    return float(np.clip(final_score, 0.0, 1.0))


def score_feature_group(submitted, reference, prefix):
    ref_cols = [c for c in reference.columns if c.startswith(prefix)]
    sub_cols = [c for c in submitted.columns if c.startswith(prefix)]

    if not ref_cols:
        return 1.0
    if not sub_cols:
        return 0.0

    common = sorted(set(ref_cols) & set(sub_cols))
    coverage = len(common) / len(ref_cols)

    if not common:
        return 0.0

    min_len = min(len(submitted), len(reference))
    correlations = []

    for col in common:
        sub_vals = submitted[col].values[:min_len].astype(float)
        ref_vals = reference[col].values[:min_len].astype(float)
        valid = ~(np.isnan(sub_vals) | np.isnan(ref_vals))
        if valid.sum() < 30:
            continue
        sv = sub_vals[valid]
        rv = ref_vals[valid]
        if sv.std() < 1e-10 or rv.std() < 1e-10:
            continue
        corr = np.corrcoef(sv, rv)[0, 1]
        if not np.isnan(corr):
            correlations.append(max(0.0, corr))

    if not correlations:
        return coverage * 0.3

    return coverage * np.mean(correlations)


def score_bandpower(submitted, reference):
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    ref_bp_cols = [c for c in reference.columns if c.startswith("bandpower_")]
    sub_bp_cols = [c for c in submitted.columns if c.startswith("bandpower_")]

    if not ref_bp_cols:
        return 1.0
    if not sub_bp_cols:
        return 0.0

    band_scores = []
    for band in bands:
        ref_band = [c for c in ref_bp_cols if c.endswith(f"_{band}")]
        sub_band = [c for c in sub_bp_cols if c.endswith(f"_{band}")]
        common = sorted(set(ref_band) & set(sub_band))

        if not ref_band:
            continue

        if not common:
            band_scores.append(0.0)
            continue

        min_len = min(len(submitted), len(reference))
        corrs = []
        for col in common:
            sv = submitted[col].values[:min_len].astype(float)
            rv = reference[col].values[:min_len].astype(float)
            valid = ~(np.isnan(sv) | np.isnan(rv))
            if valid.sum() < 30:
                continue
            s, r = sv[valid], rv[valid]
            if s.std() < 1e-10 or r.std() < 1e-10:
                continue
            c = np.corrcoef(s, r)[0, 1]
            if not np.isnan(c):
                corrs.append(max(0.0, c))

        if corrs:
            coverage = len(common) / len(ref_band)
            band_scores.append(coverage * np.mean(corrs))
        else:
            band_scores.append(0.0)

    if not band_scores:
        return 0.0

    return np.mean(band_scores)


def score_completeness(submitted, reference):
    ref_groups = {
        "velocity": len([c for c in reference.columns if c.startswith("velocity_")]),
        "acceleration": len([c for c in reference.columns if c.startswith("acceleration_")]),
        "distance": len([c for c in reference.columns if c.startswith("dist_")]),
        "bandpower": len([c for c in reference.columns if c.startswith("bandpower_")]),
    }
    sub_groups = {
        "velocity": len([c for c in submitted.columns if c.startswith("velocity_")]),
        "acceleration": len([c for c in submitted.columns if c.startswith("acceleration_")]),
        "distance": len([c for c in submitted.columns if c.startswith("dist_")]),
        "bandpower": len([c for c in submitted.columns if c.startswith("bandpower_")]),
    }

    group_scores = []
    for grp in ref_groups:
        if ref_groups[grp] == 0:
            group_scores.append(1.0)
        elif sub_groups[grp] == 0:
            group_scores.append(0.0)
        else:
            group_scores.append(min(1.0, sub_groups[grp] / ref_groups[grp]))

    return np.mean(group_scores) if group_scores else 0.0


if __name__ == "__main__":
    submission_path = "/workspace/submission/step3_features.parquet"
    reference_path = "/workspace/scoring_data/reference/step3_features.parquet"
    score = judge_step3(submission_path, reference_path)
    print(f"SCORE:{score:.4f}")
```

### 6.4 What's in env_data vs scoring_data

| Location | Contents |
|----------|----------|
| env_data | NWB files (for raw 500 Hz ECoG access); optionally `step2_reference_aligned.parquet` (fallback) |
| scoring_data | `reference/step3_features.parquet` (reference feature DataFrame) |

### 6.5 Anti-Gaming Measures

1. **Nyquist trap** -- the aligned data is at 30 Hz, so Nyquist = 15 Hz. Computing beta (13-30 Hz) or gamma (30-100 Hz) from the downsampled data is WRONG. The student must go back to the raw 500 Hz data. This catches naive agents that just FFT the downsampled signal.
2. **Correlation check** -- features must correlate with the reference (not just be non-zero). Random noise scores ~0.
3. **Multiple feature groups** -- cannot ace the task with just velocity; band power is 40% of the score.
4. **Coverage weighting** -- computing only 1 of 5 bands yields partial credit but not much.

### 6.6 Expected Score Distribution

| Score Range | Interpretation | Expected % |
|-------------|---------------|------------|
| 0.80 -- 1.0 | Correct band power from raw ECoG, all features match | 10% |
| 0.60 -- 0.79 | Good velocity/distance, but band power only for low freqs | 25% |
| 0.40 -- 0.59 | Velocity correct, band power wrong or missing | 30% |
| 0.20 -- 0.39 | Some features present, many wrong or low correlation | 20% |
| 0.00 -- 0.19 | Mostly broken | 15% |

---

## 7. Step 4: Prediction

### 7.1 Task Instructions (shown to student)

```
STEP 4: Movement Prediction (Cross-Subject)
=============================================

Build a binary classifier that predicts whether a given time point
is during a reaching movement (is_reach == True) or not.

TRAINING DATA:
Use data from sub-01 and sub-02 (all sessions) in /workspace/env_data/.
You may use the features you computed in Step 3, or compute new features.

EVALUATION:
Your classifier will be evaluated on a HIDDEN session from sub-03
that is NOT in env_data. You must generalize across subjects.

WHAT TO SUBMIT:
Write a Python script at /workspace/submission/step4_predict.py
that implements the following function:

    def predict(nwb_path: str) -> np.ndarray:
        """
        Given a path to an NWB file, return a 1D numpy array of
        predicted probabilities (float, 0.0 to 1.0) for is_reach,
        one per pose frame.

        The function will be called with the hidden test NWB file.
        It must return predictions for ALL pose frames in the file.
        """

Your script will be executed in the container with all dependencies
available. It may load pre-trained models from /workspace/submission/.

CONSTRAINTS:
- Your predict() function must complete in under 10 minutes.
- You may train on sub-01 and sub-02 data.
- You MUST NOT assume any specific session structure for the test file.
- Metric: AUC-ROC on the hidden test data.

TIPS:
- Random forest and logistic regression are good baselines.
- Feature engineering matters more than model choice for this data.
- Cross-subject generalization is hard -- features that are
  electrode-index-specific may not transfer.
- Consider using anatomical features (band power by brain region)
  rather than raw electrode indices.

Time limit: 90 minutes recommended.
```

### 7.2 Judge: ExecutableJudge (Continuous)

```python
# scoring_data/scoring_scripts/judge_step4.py
"""
Step 4 Judge: Movement prediction AUC-ROC.
Score = AUC-ROC on hidden test subject (continuous 0-1).

The judge:
1. Loads the student's predict() function
2. Calls it with the hidden test NWB file
3. Computes AUC-ROC against ground truth reach labels
4. Applies penalties for trivial solutions
"""

import sys
import importlib.util
import traceback
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TIMEOUT_SECONDS = 600  # 10 minutes


def judge_step4(
    predict_script_path: str,
    test_nwb_path: str,
    test_labels_path: str,
) -> float:
    # --- Load ground truth ---
    try:
        labels_df = pd.read_parquet(test_labels_path)
        true_labels = labels_df["is_reach"].values.astype(int)
        n_frames = len(true_labels)
    except Exception as e:
        print(f"ERROR: Could not load test labels: {e}", file=sys.stderr)
        return 0.0

    # --- Load student's predict function ---
    try:
        spec = importlib.util.spec_from_file_location("student_predictor", predict_script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predict_fn = module.predict
    except Exception as e:
        print(f"ERROR: Could not load predict function: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 0.0

    # --- Run prediction ---
    try:
        start_time = time.time()
        predictions = predict_fn(test_nwb_path)
        elapsed = time.time() - start_time
        print(f"Prediction completed in {elapsed:.1f}s")

        if elapsed > TIMEOUT_SECONDS:
            print(f"WARNING: Prediction took {elapsed:.1f}s (limit: {TIMEOUT_SECONDS}s)")
            return 0.0
    except Exception as e:
        print(f"ERROR: predict() raised exception: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 0.0

    # --- Validate predictions ---
    predictions = np.asarray(predictions, dtype=float)

    if predictions.ndim != 1:
        print(f"ERROR: predictions must be 1D, got shape {predictions.shape}", file=sys.stderr)
        return 0.0

    if len(predictions) != n_frames:
        print(f"WARNING: prediction length {len(predictions)} != expected {n_frames}")
        min_len = min(len(predictions), n_frames)
        predictions = predictions[:min_len]
        true_labels = true_labels[:min_len]

    if np.any(np.isnan(predictions)):
        print("WARNING: NaN in predictions, replacing with 0.5")
        predictions = np.nan_to_num(predictions, nan=0.5)

    predictions = np.clip(predictions, 0.0, 1.0)

    # --- Anti-gaming checks ---
    penalty = 0.0

    # Check 1: constant predictions
    if predictions.std() < 1e-6:
        print("PENALTY: Constant predictions detected")
        return 0.0

    # Check 2: only two unique values (binary, not probabilistic)
    if len(np.unique(predictions)) <= 2:
        print("NOTE: Binary predictions (no probability calibration)")
        # Not a hard penalty, but AUC will be suboptimal

    # Check 3: predictions inversely correlated (predicting is_rest instead of is_reach)
    if np.corrcoef(predictions, true_labels.astype(float))[0, 1] < -0.1:
        print("NOTE: Predictions appear inverted, trying 1 - predictions")
        predictions_inv = 1.0 - predictions
        auc_orig = safe_auc(true_labels, predictions)
        auc_inv = safe_auc(true_labels, predictions_inv)
        if auc_inv > auc_orig:
            predictions = predictions_inv
            print("  Using inverted predictions (agent confused class labels)")

    # --- Compute AUC-ROC ---
    n_positive = true_labels.sum()
    n_negative = (1 - true_labels).sum()

    if n_positive == 0 or n_negative == 0:
        print("ERROR: Test set has only one class -- cannot compute AUC", file=sys.stderr)
        return 0.0

    auc = safe_auc(true_labels, predictions)

    if auc is None:
        return 0.0

    # Map AUC to score:
    # AUC 0.5 = random (score 0.0)
    # AUC 0.7 = decent (score 0.4)
    # AUC 0.8 = good (score 0.6)
    # AUC 0.85 = very good (score 0.75)
    # AUC 0.9 = excellent (score 0.9)
    # AUC 0.95+ = outstanding (score 1.0)
    score = auc_to_score(auc)
    score = max(0.0, score - penalty)

    print(f"Step 4 results:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  n_positive: {n_positive}, n_negative: {n_negative}")
    print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  SCORE: {score:.4f}")

    return float(np.clip(score, 0.0, 1.0))


def safe_auc(true_labels, predictions):
    try:
        return roc_auc_score(true_labels, predictions)
    except ValueError as e:
        print(f"ERROR: AUC computation failed: {e}", file=sys.stderr)
        return None


def auc_to_score(auc):
    if auc <= 0.5:
        return 0.0
    elif auc <= 0.7:
        return 0.4 * (auc - 0.5) / 0.2
    elif auc <= 0.8:
        return 0.4 + 0.2 * (auc - 0.7) / 0.1
    elif auc <= 0.85:
        return 0.6 + 0.15 * (auc - 0.8) / 0.05
    elif auc <= 0.9:
        return 0.75 + 0.15 * (auc - 0.85) / 0.05
    elif auc <= 0.95:
        return 0.9 + 0.1 * (auc - 0.9) / 0.05
    else:
        return 1.0


if __name__ == "__main__":
    predict_script = "/workspace/submission/step4_predict.py"
    test_nwb = "/workspace/scoring_data/sub-03/sub-03_ses-3_slice.nwb"
    test_labels = "/workspace/scoring_data/reference/step4_labels.parquet"
    score = judge_step4(predict_script, test_nwb, test_labels)
    print(f"SCORE:{score:.4f}")
```

### 7.3 What's in env_data vs scoring_data

| Location | Contents |
|----------|----------|
| env_data | Training NWB files for sub-01 (x3 sessions), sub-02 (x3), sub-03 (x2 sessions visible) |
| scoring_data | Hidden `sub-03_ses-3_slice.nwb` + `reference/step4_labels.parquet` (true reach labels for that session) |

### 7.4 Anti-Gaming Measures

1. **Hidden test subject session** -- sub-03 ses-3 is NEVER in env_data. The student has 2 sessions of sub-03 for validation but is evaluated on a session they cannot see.
2. **Cross-subject generalization required** -- training on sub-01/02 and testing on sub-03 means electrode-index features don't transfer (different patients have different electrode placements). The agent must discover anatomical or frequency-band features that generalize.
3. **Constant prediction detection** -- if std(predictions) < 1e-6, score = 0.
4. **Inverted label detection** -- if the agent accidentally predicts is_rest instead of is_reach, the judge checks and corrects. This is generous, not a penalty.
5. **AUC-ROC metric** -- threshold-invariant, so the agent cannot game by choosing a lucky threshold.
6. **10-minute timeout** -- prevents the agent from doing exhaustive grid search or brute-force approaches.

### 7.5 Expected Score Distribution

| AUC Range | pm_env Score | Interpretation | Expected % |
|-----------|-------------|---------------|------------|
| 0.90 -- 1.0 | 0.90 -- 1.0 | Excellent cross-subject transfer | 5% |
| 0.80 -- 0.89 | 0.60 -- 0.89 | Good features, decent generalization | 20% |
| 0.70 -- 0.79 | 0.40 -- 0.59 | Basic features work, some transfer | 30% |
| 0.55 -- 0.69 | 0.10 -- 0.39 | Weak features or overfitting to training subjects | 25% |
| 0.50 -- 0.54 | 0.00 -- 0.09 | Near random | 15% |
| < 0.50 | 0.00 | Worse than random (inverted) | 5% |

---

## 8. tasks.py Skeleton

```python
# tasks.py
"""
E01: Messy Multi-Format Data Science (AJILE12)
pm_env task definition with 4 progressive steps.
"""

import json
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path("/workspace")
ENV_DATA = WORKSPACE / "env_data"
SCORING_DATA = WORKSPACE / "scoring_data"
SUBMISSION = WORKSPACE / "submission"
SCORING_SCRIPTS = SCORING_DATA / "scoring_scripts"


def get_tasks():
    return [
        {
            "id": "ajile12_wrangling",
            "name": "AJILE12 Multi-Format Neural Data Wrangling",
            "description": (
                "Explore, align, featurize, and predict from multi-modal "
                "neuroscience data (ECoG + pose tracking). Four progressive "
                "steps from data exploration to cross-subject prediction."
            ),
            "time_limit_seconds": 14400,  # 4 hours
            "steps": [
                step1_config(),
                step2_config(),
                step3_config(),
                step4_config(),
            ],
        }
    ]


def step1_config():
    return {
        "id": "data_exploration",
        "name": "Step 1: Data Exploration and Schema Discovery",
        "instructions": STEP1_INSTRUCTIONS,
        "judge": {
            "type": "ExecutableJudge",
            "script": str(SCORING_SCRIPTS / "judge_step1.py"),
            "timeout_seconds": 60,
        },
        "submission_path": str(SUBMISSION / "step1_answers.json"),
    }


def step2_config():
    return {
        "id": "align_multirate",
        "name": "Step 2: Load and Align Multi-Rate Data",
        "instructions": STEP2_INSTRUCTIONS,
        "judge": {
            "type": "ExecutableJudge",
            "script": str(SCORING_SCRIPTS / "judge_step2.py"),
            "timeout_seconds": 300,
        },
        "submission_path": str(SUBMISSION / "step2_aligned.parquet"),
        "post_judge_hook": "provide_step2_reference",
    }


def step3_config():
    return {
        "id": "feature_engineering",
        "name": "Step 3: Feature Engineering",
        "instructions": STEP3_INSTRUCTIONS,
        "judge": {
            "type": "ExecutableJudge",
            "script": str(SCORING_SCRIPTS / "judge_step3.py"),
            "timeout_seconds": 300,
        },
        "submission_path": str(SUBMISSION / "step3_features.parquet"),
        "post_judge_hook": "provide_step3_reference",
    }


def step4_config():
    return {
        "id": "prediction",
        "name": "Step 4: Movement Prediction (Cross-Subject)",
        "instructions": STEP4_INSTRUCTIONS,
        "judge": {
            "type": "ExecutableJudge",
            "script": str(SCORING_SCRIPTS / "judge_step4.py"),
            "timeout_seconds": 660,  # 10 min for predict() + 1 min overhead
        },
        "submission_path": str(SUBMISSION / "step4_predict.py"),
    }


def provide_step2_reference():
    """After Step 2 grading, copy reference alignment to env_data for Step 3."""
    src = SCORING_DATA / "reference" / "step2_aligned.parquet"
    dst = ENV_DATA / "step2_reference_aligned.parquet"
    if src.exists() and not dst.exists():
        import shutil
        shutil.copy2(src, dst)


def provide_step3_reference():
    """After Step 3 grading, copy reference features to env_data for Step 4."""
    src = SCORING_DATA / "reference" / "step3_features.parquet"
    dst = ENV_DATA / "step3_reference_features.parquet"
    if src.exists() and not dst.exists():
        import shutil
        shutil.copy2(src, dst)


# ============================================================
# STEP INSTRUCTION TEXTS
# ============================================================

STEP1_INSTRUCTIONS = """
STEP 1: Data Exploration and Schema Discovery
==============================================

You have been given a set of NWB files in /workspace/env_data/.
These files contain neuroscience data, but you have NO documentation
about their internal structure.

Your task: explore the files and answer the following questions.

Write your answers to /workspace/submission/step1_answers.json
in the following format:

{
    "num_subjects": <int>,
    "sessions_per_subject": {"sub-01": <int>, "sub-02": <int>, "sub-03": <int>},
    "ecog_sampling_rate_hz": <number>,
    "num_pose_keypoints": <int>,
    "electrode_table_columns": [<list of column name strings, sorted alphabetically>],
    "num_good_electrodes_sub01_ses1": <int>,
    "pose_data_unit": <string>,
    "num_reach_events_sub01_ses1": <int>
}

Hints:
- The files are in NWB format (HDF5-based). Use h5py or pynwb to open them.
- Start by listing what is inside one file, then generalize.
- Be precise: column names must match exactly (case-sensitive, sorted).

Time limit: 30 minutes recommended (of the 4 hour total).
""".strip()


STEP2_INSTRUCTIONS = """
STEP 2: Load and Align Multi-Rate Data
========================================

Given the NWB files you explored in Step 1, produce an aligned
DataFrame combining ECoG and pose data at a common 30 Hz timebase.

SPECIFIC REQUEST:
- Subject: sub-01, Session: ses-1
- Time window: the FIRST 5 MINUTES (0 to 300 seconds from session start)
- Include ALL good electrodes (where the 'good' column is True)
- Include ALL 9 pose keypoints

REQUIREMENTS:
1. Downsample ECoG from 500 Hz to 30 Hz by averaging samples within
   each 33.3ms window centered on each pose timestamp.
2. Use pose timestamps as the reference timebase.
3. Exclude bad electrodes (good == False).
4. Handle any missing pose data with NaN.
5. Include movement labels from the reaches interval table.

OUTPUT FORMAT:
Save a Parquet file to /workspace/submission/step2_aligned.parquet

The DataFrame must have these columns:
- timestamp: float64 (seconds, from pose timestamps)
- ecog_{electrode_index}: float32 (one column per good electrode,
  e.g., ecog_0, ecog_1, ..., ecog_51)
- {keypoint}_x: float64 (e.g., nose_x, left_ear_x, ...)
- {keypoint}_y: float64 (e.g., nose_y, left_ear_y, ...)
- is_reach: bool (True if timestamp falls within a reach interval)

Keypoint names (in order): nose, left_ear, right_ear, left_shoulder,
right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist

Expected row count: approximately 9000 rows (5 min x 60 sec x 30 Hz)

Time limit: 45 minutes recommended.
""".strip()


STEP3_INSTRUCTIONS = """
STEP 3: Feature Engineering
============================

Starting from aligned neural + pose data, compute neuroscience features.

INPUT:
If your Step 2 output exists at /workspace/submission/step2_aligned.parquet
and is valid, use it. Otherwise, use the reference alignment provided at
/workspace/env_data/step2_reference_aligned.parquet.

REQUIRED FEATURES (compute all of these):

1. ECoG Band Power (per electrode, up to first 10 good electrodes):
   - delta:  1-4 Hz
   - theta:  4-8 Hz
   - alpha:  8-13 Hz
   - beta:   13-30 Hz
   - gamma:  30-100 Hz

   IMPORTANT: The aligned data is at 30 Hz (Nyquist = 15 Hz). You can
   only compute delta/theta/alpha from the downsampled data. For beta
   and gamma, you MUST use the original NWB files at 500 Hz. Filter
   the raw signal, compute power, then downsample to 30 Hz.

   Column names: bandpower_{electrode_index}_{band}

2. Movement Velocity (per keypoint):
   - Frame-to-frame displacement / dt
   Column names: velocity_{keypoint}

3. Movement Acceleration (per keypoint):
   - First derivative of velocity
   Column names: acceleration_{keypoint}

4. Inter-Keypoint Distances (5 pairs):
   - left_wrist to right_wrist
   - nose to left_wrist
   - nose to right_wrist
   - left_shoulder to left_wrist
   - right_shoulder to right_wrist
   Column names: dist_{kp1}_{kp2}

OUTPUT FORMAT:
Save to /workspace/submission/step3_features.parquet

Include: timestamp, all features above, and is_reach.

Time limit: 60 minutes recommended.
""".strip()


STEP4_INSTRUCTIONS = """
STEP 4: Movement Prediction (Cross-Subject)
=============================================

Build a binary classifier that predicts whether each time point
is during a reaching movement (is_reach == True) or not.

TRAINING DATA:
Use data from sub-01 and sub-02 (all sessions) in /workspace/env_data/.
You may compute any features you want from the NWB files.

EVALUATION:
Your classifier will be evaluated on a HIDDEN session from sub-03.
You must generalize across subjects.

WHAT TO SUBMIT:
Write a Python script at /workspace/submission/step4_predict.py
that implements:

    import numpy as np

    def predict(nwb_path: str) -> np.ndarray:
        \"\"\"
        Given a path to an NWB file, return a 1D numpy array of
        predicted probabilities (float, 0 to 1) for is_reach,
        one per pose frame in the file.
        \"\"\"

Your script may load saved models from /workspace/submission/.
It will be executed with all container dependencies available.

CONSTRAINTS:
- predict() must complete in under 10 minutes.
- Metric: AUC-ROC on the hidden test data.

TIPS:
- Features that depend on specific electrode indices will NOT transfer
  across subjects (different patients, different electrode placements).
- Consider brain-region-based features or frequency-band summaries.
- Random forest with hand-crafted features is a strong baseline.
- Logistic regression with band power features is also reasonable.

Time limit: 90 minutes recommended.
""".strip()
```

---

## 9. setup_data.py (Pre-Download)

This script runs BEFORE the container is built. It downloads data from DANDI, extracts 1-hour slices, and generates all reference outputs.

```python
#!/usr/bin/env python3
# setup_data.py
"""
E01 AJILE12 Data Setup
======================
Downloads AJILE12 data from DANDI Archive and prepares env_data/ and scoring_data/.

Run this ONCE before building the container image.
Requires network access (DANDI API).

Usage:
    uv run setup_data.py --output-dir /path/to/build/context

Output structure:
    output_dir/
    ├── env_data/
    │   ├── sub-01/ (3 session slices)
    │   ├── sub-02/ (3 session slices)
    │   └── sub-03/ (2 session slices -- ses-3 hidden)
    └── scoring_data/
        ├── sub-03/sub-03_ses-3_slice.nwb
        └── reference/
            ├── step1_answers.json
            ├── step2_aligned.parquet
            ├── step3_features.parquet
            └── step4_labels.parquet
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.file import Subject
from datetime import datetime
from dateutil.tz import tzlocal

try:
    import remfile
except ImportError:
    remfile = None

DANDISET_ID = "000055"

SUBJECTS = {
    "sub-01": {"sessions": [1, 2, 3], "visible": [1, 2, 3]},
    "sub-02": {"sessions": [1, 2, 3], "visible": [1, 2, 3]},
    "sub-03": {"sessions": [1, 2, 3], "visible": [1, 2]},  # ses-3 is hidden
}

SLICE_DURATION_SEC = 3600  # 1 hour
SLICE_START_SEC = 300      # start 5 min into recording to avoid edge artifacts


def get_s3_url(subject_id: str, session_id: int) -> str:
    filename = f"{subject_id}/{subject_id}_ses-{session_id}_behavior+ecephys.nwb"
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(DANDISET_ID)
        asset = dandiset.get_asset_by_path(filename)
        return asset.get_content_url(follow_redirects=1, strip_query=True)


def open_remote_nwb(s3_url: str):
    if remfile is not None:
        cache = remfile.DiskCache("/tmp/nwb_setup_cache")
        rf = remfile.File(s3_url, disk_cache=cache)
    else:
        import fsspec
        fs = fsspec.filesystem("http")
        rf = fs.open(s3_url, "rb")

    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwb = io.read()
    return nwb, h5, io


def extract_slice(nwb, start_sec: float, end_sec: float, output_path: Path):
    """Extract a time slice from a remote NWB and write a local NWB file."""

    ecog = nwb.acquisition["ElectricalSeries"]
    ecog_rate = ecog.rate
    ecog_start = ecog.starting_time if ecog.starting_time else 0.0

    s0 = int((start_sec - ecog_start) * ecog_rate)
    s1 = int((end_sec - ecog_start) * ecog_rate)
    s1 = min(s1, ecog.data.shape[0])
    ecog_data = ecog.data[s0:s1, :]

    position = nwb.processing["behavior"]["Position"]
    spatial = list(position.spatial_series.values())[0]
    pose_ts_all = spatial.timestamps[:]
    pose_data_all = spatial.data[:]

    pose_mask = (pose_ts_all >= start_sec) & (pose_ts_all < end_sec)
    pose_ts = pose_ts_all[pose_mask]
    pose_data = pose_data_all[pose_mask]

    electrodes_df = nwb.electrodes.to_dataframe()

    reaches_df = nwb.intervals["reaches"].to_dataframe()
    reaches_slice = reaches_df[
        (reaches_df["stop_time"] > start_sec) & (reaches_df["start_time"] < end_sec)
    ].copy()

    has_epochs = "epochs" in nwb.intervals
    if has_epochs:
        epochs_df = nwb.intervals["epochs"].to_dataframe()
        epochs_slice = epochs_df[
            (epochs_df["stop_time"] > start_sec) & (epochs_df["start_time"] < end_sec)
        ].copy()

    # Write new NWB file
    nwb_out = NWBFile(
        session_description=f"AJILE12 slice [{start_sec}-{end_sec}]s",
        identifier=f"{output_path.stem}_{start_sec}_{end_sec}",
        session_start_time=datetime.now(tzlocal()),
    )

    device = nwb_out.create_device(name="ECoG_grid")
    group = nwb_out.create_electrode_group(
        name="all_electrodes",
        description="ECoG electrodes",
        location="cortex",
        device=device,
    )

    for _, row in electrodes_df.iterrows():
        nwb_out.add_electrode(
            x=float(row.get("x", 0)),
            y=float(row.get("y", 0)),
            z=float(row.get("z", 0)),
            group=group,
            group_name=str(row.get("group_name", "")),
            location=str(row.get("location", "")),
        )
    electrode_extra_cols = ["good", "std", "kurtosis", "mad", "lfb_r2", "hfb_r2"]
    for col in electrode_extra_cols:
        if col in electrodes_df.columns:
            nwb_out.add_electrode_column(
                name=col,
                description=col,
                data=electrodes_df[col].tolist(),
            )

    electrode_region = nwb_out.create_electrode_table_region(
        region=list(range(len(electrodes_df))),
        description="all electrodes",
    )

    es = ElectricalSeries(
        name="ElectricalSeries",
        data=ecog_data,
        electrodes=electrode_region,
        starting_time=start_sec,
        rate=ecog_rate,
    )
    nwb_out.add_acquisition(es)

    behavior_module = nwb_out.create_processing_module("behavior", "Pose tracking")
    pos = Position(name="Position")
    ss = SpatialSeries(
        name="SpatialSeries",
        data=pose_data,
        timestamps=pose_ts,
        reference_frame="video pixel coordinates",
        unit="pixels",
    )
    pos.add_spatial_series(ss)
    behavior_module.add(pos)

    for _, row in reaches_slice.iterrows():
        nwb_out.add_trial(
            start_time=float(row["start_time"]),
            stop_time=float(row["stop_time"]),
        )
    if len(reaches_slice) > 0:
        extra_reach_cols = ["reach_magnitude", "reach_duration", "reach_angle"]
        for col in extra_reach_cols:
            if col in reaches_slice.columns:
                nwb_out.add_trial_column(
                    name=col,
                    description=col,
                    data=reaches_slice[col].tolist(),
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(output_path), "w") as io_out:
        io_out.write(nwb_out)

    print(f"  Written: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return electrodes_df, reaches_slice, pose_ts, pose_data


def generate_step1_answers(env_data_dir: Path) -> dict:
    """Generate ground truth for Step 1 from the actual extracted files."""
    answers = {}

    subjects = sorted([d.name for d in env_data_dir.iterdir() if d.is_dir()])
    answers["num_subjects"] = len(subjects)

    sessions_per = {}
    for subj_dir in sorted(env_data_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        nwb_files = sorted(subj_dir.glob("*.nwb"))
        sessions_per[subj_dir.name] = len(nwb_files)
    answers["sessions_per_subject"] = sessions_per

    first_file = sorted((env_data_dir / "sub-01").glob("*.nwb"))[0]
    with NWBHDF5IO(str(first_file), "r") as io:
        nwb = io.read()

        ecog = nwb.acquisition["ElectricalSeries"]
        answers["ecog_sampling_rate_hz"] = float(ecog.rate)

        position = nwb.processing["behavior"]["Position"]
        spatial = list(position.spatial_series.values())[0]
        if spatial.data.ndim == 3:
            answers["num_pose_keypoints"] = spatial.data.shape[1]
        elif spatial.data.ndim == 2:
            answers["num_pose_keypoints"] = len(position.spatial_series)

        electrodes_df = nwb.electrodes.to_dataframe()
        col_names = sorted(electrodes_df.columns.tolist())
        answers["electrode_table_columns"] = col_names

        answers["num_good_electrodes_sub01_ses1"] = int(electrodes_df["good"].sum())

        answers["pose_data_unit"] = spatial.unit if spatial.unit else "pixels"

        trials_df = nwb.trials.to_dataframe() if nwb.trials else pd.DataFrame()
        answers["num_reach_events_sub01_ses1"] = len(trials_df)

    return answers


def generate_step2_reference(env_data_dir: Path) -> pd.DataFrame:
    """Generate reference aligned DataFrame for Step 2."""
    nwb_path = sorted((env_data_dir / "sub-01").glob("*.nwb"))[0]

    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()

        ecog = nwb.acquisition["ElectricalSeries"]
        ecog_rate = ecog.rate
        ecog_start = ecog.starting_time if ecog.starting_time else 0.0

        position = nwb.processing["behavior"]["Position"]
        spatial = list(position.spatial_series.values())[0]
        pose_ts_all = spatial.timestamps[:]
        pose_data_all = spatial.data[:]

        electrodes_df = nwb.electrodes.to_dataframe()
        good_mask = electrodes_df["good"].values
        good_indices = np.where(good_mask)[0].tolist()

        trials_df = nwb.trials.to_dataframe() if nwb.trials else pd.DataFrame()

        start_sec = ecog_start
        end_sec = ecog_start + 300.0

        pose_mask = (pose_ts_all >= start_sec) & (pose_ts_all < end_sec)
        pose_ts = pose_ts_all[pose_mask]
        pose_data = pose_data_all[pose_mask]

        s0 = 0
        s1 = int(300.0 * ecog_rate)
        s1 = min(s1, ecog.data.shape[0])
        ecog_data = np.array(ecog.data[s0:s1, :])
        ecog_ts = ecog_start + np.arange(s0, s1) / ecog_rate

        half_window = 0.5 / 30.0
        n_pose = len(pose_ts)
        n_good = len(good_indices)
        aligned_ecog = np.full((n_pose, n_good), np.nan, dtype=np.float32)

        for i, t in enumerate(pose_ts):
            mask = (ecog_ts >= t - half_window) & (ecog_ts < t + half_window)
            if mask.sum() > 0:
                aligned_ecog[i] = ecog_data[mask][:, good_indices].mean(axis=0)

        df = pd.DataFrame()
        df["timestamp"] = pose_ts

        for j, idx in enumerate(good_indices):
            df[f"ecog_{idx}"] = aligned_ecog[:, j]

        keypoint_names = [
            "nose", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        ]
        for k, name in enumerate(keypoint_names):
            if pose_data.ndim == 3 and k < pose_data.shape[1]:
                df[f"{name}_x"] = pose_data[:, k, 0]
                df[f"{name}_y"] = pose_data[:, k, 1]
            else:
                df[f"{name}_x"] = np.nan
                df[f"{name}_y"] = np.nan

        df["is_reach"] = False
        if len(trials_df) > 0:
            for _, trial in trials_df.iterrows():
                if trial["stop_time"] < start_sec or trial["start_time"] >= end_sec:
                    continue
                rmask = (df["timestamp"] >= trial["start_time"]) & (
                    df["timestamp"] <= trial["stop_time"]
                )
                df.loc[rmask, "is_reach"] = True

    return df


def generate_step4_labels(scoring_nwb_path: Path) -> pd.DataFrame:
    """Generate reach labels for the hidden test session."""
    with NWBHDF5IO(str(scoring_nwb_path), "r") as io:
        nwb = io.read()

        position = nwb.processing["behavior"]["Position"]
        spatial = list(position.spatial_series.values())[0]
        pose_ts = spatial.timestamps[:]

        trials_df = nwb.trials.to_dataframe() if nwb.trials else pd.DataFrame()

        df = pd.DataFrame()
        df["timestamp"] = pose_ts
        df["is_reach"] = False

        if len(trials_df) > 0:
            for _, trial in trials_df.iterrows():
                rmask = (df["timestamp"] >= trial["start_time"]) & (
                    df["timestamp"] <= trial["stop_time"]
                )
                df.loc[rmask, "is_reach"] = True

    return df


def main():
    parser = argparse.ArgumentParser(description="Setup E01 AJILE12 data")
    parser.add_argument("--output-dir", required=True, help="Build context directory")
    args = parser.parse_args()

    output = Path(args.output_dir)
    env_data = output / "env_data"
    scoring_data = output / "scoring_data"
    reference = scoring_data / "reference"
    scoring_scripts = scoring_data / "scoring_scripts"

    for d in [env_data, scoring_data, reference, scoring_scripts]:
        d.mkdir(parents=True, exist_ok=True)

    (output / "submission").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("E01 AJILE12 Data Setup")
    print("=" * 60)

    for subject_id, config in SUBJECTS.items():
        for ses_id in config["sessions"]:
            print(f"\nProcessing {subject_id} ses-{ses_id}...")

            try:
                s3_url = get_s3_url(subject_id, ses_id)
                nwb, h5, io = open_remote_nwb(s3_url)

                start = SLICE_START_SEC
                end = SLICE_START_SEC + SLICE_DURATION_SEC

                if ses_id in config["visible"]:
                    out_path = env_data / subject_id / f"{subject_id}_ses-{ses_id}_slice.nwb"
                else:
                    out_path = scoring_data / subject_id / f"{subject_id}_ses-{ses_id}_slice.nwb"

                extract_slice(nwb, start, end, out_path)

                io.close()
                h5.close()
            except Exception as e:
                print(f"  ERROR: {e}")
                raise

    # --- Generate reference outputs ---
    print("\n" + "=" * 60)
    print("Generating reference outputs...")
    print("=" * 60)

    # Step 1
    print("\nStep 1: answers...")
    answers = generate_step1_answers(env_data)
    with open(reference / "step1_answers.json", "w") as f:
        json.dump(answers, f, indent=2)
    print(f"  Written: {reference / 'step1_answers.json'}")

    # Step 2
    print("\nStep 2: aligned DataFrame...")
    aligned_df = generate_step2_reference(env_data)
    aligned_df.to_parquet(reference / "step2_aligned.parquet")
    print(f"  Written: {reference / 'step2_aligned.parquet'} ({len(aligned_df)} rows)")

    # Step 3 (requires aligned data + raw NWB for band power)
    # NOTE: generate_step3_reference would follow the same pattern as
    # compute_reference_features() shown in Section 6.2.
    # For brevity, we call it here as a placeholder.
    print("\nStep 3: features... (placeholder -- implement compute_reference_features)")
    # step3_df = compute_reference_features(
    #     aligned_df,
    #     str(sorted((env_data / "sub-01").glob("*.nwb"))[0]),
    #     ecog_start, ecog_start + 300.0
    # )
    # step3_df.to_parquet(reference / "step3_features.parquet")

    # Step 4
    print("\nStep 4: test labels...")
    test_nwb = scoring_data / "sub-03" / "sub-03_ses-3_slice.nwb"
    if test_nwb.exists():
        labels_df = generate_step4_labels(test_nwb)
        labels_df.to_parquet(reference / "step4_labels.parquet")
        print(f"  Written: {reference / 'step4_labels.parquet'} ({len(labels_df)} rows)")

    # --- Copy scoring scripts ---
    print("\nCopying scoring scripts...")
    # In production, these would be copied from the repo.
    # Here we note where they live.
    print("  judge_step1.py -> scoring_scripts/")
    print("  judge_step2.py -> scoring_scripts/")
    print("  judge_step3.py -> scoring_scripts/")
    print("  judge_step4.py -> scoring_scripts/")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"  env_data:     {env_data}")
    print(f"  scoring_data: {scoring_data}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 10. env_requirements.txt

```
# Core scientific computing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0

# NWB / HDF5
pynwb>=2.5.0
h5py>=3.9.0
hdmf>=3.9.0

# Data formats
pyarrow>=12.0.0

# Signal processing (already in scipy but explicit)
# scipy covers this

# Plotting (optional but common)
matplotlib>=3.7.0

# Utilities
tqdm>=4.65.0
```

NOTE: `dandi`, `remfile`, and `fsspec` are NOT included because the container has no network access. All data is pre-downloaded.

---

## 11. pre_hook Implementation

The pre_hook runs at container startup before the student begins. It validates data integrity and sets up the submission directory.

```python
# pre_hook.py
"""
Pre-hook: runs at container startup before student interaction.
Validates data integrity and prepares workspace.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np


WORKSPACE = Path("/workspace")
ENV_DATA = WORKSPACE / "env_data"
SCORING_DATA = WORKSPACE / "scoring_data"
SUBMISSION = WORKSPACE / "submission"


def validate_nwb_file(path: Path) -> bool:
    try:
        with h5py.File(str(path), "r") as h5:
            assert "acquisition" in h5, f"Missing /acquisition in {path.name}"
            assert "ElectricalSeries" in h5["acquisition"], f"Missing ElectricalSeries"
            assert "processing" in h5, f"Missing /processing"
            assert "behavior" in h5["processing"], f"Missing /processing/behavior"
        return True
    except Exception as e:
        print(f"VALIDATION FAILED for {path}: {e}", file=sys.stderr)
        return False


def pre_hook():
    print("=" * 50)
    print("E01 AJILE12 Environment Pre-Hook")
    print("=" * 50)

    # 1. Create submission directory
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Submission directory: {SUBMISSION}")

    # 2. Validate env_data exists and has expected structure
    assert ENV_DATA.exists(), f"env_data directory missing: {ENV_DATA}"

    subject_dirs = sorted([d for d in ENV_DATA.iterdir() if d.is_dir()])
    assert len(subject_dirs) >= 2, f"Expected >=2 subject dirs, found {len(subject_dirs)}"

    nwb_files = list(ENV_DATA.glob("**/*.nwb"))
    assert len(nwb_files) >= 6, f"Expected >=6 NWB files, found {len(nwb_files)}"
    print(f"[OK] Found {len(nwb_files)} NWB files across {len(subject_dirs)} subjects")

    # 3. Validate each NWB file
    all_valid = True
    for nwb_path in sorted(nwb_files):
        if not validate_nwb_file(nwb_path):
            all_valid = False
            print(f"[FAIL] {nwb_path.name}")
        else:
            print(f"[OK]   {nwb_path.name}")

    if not all_valid:
        print("\nWARNING: Some NWB files failed validation!", file=sys.stderr)
        print("The environment may not work correctly.", file=sys.stderr)

    # 4. Validate scoring_data exists
    assert SCORING_DATA.exists(), f"scoring_data directory missing"
    ref_dir = SCORING_DATA / "reference"
    assert ref_dir.exists(), f"reference directory missing"

    required_refs = [
        "step1_answers.json",
        "step2_aligned.parquet",
        "step4_labels.parquet",
    ]
    for ref_file in required_refs:
        path = ref_dir / ref_file
        if path.exists():
            print(f"[OK]   Reference: {ref_file}")
        else:
            print(f"[WARN] Missing reference: {ref_file}", file=sys.stderr)

    # 5. Verify scoring scripts
    scripts_dir = SCORING_DATA / "scoring_scripts"
    for step in [1, 2, 3, 4]:
        script = scripts_dir / f"judge_step{step}.py"
        if script.exists():
            print(f"[OK]   Scoring script: judge_step{step}.py")
        else:
            print(f"[WARN] Missing: judge_step{step}.py", file=sys.stderr)

    # 6. Print workspace summary
    print("\n" + "=" * 50)
    print("Workspace ready.")
    print(f"  env_data:    {sum(1 for _ in ENV_DATA.glob('**/*.nwb'))} NWB files")
    print(f"  submission:  {SUBMISSION}")
    print("  No network access available.")
    print("=" * 50)

    return True


if __name__ == "__main__":
    try:
        pre_hook()
    except AssertionError as e:
        print(f"PRE-HOOK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
```

---

## 12. Expected Score Distributions

### 12.1 Overall Task Score

The overall task score is the average of the 4 step scores (or a weighted average if the framework supports it). Suggested weights:

| Step | Weight | Rationale |
|------|--------|-----------|
| Step 1 (Exploration) | 0.15 | Straightforward for capable agents |
| Step 2 (Alignment) | 0.25 | Core data engineering skill |
| Step 3 (Features) | 0.25 | Domain knowledge + engineering |
| Step 4 (Prediction) | 0.35 | End-to-end ML, hardest to game |

### 12.2 Score Distribution by Agent Capability

| Agent Level | Step 1 | Step 2 | Step 3 | Step 4 | Overall |
|-------------|--------|--------|--------|--------|---------|
| Random/broken | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| Weak agent | 0.5 | 0.2 | 0.1 | 0.0 | 0.10 |
| Median agent | 0.88 | 0.55 | 0.40 | 0.30 | 0.44 |
| Good agent | 1.0 | 0.75 | 0.60 | 0.55 | 0.66 |
| Strong agent | 1.0 | 0.90 | 0.80 | 0.75 | 0.84 |
| Expert/human | 1.0 | 0.95 | 0.90 | 0.90 | 0.93 |

### 12.3 Why This Distribution Is Good for RL Training

1. **Not binary** -- most agents land between 0.1 and 0.8, providing gradient signal
2. **Progressive difficulty** -- easy win on Step 1, hard ceiling on Step 4
3. **Many failure modes** -- different agents fail at different points (alignment vs features vs prediction)
4. **Hill-climbing possible** -- small improvements in feature engineering directly improve Step 3 and Step 4 scores
5. **Open-ended** -- no single strategy dominates; agents can innovate

---

## 13. References

### Dataset

1. Peterson, S. M., et al. (2022). AJILE12: Long-term naturalistic human intracranial neural recordings and pose. *Scientific Data*, 9, 184. [https://doi.org/10.1038/s41597-022-01280-y](https://doi.org/10.1038/s41597-022-01280-y)

2. DANDI Archive -- Dandiset 000055. [https://dandiarchive.org/dandiset/000055](https://dandiarchive.org/dandiset/000055)

### Analysis Methods

3. Singh, S. H., Peterson, S. M., Rao, R. P., & Brunton, B. W. (2021). Mining naturalistic human behaviors in long-term video and neural recordings. *Journal of Neuroscience Methods*, 358, 109199. [https://doi.org/10.1016/j.jneumeth.2021.109199](https://doi.org/10.1016/j.jneumeth.2021.109199)

4. Peterson, S. M., Singh, S. H., Wang, N. X., Rao, R. P., & Brunton, B. W. (2021). Behavioral and Neural Variability of Naturalistic Arm Movements. *eNeuro*, 8(3). [https://doi.org/10.1523/ENEURO.0007-21.2021](https://doi.org/10.1523/ENEURO.0007-21.2021)

5. Wang, N. X., Farhadi, A., Rao, R. P., & Brunton, B. W. (2018). AJILE Movement Prediction: Multimodal Deep Learning for Natural Human Neural Recordings and Video. *AAAI*, 32(1). [https://arxiv.org/abs/1709.05939](https://arxiv.org/abs/1709.05939)

### Technical References

6. PyNWB Documentation -- Streaming NWB Files. [https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/streaming.html](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/streaming.html)

7. NWB Overview -- File Structure. [https://nwb-overview.readthedocs.io/en/latest/intro_to_nwb/2_file_structure.html](https://nwb-overview.readthedocs.io/en/latest/intro_to_nwb/2_file_structure.html)

### Code Repositories

8. BruntonUWBio/ajile12-nwb-data -- Official AJILE12 NWB tools. [https://github.com/BruntonUWBio/ajile12-nwb-data](https://github.com/BruntonUWBio/ajile12-nwb-data)

9. neurovium/Neuromatch-AJILE12 -- NMA educational materials. [https://github.com/neurovium/Neuromatch-AJILE12](https://github.com/neurovium/Neuromatch-AJILE12)

10. stepeter/naturalistic_arm_movements_ecog -- Analysis code for Peterson et al. 2021. [https://github.com/stepeter/naturalistic_arm_movements_ecog](https://github.com/stepeter/naturalistic_arm_movements_ecog)

---

*end of specification*
