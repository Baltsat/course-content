# E06: CalMS21 Mouse Behavior Data Curation -- pm_env Specification

> production-ready pm_env specification for E06
> dataset: CalMS21 mouse social behavior (Task 1 classic classification)
> paradigm: data curation -- student curates, system trains downstream model, score = downstream performance
> created: 2026-02-09 | author: claude opus 4.6

---

## Table of Contents

1. [Environment Overview](#1-environment-overview)
2. [Data Layout](#2-data-layout)
3. [Task Definition: 4 Progressive Steps](#3-task-definition)
4. [Step 1: Data Exploration and Quality Assessment](#4-step-1)
5. [Step 2: Feature Engineering](#5-step-2)
6. [Step 3: Handle Class Imbalance](#6-step-3)
7. [Step 4: Full Pipeline Prediction](#7-step-4)
8. [tasks.py](#8-taskspy)
9. [setup_data.py](#9-setup_datapy)
10. [env_requirements.txt](#10-env_requirementstxt)
11. [pre_hook Implementation](#11-pre_hook)
12. [Expected Score Distributions](#12-expected-score-distributions)
13. [Anti-Gaming Summary](#13-anti-gaming-summary)
14. [References](#14-references)

---

## 1. Environment Overview

### 1.1 What This Environment Tests

An AI agent receives pre-downloaded CalMS21 mouse behavior data -- pose keypoints from two interacting mice with behavior annotations. The agent must:

1. Explore and understand a real neuroscience dataset (class distribution, data schema, temporal structure)
2. Engineer behavior-relevant features from raw pose keypoints (distances, velocities, orientations, temporal context)
3. Handle extreme class imbalance (~63% "other" vs ~2.8% "attack") through data curation strategies
4. Produce predictions that generalize to hidden test sequences

This is a **data curation** environment: the student does NOT train the downstream model. The student curates features, handles imbalance, and cleans data. The SYSTEM trains a fixed downstream classifier (logistic regression) and evaluates on hidden data. Score = downstream model performance.

### 1.2 Why CalMS21

- **Real data** (not synthetic) -- published NeurIPS 2021 dataset from Caltech
- **Genuinely messy** -- extreme class imbalance, noisy frame-level annotations, pose tracking failures during mouse overlap, temporal dependencies
- **Data curation is the bottleneck** -- feature engineering and imbalance handling matter more than model choice
- **Continuous metrics** -- macro F1 score is naturally continuous
- **Open-ended** -- many valid feature engineering and curation strategies
- **Well-documented failure modes** -- naive approaches (predict "other" always) get macro F1 = 0 on rare classes

### 1.3 The Core Pattern: Data Curation

```
Student curates data (features, balancing, cleaning)
    |
    v
System trains fixed downstream model (logistic regression)
    |
    v
Score = downstream model performance on hidden test sequences
```

The student never sees the test data. The student never knows the exact scoring formula. The student is evaluated on how well their curated data enables a simple model to perform.

### 1.4 Hardware and Constraints

| Constraint | Value |
|------------|-------|
| Compute | CPU only (no GPU/TPU) |
| Container memory | 8 GB |
| Container disk | 10 GB |
| Network | DISABLED (all data pre-downloaded) |
| Time limit | 4 hours total |
| Max tool call duration | 10 minutes |
| Grading step timeout | 4 hours |

### 1.5 Container Image

```
base_image: python:3.11-slim
```

All Python dependencies are pre-installed. CalMS21 data is pre-downloaded into `env_data/`. The student never sees `scoring_data/`.

---

## 2. Data Layout

### 2.1 What Gets Pre-Downloaded

We download CalMS21 Task 1 data from Caltech Data (https://data.caltech.edu/records/s0vdx-0k302). The raw data is a ZIP containing `.npy` files with pose keypoints and behavior annotations organized by sequence.

- **Training set**: 70 annotated sequences (~507,738 total frames) -- all go to `env_data/train/`
- **Test set**: 19 sequences (~262,107 frames) -- split:
  - 12 sequences -> `env_data/val/` (student uses for development, labels visible)
  - 7 sequences -> `scoring_data/test/` (HIDDEN, for evaluation)

The 19 test sequences are split at the sequence level. The student sees 12 of them with labels. The remaining 7 are hidden.

### 2.2 Directory Structure

```
/workspace/
├── env_data/                              # VISIBLE to student
│   ├── train/
│   │   └── train_data.npy                 # 70 sequences, allow_pickle=True
│   ├── val/
│   │   └── val_data.npy                   # 12 sequences with labels
│   └── README.txt                         # brief data description
│
├── scoring_data/                          # HIDDEN from student
│   ├── test/
│   │   └── test_data.npy                  # 7 hidden test sequences
│   ├── reference/
│   │   ├── step1_answers.json             # ground truth for Step 1
│   │   ├── step2_test_features.npz        # hidden test features (reference-computed)
│   │   ├── hidden_sequence_keys.json      # which sequence keys are hidden
│   │   └── val_hidden_indices.npz         # frame indices of hidden seqs in combined val
│   └── scoring_scripts/
│       ├── judge_step1.py
│       ├── judge_step2.py
│       ├── judge_step3.py
│       └── judge_step4.py
│
└── submission/                            # student writes here
    └── (empty at start)
```

### 2.3 Data Schema

Each `.npy` file loads as a dictionary (via `np.load(path, allow_pickle=True).item()`):

```python
{
    "annotator-id_0": {
        "sequence_0": {
            "keypoints": np.ndarray,   # shape: (n_frames, 2, 2, 7)
                                        # axes: [frame, mouse_id, xy_coord, keypoint]
            "annotations": np.ndarray,  # shape: (n_frames,)
                                        # integer class labels: 0-3
        },
        "sequence_1": { ... },
        ...
    },
    "metadata": {
        "vocab": {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
    }
}
```

**Notes:**
- Mouse 0 = resident (black mouse, behavior subject)
- Mouse 1 = intruder (white mouse)
- 7 keypoints per mouse: nose (0), left_ear (1), right_ear (2), neck (3), left_hip (4), right_hip (5), tail_base (6)
- Image dimensions: 1024 x 570 pixels
- Frame rate: 30 Hz
- Keypoint coordinates are in pixel space (x, y)
- Annotations: 0=attack, 1=investigation, 2=mount, 3=other

### 2.4 Class Distribution (Training Set)

| Behavior | Class ID | Percentage | Approx Frames |
|----------|----------|------------|---------------|
| attack | 0 | 2.76% | ~14,000 |
| investigation | 1 | 28.88% | ~147,000 |
| mount | 2 | 5.64% | ~29,000 |
| other | 3 | 62.72% | ~318,000 |

**Total training frames:** ~507,738 across 70 sequences
**Sequence lengths:** ~900-12,000 frames each (variable)

### 2.5 Data Volume Estimates

| Item | Size |
|------|------|
| Training data (70 sequences, keypoints + annotations) | ~300 MB |
| Validation data (12 sequences) | ~100 MB |
| Hidden test data (7 sequences) | ~60 MB |
| Reference scoring artifacts | ~20 MB |
| **Total container disk used** | **~500 MB** |

---

## 3. Task Definition

### 3.1 Single Task, Four Progressive Steps

The environment contains ONE task (`calms21_data_curation`) with 4 sequential Steps. Each Step builds on the previous but is independently scorable.

| Step | Name | Judge Type | Reward Range |
|------|------|------------|--------------|
| 1 | Data Exploration & Quality Assessment | ExecutableJudge | 0.0 -- 1.0 (fraction correct) |
| 2 | Feature Engineering | ExecutableJudge (continuous) | 0.0 -- 1.0 |
| 3 | Handle Class Imbalance | ExecutableJudge (continuous) | 0.0 -- 1.0 |
| 4 | Full Pipeline Prediction | ExecutableJudge (continuous) | 0.0 -- 1.0 |

### 3.2 Step Dependencies

- Step 2 uses training data explored in Step 1
- Step 3 can use the student's Step 2 features OR load data fresh
- Step 4 uses the student's full pipeline from Steps 2-3 to make predictions

Steps are independent in the sense that a student who fails Step 2 can still attempt Step 3 by loading raw data. But good features in Step 2 help Step 3 and Step 4.

---

## 4. Step 1: Data Exploration and Quality Assessment

### 4.1 Task Instructions (shown to student)

```
STEP 1: Data Exploration and Quality Assessment
=================================================

You have access to mouse behavior data in /workspace/env_data/.
This is the CalMS21 dataset -- pose tracking of two interacting mice
with behavior annotations.

Explore the data and report your findings.

Write your answers to /workspace/submission/step1_answers.json
in the following format:

{
    "num_train_sequences": <int>,
    "num_val_sequences": <int>,
    "num_keypoints_per_mouse": <int>,
    "num_classes": <int>,
    "class_names": "<comma-separated class names in alphabetical order>",
    "pct_attack": <float, percentage of training frames labeled attack, rounded to 1 decimal>,
    "pct_other": <float, percentage of training frames labeled other, rounded to 1 decimal>,
    "most_common_transition": "<string, e.g. 'attack->other'>"
}

Notes:
- Load the .npy files with np.load(path, allow_pickle=True).item()
- The data is organized by annotator, then by sequence.
- Use annotator "annotator-id_0" for all your analysis.
- For "most_common_transition": count all frame-to-frame label changes
  where the label actually changes (exclude self-transitions like
  other->other). Report the most frequent non-self transition as
  "label_a->label_b" using class names (not numbers).
- For percentages: count all training frames, compute the fraction
  belonging to each class, multiply by 100, round to 1 decimal place.

Hints:
- Start by loading one file and examining its structure.
- The transition question requires iterating over consecutive frames.
- There are multiple annotators but use annotator-id_0.

Time limit: 30 minutes recommended (of the 4 hour total).
```

### 4.2 Ground Truth (scoring_data/reference/step1_answers.json)

Generated by `setup_data.py` from the actual downloaded data. Example:

```json
{
    "num_train_sequences": 70,
    "num_val_sequences": 12,
    "num_keypoints_per_mouse": 7,
    "num_classes": 4,
    "class_names": "attack,investigation,mount,other",
    "pct_attack": 2.8,
    "pct_other": 62.7,
    "most_common_transition": "other->investigation"
}
```

NOTE: Exact values depend on the actual data. `setup_data.py` computes these from the downloaded dataset.

### 4.3 Judge: ExecutableJudge (fraction correct)

```python
# scoring_data/scoring_scripts/judge_step1.py
"""
Step 1 Judge: Fraction of correct answers.
Score = num_correct / num_total (8 checks).
"""

import json
import sys


def judge_step1(submission_path, reference_path, output_path):
    try:
        with open(submission_path, "r") as f:
            submitted = json.load(f)
        with open(reference_path, "r") as f:
            reference = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        json.dump({"score": 0.0, "metadata": {"error": str(e)}}, open(output_path, "w"))
        return

    total_checks = 0
    correct_checks = 0
    details = {}

    # Check 1: num_train_sequences
    total_checks += 1
    ref_val = reference["num_train_sequences"]
    sub_val = submitted.get("num_train_sequences")
    match = sub_val is not None and int(sub_val) == ref_val
    if match:
        correct_checks += 1
    details["num_train_sequences"] = {"correct": match, "expected": ref_val, "got": sub_val}

    # Check 2: num_val_sequences
    total_checks += 1
    ref_val = reference["num_val_sequences"]
    sub_val = submitted.get("num_val_sequences")
    match = sub_val is not None and int(sub_val) == ref_val
    if match:
        correct_checks += 1
    details["num_val_sequences"] = {"correct": match, "expected": ref_val, "got": sub_val}

    # Check 3: num_keypoints_per_mouse
    total_checks += 1
    ref_val = reference["num_keypoints_per_mouse"]
    sub_val = submitted.get("num_keypoints_per_mouse")
    match = sub_val is not None and int(sub_val) == ref_val
    if match:
        correct_checks += 1
    details["num_keypoints_per_mouse"] = {"correct": match, "expected": ref_val, "got": sub_val}

    # Check 4: num_classes
    total_checks += 1
    ref_val = reference["num_classes"]
    sub_val = submitted.get("num_classes")
    match = sub_val is not None and int(sub_val) == ref_val
    if match:
        correct_checks += 1
    details["num_classes"] = {"correct": match, "expected": ref_val, "got": sub_val}

    # Check 5: class_names (alphabetical, comma-separated)
    total_checks += 1
    ref_names = reference["class_names"].strip().lower()
    sub_names = str(submitted.get("class_names", "")).strip().lower()
    sub_names_normalized = ",".join(sorted([s.strip() for s in sub_names.split(",") if s.strip()]))
    ref_names_normalized = ",".join(sorted([s.strip() for s in ref_names.split(",") if s.strip()]))
    match = sub_names_normalized == ref_names_normalized
    if match:
        correct_checks += 1
    details["class_names"] = {"correct": match, "expected": ref_names, "got": sub_names}

    # Check 6: pct_attack (within 0.5 tolerance)
    total_checks += 1
    ref_val = reference["pct_attack"]
    sub_val = submitted.get("pct_attack")
    match = False
    if sub_val is not None:
        try:
            match = abs(float(sub_val) - ref_val) <= 0.5
        except (ValueError, TypeError):
            pass
    if match:
        correct_checks += 1
    details["pct_attack"] = {"correct": match, "expected": ref_val, "got": sub_val}

    # Check 7: pct_other (within 0.5 tolerance)
    total_checks += 1
    ref_val = reference["pct_other"]
    sub_val = submitted.get("pct_other")
    match = False
    if sub_val is not None:
        try:
            match = abs(float(sub_val) - ref_val) <= 0.5
        except (ValueError, TypeError):
            pass
    if match:
        correct_checks += 1
    details["pct_other"] = {"correct": match, "expected": ref_val, "got": sub_val}

    # Check 8: most_common_transition
    total_checks += 1
    ref_trans = reference["most_common_transition"].strip().lower().replace(" ", "")
    sub_trans = str(submitted.get("most_common_transition", "")).strip().lower().replace(" ", "")
    match = sub_trans == ref_trans
    if match:
        correct_checks += 1
    details["most_common_transition"] = {"correct": match, "expected": ref_trans, "got": sub_trans}

    score = correct_checks / total_checks

    result = {
        "score": round(score, 4),
        "metadata": {
            "correct": correct_checks,
            "total": total_checks,
            "details": details,
        }
    }
    json.dump(result, open(output_path, "w"), indent=2)


if __name__ == "__main__":
    judge_step1(sys.argv[1], sys.argv[2], sys.argv[3])
```

### 4.4 What's in env_data vs scoring_data

| Location | Contents |
|----------|----------|
| env_data | `train/train_data.npy` (70 sequences), `val/val_data.npy` (12 sequences), `README.txt` |
| scoring_data | `reference/step1_answers.json` (ground truth computed from actual data) |

### 4.5 Anti-Gaming Measures

1. **Data-dependent answers** -- percentages and counts require actually loading and iterating through the data; cannot guess
2. **Transition analysis** -- requires temporal reasoning (counting frame-to-frame label changes); cannot derive from class frequencies alone
3. **Exact match on class names** -- must explore the metadata/vocab field
4. **Tolerance on percentages** -- generous (0.5%) but still requires actual counting, not made-up numbers

### 4.6 Expected Score Distribution

| Score | Interpretation | Expected % of agents |
|-------|---------------|---------------------|
| 1.0 (8/8) | All correct | 35% |
| 0.875 (7/8) | Transition question wrong | 25% |
| 0.75 (6/8) | Two fields wrong | 15% |
| 0.5-0.625 | Partial understanding | 15% |
| < 0.5 | Major confusion | 10% |

---

## 5. Step 2: Feature Engineering

### 5.1 Task Instructions (shown to student)

```
STEP 2: Feature Engineering
=============================

Engineer features from the raw keypoint data that capture
behavior-relevant information.

Using the training data in /workspace/env_data/train/ and
validation data in /workspace/env_data/val/, create feature matrices.

Save to /workspace/submission/features.npz containing:
- "X_train": numpy array [n_train_frames, n_features]
- "y_train": numpy array [n_train_frames] (labels: 0=attack, 1=investigation, 2=mount, 3=other)
- "X_val": numpy array [n_val_frames, n_features]
- "y_val": numpy array [n_val_frames] (validation labels)
- "feature_names": array of feature name strings
- "sequence_ids_train": numpy array [n_train_frames] (which sequence each frame belongs to)
- "sequence_ids_val": numpy array [n_val_frames]

IMPORTANT: Process sequences in the same order they appear in the data
file (sorted by sequence key). Concatenate all frames from all sequences
sequentially. Include ALL frames -- do not drop any.

Feature ideas (not exhaustive):
- Inter-animal distances (nose-to-nose, centroid-to-centroid, etc.)
- Per-mouse velocity and acceleration of centroid
- Body orientation (angle from tail_base to nose)
- Relative orientation between mice
- Area of bounding polygon per mouse
- Temporal features: rolling mean/std of distances over N-frame windows
- All pairwise inter-mouse keypoint distances (7x7 = 49 features)
- Relative position in egocentric coordinates

Good features should capture the DYNAMICS of behavior, not just
static pose. A single frame is often insufficient -- behaviors
unfold over time. Consider temporal context windows.

Your features will be evaluated by:
1. Feature diversity (do you have distance, velocity, orientation,
   and temporal features?)
2. Downstream classification performance (a fixed logistic regression
   trained on your features, evaluated on hidden test data)

Save as: /workspace/submission/features.npz

Time limit: 60 minutes recommended.
```

### 5.2 Judge: ExecutableJudge (Continuous)

```python
# scoring_data/scoring_scripts/judge_step2.py
"""
Step 2 Judge: Feature engineering quality.
Score = 0.6 * downstream_f1_mapped + 0.4 * diversity_score

Downstream model: LogisticRegression(C=0.1, max_iter=2000, multi_class="multinomial")
F1: macro F1 on classes 0, 1, 2 (excluding "other") on validation set.
Diversity: checks for distance, velocity, orientation, temporal features by name.
"""

import json
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


def judge_step2(submission_path, reference_path, output_path):
    try:
        data = np.load(submission_path, allow_pickle=True)
        X_train = data["X_train"]
        y_train = data["y_train"]
        feature_names = list(data.get("feature_names", []))
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": f"Load failed: {e}"}},
                  open(output_path, "w"))
        return

    # Validate basic structure
    if X_train.ndim != 2:
        json.dump({"score": 0.0, "metadata": {"error": f"X_train must be 2D, got {X_train.ndim}D"}},
                  open(output_path, "w"))
        return

    if X_train.shape[0] < 100:
        json.dump({"score": 0.0, "metadata": {"error": f"Too few samples: {X_train.shape[0]}"}},
                  open(output_path, "w"))
        return

    if X_train.shape[1] < 3:
        json.dump({"score": 0.0, "metadata": {"error": f"Too few features: {X_train.shape[1]}"}},
                  open(output_path, "w"))
        return

    if len(y_train) != X_train.shape[0]:
        json.dump({"score": 0.0, "metadata": {"error": "X_train and y_train length mismatch"}},
                  open(output_path, "w"))
        return

    # Clean NaN/inf
    X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Check feature diversity by name
    fn_lower = [n.lower() for n in feature_names]
    has_distance = any("dist" in n or "distance" in n for n in fn_lower)
    has_velocity = any("vel" in n or "speed" in n for n in fn_lower)
    has_orientation = any("angle" in n or "orient" in n or "heading" in n for n in fn_lower)
    has_temporal = any("roll" in n or "mean" in n or "std" in n or "window" in n
                       or "temporal" in n for n in fn_lower)

    diversity_score = (
        has_distance * 0.25 +
        has_velocity * 0.25 +
        has_orientation * 0.25 +
        has_temporal * 0.25
    )

    # Train downstream model and evaluate on validation set
    macro_f1 = 0.0
    train_error = None

    if "X_val" in data and "y_val" in data:
        X_val = np.nan_to_num(data["X_val"], nan=0.0, posinf=0.0, neginf=0.0)
        y_val = data["y_val"]

        if X_val.shape[1] != X_train_clean.shape[1]:
            train_error = f"Feature dim mismatch: train={X_train_clean.shape[1]}, val={X_val.shape[1]}"
        elif len(y_val) < 10:
            train_error = "Too few validation samples"
        else:
            try:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train_clean)
                X_val_s = scaler.transform(X_val)

                model = LogisticRegression(
                    C=0.1, max_iter=2000,
                    multi_class="multinomial", solver="lbfgs",
                    n_jobs=-1,
                )
                model.fit(X_tr_s, y_train)
                y_pred = model.predict(X_val_s)

                # Macro F1 excluding "other" (class 3)
                mask = y_val != 3
                if mask.sum() > 0:
                    macro_f1 = f1_score(
                        y_val[mask], y_pred[mask],
                        average="macro", labels=[0, 1, 2], zero_division=0,
                    )
            except Exception as e:
                train_error = f"Training failed: {e}"
    else:
        train_error = "Missing X_val or y_val"

    # Map F1 to score
    # 0.0 -> 0.0, 0.3 -> 0.3, 0.5 -> 0.6, 0.7 -> 0.9, 0.8+ -> 1.0
    if macro_f1 <= 0.0:
        f1_mapped = 0.0
    elif macro_f1 <= 0.3:
        f1_mapped = macro_f1
    elif macro_f1 <= 0.5:
        f1_mapped = 0.3 + (macro_f1 - 0.3) / 0.2 * 0.3
    elif macro_f1 <= 0.7:
        f1_mapped = 0.6 + (macro_f1 - 0.5) / 0.2 * 0.3
    else:
        f1_mapped = min(0.9 + (macro_f1 - 0.7) / 0.1 * 0.1, 1.0)

    final = 0.6 * f1_mapped + 0.4 * diversity_score

    result = {
        "score": round(final, 4),
        "metadata": {
            "macro_f1_excl_other": round(macro_f1, 4),
            "f1_mapped": round(f1_mapped, 4),
            "diversity_score": round(diversity_score, 4),
            "has_distance": has_distance,
            "has_velocity": has_velocity,
            "has_orientation": has_orientation,
            "has_temporal": has_temporal,
            "n_features": int(X_train.shape[1]),
            "n_train_samples": int(X_train.shape[0]),
            "n_feature_names": len(feature_names),
            "train_error": train_error,
        }
    }
    json.dump(result, open(output_path, "w"), indent=2)


if __name__ == "__main__":
    judge_step2(sys.argv[1], sys.argv[2], sys.argv[3])
```

### 5.3 What's in env_data vs scoring_data

| Location | Contents |
|----------|----------|
| env_data | `train/train_data.npy` (training sequences), `val/val_data.npy` (12 validation sequences with labels) |
| scoring_data | `scoring_scripts/judge_step2.py` (trains logistic regression on student's features, evaluates on val) |

### 5.4 Anti-Gaming Measures

1. **Feature diversity checked by name** -- the student must include distance, velocity, orientation, AND temporal features to get full diversity score (40% of total)
2. **Downstream model evaluation** -- features that don't actually help classification get low F1 (60% of total)
3. **Fixed downstream model** -- student cannot overfit model choice; logistic regression is fixed
4. **Macro F1 excludes "other"** -- predicting "other" always yields F1=0 on rare classes
5. **Validation set contains only student-visible sequences** -- but Step 4 uses hidden sequences

### 5.5 Expected Score Distribution

| Score Range | Interpretation | Expected % |
|-------------|---------------|------------|
| 0.80 -- 1.0 | All 4 feature types + high downstream F1 | 10% |
| 0.60 -- 0.79 | Good features, 3-4 types, decent F1 | 25% |
| 0.40 -- 0.59 | Some features work, missing temporal or orientation | 30% |
| 0.20 -- 0.39 | Basic features only, low F1 | 20% |
| 0.00 -- 0.19 | Broken submission or trivial features | 15% |

---

## 6. Step 3: Handle Class Imbalance

### 6.1 Task Instructions (shown to student)

```
STEP 3: Handle Class Imbalance
================================

The behavior classes are extremely imbalanced:
- "other" dominates at ~63% of frames
- "attack" is only ~3% of frames
- "mount" is only ~6%

A naive classifier will predict "other" for everything and get
high accuracy but ZERO macro F1 on the rare behaviors.

Your task: create a curated, BALANCED training dataset that
enables better classification of rare behaviors.

You may:
- Load your features from Step 2 (/workspace/submission/features.npz)
- Or load the raw data and compute features fresh
- Or any combination

Apply data curation strategies such as:
- Undersample the "other" class
- Oversample rare classes (random oversampling, SMOTE, etc.)
- Use sample weights to rebalance
- Remove ambiguous boundary frames (frames where the label
  changes within a short window -- these often have noisy labels)
- Use temporal context to improve label quality (majority vote
  within a sliding window)
- Combine undersampling + oversampling
- Any other strategy you believe will help

Save to /workspace/submission/curated_train.npz containing:
- "X": numpy array [n_samples, n_features] -- curated feature matrix
- "y": numpy array [n_samples] -- curated labels (0, 1, 2, 3)
- "sample_weights": numpy array [n_samples] -- per-sample weights
  (use ones if not applying weighting)
- "curation_strategy": string describing what you did

Your curated data will be evaluated by training a downstream
classifier and measuring macro F1 on hidden test sequences
(excluding "other" class from F1 computation).

Good curation should:
1. Reduce the dominance of "other"
2. Preserve enough "other" examples that the model still learns
   the class (don't remove it entirely)
3. Increase representation of rare classes (especially "attack")
4. Remove noisy labels near behavior transitions
5. Keep the total dataset size manageable (< 1M samples)

Save as: /workspace/submission/curated_train.npz

Time limit: 45 minutes recommended.
```

### 6.2 Judge: ExecutableJudge (Continuous)

```python
# scoring_data/scoring_scripts/judge_step3.py
"""
Step 3 Judge: Curated data quality.
Trains logistic regression on curated data, evaluates on hidden test sequences.
Score = macro F1 on rare behaviors (0=attack, 1=investigation, 2=mount)
mapped to [0, 1] range.

Anti-gaming:
- Hidden test sequences (student never sees them)
- Macro F1 excludes "other" (predicting "other" always = score 0)
- Degenerate submissions caught (all same class, too few samples, etc.)
"""

import json
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


def judge_step3(submission_path, hidden_test_path, output_path):
    # Load student's curated data
    try:
        data = np.load(submission_path, allow_pickle=True)
        X_curated = data["X"]
        y_curated = data["y"]
        sample_weights = data.get("sample_weights", None)
        curation_strategy = str(data.get("curation_strategy", "not provided"))
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": f"Load failed: {e}"}},
                  open(output_path, "w"))
        return

    # Load hidden test data (features pre-computed by setup_data.py using reference pipeline)
    try:
        test = np.load(hidden_test_path, allow_pickle=True)
        X_test = test["X"]
        y_test = test["y"]
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": f"Test data load failed: {e}"}},
                  open(output_path, "w"))
        return

    # Validate curated data
    if X_curated.ndim != 2:
        json.dump({"score": 0.0, "metadata": {"error": f"X must be 2D, got {X_curated.ndim}D"}},
                  open(output_path, "w"))
        return

    if X_curated.shape[0] < 100:
        json.dump({"score": 0.0, "metadata": {"error": f"Too few samples: {X_curated.shape[0]}"}},
                  open(output_path, "w"))
        return

    if X_curated.shape[1] != X_test.shape[1]:
        # Feature dimension mismatch -- try to use a common subset
        # This happens when student uses different features than reference
        # In this case, score based on validation data only
        json.dump({"score": 0.0, "metadata": {
            "error": f"Feature dim mismatch: curated={X_curated.shape[1]}, test={X_test.shape[1]}. "
                     "Your features must match the feature pipeline used in Step 2."
        }}, open(output_path, "w"))
        return

    # Check for degenerate submissions
    unique_labels = np.unique(y_curated)
    if len(unique_labels) <= 1:
        json.dump({"score": 0.0, "metadata": {"error": "All same class -- degenerate"}},
                  open(output_path, "w"))
        return

    if X_curated.shape[0] > 2_000_000:
        json.dump({"score": 0.0, "metadata": {"error": f"Too many samples: {X_curated.shape[0]}. Max 2M."}},
                  open(output_path, "w"))
        return

    # Clean
    X_curated = np.nan_to_num(X_curated, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Train downstream model
    try:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_curated)
        X_test_s = scaler.transform(X_test)

        sw = None
        if sample_weights is not None:
            sw = np.nan_to_num(np.asarray(sample_weights, dtype=float), nan=1.0)
            sw = np.clip(sw, 0.01, 100.0)
            if len(sw) != len(y_curated):
                sw = None

        model = LogisticRegression(
            C=0.1, max_iter=2000,
            multi_class="multinomial", solver="lbfgs",
            n_jobs=-1,
        )
        model.fit(X_train_s, y_curated, sample_weight=sw)

        y_pred = model.predict(X_test_s)
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": f"Training failed: {e}"}},
                  open(output_path, "w"))
        return

    # Compute macro F1 excluding "other" (class 3)
    mask = y_test != 3
    if mask.sum() < 10:
        json.dump({"score": 0.0, "metadata": {"error": "Too few non-other test samples"}},
                  open(output_path, "w"))
        return

    macro_f1 = f1_score(
        y_test[mask], y_pred[mask],
        average="macro", labels=[0, 1, 2], zero_division=0,
    )

    # Per-class F1
    per_class = {}
    for cls, name in enumerate(["attack", "investigation", "mount"]):
        cls_mask = y_test == cls
        if cls_mask.sum() > 0:
            cls_pred_mask = y_pred[cls_mask]
            per_class[name] = round(float(
                f1_score(y_test[cls_mask] == cls, cls_pred_mask == cls,
                         average="binary", zero_division=0)
            ), 4)
        else:
            per_class[name] = 0.0

    # Map F1 to score
    # 0.0 -> 0.0, 0.3 -> 0.2, 0.5 -> 0.5, 0.65 -> 0.7, 0.79 -> 0.9, 0.86+ -> 1.0
    if macro_f1 <= 0.0:
        score = 0.0
    elif macro_f1 <= 0.3:
        score = macro_f1 * (0.2 / 0.3)
    elif macro_f1 <= 0.5:
        score = 0.2 + (macro_f1 - 0.3) / 0.2 * 0.3
    elif macro_f1 <= 0.65:
        score = 0.5 + (macro_f1 - 0.5) / 0.15 * 0.2
    elif macro_f1 <= 0.79:
        score = 0.7 + (macro_f1 - 0.65) / 0.14 * 0.2
    else:
        score = min(0.9 + (macro_f1 - 0.79) / 0.07 * 0.1, 1.0)

    # Class balance quality bonus (up to 0.05)
    class_counts = {int(c): int((y_curated == c).sum()) for c in range(4)}
    total = sum(class_counts.values())
    if total > 0:
        fracs = [class_counts.get(c, 0) / total for c in range(4)]
        # ideal: each class ~25%. measure entropy relative to uniform
        from scipy.stats import entropy
        uniform = [0.25] * 4
        kl = entropy(fracs, uniform) if all(f > 0 for f in fracs) else 10.0
        balance_bonus = max(0, 0.05 * (1 - kl / 2.0))
        score = min(score + balance_bonus, 1.0)

    result = {
        "score": round(score, 4),
        "metadata": {
            "macro_f1_rare": round(macro_f1, 4),
            "per_class_f1": per_class,
            "n_curated_samples": int(X_curated.shape[0]),
            "n_features": int(X_curated.shape[1]),
            "class_distribution": class_counts,
            "curation_strategy": curation_strategy[:500],
            "used_sample_weights": sample_weights is not None,
        }
    }
    json.dump(result, open(output_path, "w"), indent=2)


if __name__ == "__main__":
    judge_step3(sys.argv[1], sys.argv[2], sys.argv[3])
```

### 6.3 What's in env_data vs scoring_data

| Location | Contents |
|----------|----------|
| env_data | Training and validation data (student-visible sequences with labels) |
| scoring_data | `test/test_data.npy` (7 hidden sequences), `reference/step2_test_features.npz` (reference features for hidden sequences), scoring scripts |

### 6.4 Anti-Gaming Measures

1. **Hidden test sequences** -- student cannot see the test data, cannot overfit to it
2. **Macro F1 excludes "other"** -- predicting "other" for everything yields F1 = 0 on classes 0, 1, 2
3. **Degenerate submission detection** -- all-same-class, too few samples, too many samples
4. **Fixed downstream model** -- student cannot choose a model that exploits their data
5. **Feature dimension must match** -- prevents student from submitting features incompatible with the scoring pipeline (must use same features as Step 2)

### 6.5 Expected Score Distribution

| Score Range | Interpretation | Expected % |
|-------------|---------------|------------|
| 0.80 -- 1.0 | Effective balancing + good features, near SOTA | 5% |
| 0.60 -- 0.79 | Good balancing strategy, solid features | 20% |
| 0.40 -- 0.59 | Basic balancing (undersample other), moderate F1 | 30% |
| 0.20 -- 0.39 | Attempted balancing but weak features or strategy | 25% |
| 0.00 -- 0.19 | No balancing (raw data) or broken submission | 20% |

---

## 7. Step 4: Full Pipeline Prediction

### 7.1 Task Instructions (shown to student)

```
STEP 4: Full Pipeline Prediction
==================================

Final challenge: produce the best possible frame-level predictions
for all sequences in the validation set.

Apply your full pipeline: feature engineering (Step 2) +
class imbalance handling (Step 3) + any additional improvements.

Save to /workspace/submission/predictions.npz containing:
- "y_pred": numpy array [n_val_frames] -- predicted class (int, 0-3)
  for each frame in /workspace/env_data/val/val_data.npy
- "y_prob": numpy array [n_val_frames, 4] -- predicted probabilities
  per class (must sum to ~1.0 per frame)
- "pipeline_description": string describing your full approach

IMPORTANT:
- Process validation sequences in the same order they appear in
  the data file (sorted by sequence key).
- Concatenate all frames sequentially, same as Step 2.
- Include ALL frames -- do not drop any.
- n_val_frames must match the total number of frames in val_data.npy.

Your predictions will be scored on a SUBSET of the validation
sequences that you cannot identify. These hidden sequences test
whether your pipeline generalizes beyond the sequences you've
been analyzing.

Scoring: macro F1 on the 3 rare behaviors (attack, investigation,
mount) computed ONLY on the hidden subset of validation sequences.

Tips:
- You can try multiple classifiers (Random Forest, SVM, etc.)
  and select the best based on your own cross-validation.
- Temporal smoothing of predictions can help (median filter, HMM).
- Ensemble methods often outperform single classifiers.
- Feature stacking from different window sizes adds information.
- The hidden subset may have different class proportions than
  the sequences you can see.

Save as: /workspace/submission/predictions.npz

Time limit: 90 minutes recommended.
```

### 7.2 Judge: ExecutableJudge (Continuous)

```python
# scoring_data/scoring_scripts/judge_step4.py
"""
Step 4 Judge: Full pipeline prediction quality.

The validation set contains a MIX of student-visible sequences
and hidden test sequences. The student produces predictions for
ALL validation sequences. We score ONLY on the hidden subset.

Score = macro F1 on rare behaviors (attack, investigation, mount)
on hidden sequence frames, mapped to [0, 1].

Anti-gaming:
- Student cannot identify which sequences are hidden
- Constant predictions yield score 0
- Macro F1 excludes "other"
"""

import json
import sys
import numpy as np
from sklearn.metrics import f1_score


def judge_step4(submission_path, hidden_labels_path, hidden_indices_path, output_path):
    # Load student predictions
    try:
        data = np.load(submission_path, allow_pickle=True)
        y_pred = data["y_pred"].astype(int)
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": f"Load failed: {e}"}},
                  open(output_path, "w"))
        return

    # Load hidden ground truth
    try:
        hidden = np.load(hidden_labels_path, allow_pickle=True)
        y_true_hidden = hidden["y"].astype(int)
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": f"Hidden labels load failed: {e}"}},
                  open(output_path, "w"))
        return

    # Load hidden frame indices (which frames in the full val set belong to hidden sequences)
    try:
        idx_data = np.load(hidden_indices_path)
        hidden_indices = idx_data["indices"]
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": f"Hidden indices load failed: {e}"}},
                  open(output_path, "w"))
        return

    # Validate prediction length
    max_needed = int(hidden_indices.max()) + 1 if len(hidden_indices) > 0 else 0
    if len(y_pred) < max_needed:
        json.dump({"score": 0.0, "metadata": {
            "error": f"Predictions too short: {len(y_pred)} < {max_needed} needed"
        }}, open(output_path, "w"))
        return

    # Extract predictions for hidden sequences
    y_pred_hidden = y_pred[hidden_indices]

    # Sanity check: lengths must match
    if len(y_pred_hidden) != len(y_true_hidden):
        json.dump({"score": 0.0, "metadata": {
            "error": f"Length mismatch: pred={len(y_pred_hidden)}, true={len(y_true_hidden)}"
        }}, open(output_path, "w"))
        return

    # Anti-gaming: constant predictions
    if len(set(y_pred_hidden)) <= 1:
        json.dump({"score": 0.0, "metadata": {"error": "Constant predictions"}},
                  open(output_path, "w"))
        return

    # Compute macro F1 on rare behaviors (exclude "other" = class 3)
    mask = y_true_hidden != 3
    if mask.sum() < 10:
        json.dump({"score": 0.0, "metadata": {"error": "Too few non-other test frames"}},
                  open(output_path, "w"))
        return

    macro_f1 = f1_score(
        y_true_hidden[mask], y_pred_hidden[mask],
        average="macro", labels=[0, 1, 2], zero_division=0,
    )

    # Per-class F1
    per_class = {}
    for cls, name in enumerate(["attack", "investigation", "mount"]):
        cls_mask = y_true_hidden == cls
        if cls_mask.sum() > 0:
            per_class[name] = round(float(
                f1_score(y_true_hidden[cls_mask] == cls, y_pred_hidden[cls_mask] == cls,
                         average="binary", zero_division=0)
            ), 4)
        else:
            per_class[name] = 0.0

    # Also compute overall accuracy and F1 including "other"
    overall_f1 = f1_score(y_true_hidden, y_pred_hidden, average="macro", zero_division=0)

    # Map F1 to score
    # 0.0 -> 0.0, 0.3 -> 0.2, 0.5 -> 0.5, 0.65 -> 0.7, 0.79 -> 0.9, 0.86+ -> 1.0
    if macro_f1 <= 0.0:
        score = 0.0
    elif macro_f1 <= 0.5:
        score = macro_f1
    elif macro_f1 <= 0.65:
        score = 0.5 + (macro_f1 - 0.5) / 0.15 * 0.2
    elif macro_f1 <= 0.79:
        score = 0.7 + (macro_f1 - 0.65) / 0.14 * 0.2
    else:
        score = min(0.9 + (macro_f1 - 0.79) / 0.07 * 0.1, 1.0)

    # Class distribution in predictions (for hidden subset)
    pred_dist = {str(c): int((y_pred_hidden == c).sum()) for c in range(4)}
    true_dist = {str(c): int((y_true_hidden == c).sum()) for c in range(4)}

    result = {
        "score": round(score, 4),
        "metadata": {
            "macro_f1_rare": round(macro_f1, 4),
            "overall_f1": round(overall_f1, 4),
            "per_class_f1": per_class,
            "n_hidden_frames": int(len(hidden_indices)),
            "n_total_predictions": int(len(y_pred)),
            "pred_class_distribution": pred_dist,
            "true_class_distribution": true_dist,
        }
    }
    json.dump(result, open(output_path, "w"), indent=2)


if __name__ == "__main__":
    judge_step4(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
```

### 7.3 The Hidden Sequence Trick

The validation set (`env_data/val/val_data.npy`) contains 12 sequences. Of these 12:
- 5 are genuine validation sequences (student has seen labels throughout)
- 7 are hidden test sequences -- the student CAN see the keypoints (they're in val_data.npy) but the labels for these sequences are replaced with dummy values (all zeros)

The student is told that "some validation sequences have been replaced." The student must produce predictions for ALL 12 sequences. Scoring uses only the 7 hidden ones.

Wait -- this complicates things because the student would notice that some sequences have all-zero labels. A better approach:

**Revised design**: The validation set contains 12 sequences, ALL with correct labels. Separately, the 7 hidden test sequences are stored in `scoring_data/test/` and are NOT in `env_data/val/`. The "trick" in Step 4 is that we:

1. During `setup_data.py`, we compute reference features for the 7 hidden test sequences
2. During Step 4 judging, we use the student's feature extraction code (from Step 2) to compute features for hidden sequences, OR we use reference features if the student's feature pipeline matches

**Simpler revised design**: Actually, let's keep it simple and just evaluate on the hidden test sequences directly. The student submits a prediction script (like E01) that can process raw keypoint data and output predictions.

**Final design (clearest)**:

The student produces predictions for the validation set. Additionally, the 7 hidden test sequences are stored separately in scoring_data. The judge:
1. Loads the student's `features.npz` (from Step 2) for the feature extraction pattern
2. Loads the student's `curated_train.npz` (from Step 3) for the training data
3. Trains a logistic regression on the curated data
4. Applies it to reference-computed features for the hidden test sequences
5. Scores on the hidden test sequences

This means the judge for Step 4 effectively re-does the Step 3 scoring but on hidden data, plus evaluates the student's actual predictions on visible data as a secondary check.

**Simplest correct design**: The student writes a prediction script that takes a data path and returns predictions. This is the most robust approach.

### 7.4 Revised Step 4 Instructions

```
STEP 4: Full Pipeline Prediction
==================================

Final challenge: write a complete prediction pipeline.

Create a Python script at /workspace/submission/step4_predict.py
that implements:

    import numpy as np

    def predict(data_path: str) -> np.ndarray:
        """
        Given a path to a .npy file containing CalMS21 sequences,
        return predicted class labels (int, 0-3) for every frame.

        The data file has the same structure as the training data:
        data = np.load(data_path, allow_pickle=True).item()
        sequences = data["annotator-id_0"]

        Process sequences in sorted key order. Concatenate all frames.
        Return one label per frame as a 1D numpy array.
        """

You MAY load pre-trained models or pre-computed data from
/workspace/submission/. You MAY use any libraries available
in the container.

Your script will be called with a HIDDEN test file containing
sequences you have never seen. Your pipeline must generalize.

CONSTRAINTS:
- predict() must complete in under 10 minutes.
- Must return int labels (0, 1, 2, or 3).
- Must return exactly one label per frame.
- Process sequences in sorted key order.

TIPS:
- Train your model in this step and save it, or train inline.
- Use your features from Step 2 and balancing from Step 3.
- Random Forest or Logistic Regression are good choices.
- Temporal smoothing can help (median filter over predictions).
- Make sure your pipeline handles varying sequence lengths.

Save as: /workspace/submission/step4_predict.py

Time limit: 90 minutes recommended.
```

### 7.5 Revised Judge for Step 4

```python
# scoring_data/scoring_scripts/judge_step4.py
"""
Step 4 Judge: End-to-end prediction quality.

Calls the student's predict() function with the hidden test data.
Computes macro F1 on rare behaviors.
"""

import json
import sys
import importlib.util
import traceback
import time
import numpy as np
from sklearn.metrics import f1_score


TIMEOUT_SECONDS = 600  # 10 minutes


def judge_step4(predict_script_path, test_data_path, test_labels_path, output_path):
    # Load ground truth
    try:
        test_labels = np.load(test_labels_path, allow_pickle=True)
        y_true = test_labels["y"]
        n_frames = len(y_true)
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": f"Labels load failed: {e}"}},
                  open(output_path, "w"))
        return

    # Load student's predict function
    try:
        spec = importlib.util.spec_from_file_location("student_predict", predict_script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predict_fn = module.predict
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": f"Script load failed: {e}\n{traceback.format_exc()}"}},
                  open(output_path, "w"))
        return

    # Run prediction
    try:
        start = time.time()
        y_pred = predict_fn(test_data_path)
        elapsed = time.time() - start

        if elapsed > TIMEOUT_SECONDS:
            json.dump({"score": 0.0, "metadata": {"error": f"Timeout: {elapsed:.1f}s > {TIMEOUT_SECONDS}s"}},
                      open(output_path, "w"))
            return
    except Exception as e:
        json.dump({"score": 0.0, "metadata": {"error": f"predict() failed: {e}\n{traceback.format_exc()}"}},
                  open(output_path, "w"))
        return

    # Validate
    y_pred = np.asarray(y_pred, dtype=int)

    if y_pred.ndim != 1:
        json.dump({"score": 0.0, "metadata": {"error": f"Must be 1D, got shape {y_pred.shape}"}},
                  open(output_path, "w"))
        return

    if len(y_pred) != n_frames:
        # Allow slight mismatch, truncate to shorter
        min_len = min(len(y_pred), n_frames)
        if min_len < n_frames * 0.9:
            json.dump({"score": 0.0, "metadata": {
                "error": f"Length mismatch: pred={len(y_pred)}, expected={n_frames}"
            }}, open(output_path, "w"))
            return
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

    # Anti-gaming: constant predictions
    if len(set(y_pred)) <= 1:
        json.dump({"score": 0.0, "metadata": {"error": "Constant predictions"}},
                  open(output_path, "w"))
        return

    # Ensure valid class labels
    valid_mask = np.isin(y_pred, [0, 1, 2, 3])
    if valid_mask.sum() < len(y_pred) * 0.5:
        json.dump({"score": 0.0, "metadata": {"error": "Too many invalid class labels"}},
                  open(output_path, "w"))
        return
    # Replace invalid predictions with class 3 ("other")
    y_pred[~valid_mask] = 3

    # Compute macro F1 on rare behaviors
    mask = y_true != 3
    if mask.sum() < 10:
        json.dump({"score": 0.0, "metadata": {"error": "Too few non-other test frames"}},
                  open(output_path, "w"))
        return

    macro_f1 = f1_score(
        y_true[mask], y_pred[mask],
        average="macro", labels=[0, 1, 2], zero_division=0,
    )

    # Per-class F1
    per_class = {}
    for cls, name in enumerate(["attack", "investigation", "mount"]):
        cls_true = (y_true == cls).astype(int)
        cls_pred = (y_pred == cls).astype(int)
        if cls_true.sum() > 0:
            per_class[name] = round(float(
                f1_score(cls_true, cls_pred, average="binary", zero_division=0)
            ), 4)
        else:
            per_class[name] = 0.0

    # Overall metrics
    overall_accuracy = float((y_pred == y_true).mean())
    overall_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Map F1 to score (same as Step 3)
    if macro_f1 <= 0.0:
        score = 0.0
    elif macro_f1 <= 0.5:
        score = macro_f1
    elif macro_f1 <= 0.65:
        score = 0.5 + (macro_f1 - 0.5) / 0.15 * 0.2
    elif macro_f1 <= 0.79:
        score = 0.7 + (macro_f1 - 0.65) / 0.14 * 0.2
    else:
        score = min(0.9 + (macro_f1 - 0.79) / 0.07 * 0.1, 1.0)

    pred_dist = {str(c): int((y_pred == c).sum()) for c in range(4)}
    true_dist = {str(c): int((y_true == c).sum()) for c in range(4)}

    result = {
        "score": round(score, 4),
        "metadata": {
            "macro_f1_rare": round(macro_f1, 4),
            "overall_f1": round(overall_f1, 4),
            "overall_accuracy": round(overall_accuracy, 4),
            "per_class_f1": per_class,
            "pred_class_distribution": pred_dist,
            "true_class_distribution": true_dist,
            "n_frames": int(len(y_true)),
            "elapsed_seconds": round(elapsed, 1),
        }
    }
    json.dump(result, open(output_path, "w"), indent=2)


if __name__ == "__main__":
    judge_step4(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
```

### 7.6 What's in env_data vs scoring_data

| Location | Contents |
|----------|----------|
| env_data | Training data (70 sequences), validation data (12 sequences with labels) |
| scoring_data | Hidden test data (7 sequences, `test/test_data.npy`), hidden labels (`reference/test_labels.npz`), scoring scripts |

### 7.7 Anti-Gaming Measures

1. **Hidden test sequences** -- student's predict() is called on data they have NEVER seen
2. **Different animal pairs** -- test sequences come from different mice than training (per official CalMS21 split)
3. **Constant prediction detection** -- single predicted class yields score 0
4. **Macro F1 excludes "other"** -- predicting "other" always = score 0 on rare behaviors
5. **10-minute timeout** -- prevents exhaustive search or brute force
6. **Invalid label detection** -- predictions outside {0,1,2,3} are replaced with 3 ("other")

### 7.8 Expected Score Distribution

| macro F1 | pm_env Score | Interpretation | Expected % |
|----------|-------------|---------------|------------|
| 0.79+ | 0.90 -- 1.0 | Near SOTA, excellent pipeline | 5% |
| 0.65 -- 0.78 | 0.70 -- 0.89 | Strong features + good balancing | 15% |
| 0.50 -- 0.64 | 0.50 -- 0.69 | Decent features, some balancing | 25% |
| 0.30 -- 0.49 | 0.20 -- 0.49 | Basic features, weak balancing | 30% |
| 0.10 -- 0.29 | 0.05 -- 0.19 | Poor generalization | 15% |
| < 0.10 | 0.00 -- 0.04 | Broken pipeline or no balancing | 10% |

---

## 8. tasks.py

```python
# tasks.py
"""
E06: CalMS21 Mouse Behavior Data Curation
pm_env task definition with 4 progressive steps.

Paradigm: data curation
- Student curates features, handles imbalance, cleans data
- System trains a fixed downstream model
- Score = downstream model performance on hidden test data
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
REFERENCE = SCORING_DATA / "reference"


def get_tasks():
    return [
        {
            "id": "calms21_data_curation",
            "name": "CalMS21 Mouse Behavior Data Curation",
            "description": (
                "Curate mouse behavior pose data for classification. "
                "Four steps: explore data, engineer features, handle class "
                "imbalance, and build a prediction pipeline. Your curated "
                "data is evaluated by training a downstream classifier on "
                "hidden test sequences."
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
        "name": "Step 1: Data Exploration and Quality Assessment",
        "instructions": STEP1_INSTRUCTIONS,
        "judge": {
            "type": "ExecutableJudge",
            "script": str(SCORING_SCRIPTS / "judge_step1.py"),
            "args": [
                str(SUBMISSION / "step1_answers.json"),
                str(REFERENCE / "step1_answers.json"),
                "/tmp/step1_result.json",
            ],
            "timeout_seconds": 60,
        },
        "submission_path": str(SUBMISSION / "step1_answers.json"),
    }


def step2_config():
    return {
        "id": "feature_engineering",
        "name": "Step 2: Feature Engineering",
        "instructions": STEP2_INSTRUCTIONS,
        "judge": {
            "type": "ExecutableJudge",
            "script": str(SCORING_SCRIPTS / "judge_step2.py"),
            "args": [
                str(SUBMISSION / "features.npz"),
                str(REFERENCE / "step2_test_features.npz"),
                "/tmp/step2_result.json",
            ],
            "timeout_seconds": 300,
        },
        "submission_path": str(SUBMISSION / "features.npz"),
    }


def step3_config():
    return {
        "id": "class_imbalance",
        "name": "Step 3: Handle Class Imbalance",
        "instructions": STEP3_INSTRUCTIONS,
        "judge": {
            "type": "ExecutableJudge",
            "script": str(SCORING_SCRIPTS / "judge_step3.py"),
            "args": [
                str(SUBMISSION / "curated_train.npz"),
                str(REFERENCE / "step2_test_features.npz"),
                "/tmp/step3_result.json",
            ],
            "timeout_seconds": 300,
        },
        "submission_path": str(SUBMISSION / "curated_train.npz"),
    }


def step4_config():
    return {
        "id": "full_pipeline",
        "name": "Step 4: Full Pipeline Prediction",
        "instructions": STEP4_INSTRUCTIONS,
        "judge": {
            "type": "ExecutableJudge",
            "script": str(SCORING_SCRIPTS / "judge_step4.py"),
            "args": [
                str(SUBMISSION / "step4_predict.py"),
                str(SCORING_DATA / "test" / "test_data.npy"),
                str(REFERENCE / "test_labels.npz"),
                "/tmp/step4_result.json",
            ],
            "timeout_seconds": 660,  # 10 min for predict() + 1 min overhead
        },
        "submission_path": str(SUBMISSION / "step4_predict.py"),
    }


# ============================================================
# STEP INSTRUCTION TEXTS
# ============================================================

STEP1_INSTRUCTIONS = """
STEP 1: Data Exploration and Quality Assessment
=================================================

You have access to mouse behavior data in /workspace/env_data/.
This is the CalMS21 dataset -- pose tracking of two interacting mice
with behavior annotations.

Explore the data and report your findings.

Write your answers to /workspace/submission/step1_answers.json
in the following format:

{
    "num_train_sequences": <int>,
    "num_val_sequences": <int>,
    "num_keypoints_per_mouse": <int>,
    "num_classes": <int>,
    "class_names": "<comma-separated class names in alphabetical order>",
    "pct_attack": <float, percentage of training frames labeled attack, rounded to 1 decimal>,
    "pct_other": <float, percentage of training frames labeled other, rounded to 1 decimal>,
    "most_common_transition": "<string, e.g. 'attack->other'>"
}

Notes:
- Load the .npy files with np.load(path, allow_pickle=True).item()
- The data is organized by annotator, then by sequence.
- Use annotator "annotator-id_0" for all your analysis.
- For "most_common_transition": count all frame-to-frame label changes
  where the label actually changes (exclude self-transitions like
  other->other). Report the most frequent non-self transition as
  "label_a->label_b" using class names (not numbers).
- For percentages: count all training frames, compute the fraction
  belonging to each class, multiply by 100, round to 1 decimal place.

Hints:
- Start by loading one file and examining its structure.
- The transition question requires iterating over consecutive frames.
- There are multiple annotators but use annotator-id_0.

Time limit: 30 minutes recommended (of the 4 hour total).
""".strip()


STEP2_INSTRUCTIONS = """
STEP 2: Feature Engineering
=============================

Engineer features from the raw keypoint data that capture
behavior-relevant information.

Using the training data in /workspace/env_data/train/ and
validation data in /workspace/env_data/val/, create feature matrices.

Save to /workspace/submission/features.npz containing:
- "X_train": numpy array [n_train_frames, n_features]
- "y_train": numpy array [n_train_frames] (labels: 0=attack, 1=investigation, 2=mount, 3=other)
- "X_val": numpy array [n_val_frames, n_features]
- "y_val": numpy array [n_val_frames] (validation labels)
- "feature_names": array of feature name strings
- "sequence_ids_train": numpy array [n_train_frames] (which sequence each frame belongs to)
- "sequence_ids_val": numpy array [n_val_frames]

IMPORTANT: Process sequences in the same order they appear in the data
file (sorted by sequence key). Concatenate all frames from all sequences
sequentially. Include ALL frames -- do not drop any.

Feature ideas (not exhaustive):
- Inter-animal distances (nose-to-nose, centroid-to-centroid, etc.)
- Per-mouse velocity and acceleration of centroid
- Body orientation (angle from tail_base to nose)
- Relative orientation between mice
- Area of bounding polygon per mouse
- Temporal features: rolling mean/std of distances over N-frame windows
- All pairwise inter-mouse keypoint distances (7x7 = 49 features)
- Relative position in egocentric coordinates

Good features should capture the DYNAMICS of behavior, not just
static pose. A single frame is often insufficient -- behaviors
unfold over time. Consider temporal context windows.

Your features will be evaluated by:
1. Feature diversity (do you have distance, velocity, orientation,
   and temporal features?)
2. Downstream classification performance (a fixed logistic regression
   trained on your features, evaluated on hidden test data)

Save as: /workspace/submission/features.npz

Time limit: 60 minutes recommended.
""".strip()


STEP3_INSTRUCTIONS = """
STEP 3: Handle Class Imbalance
================================

The behavior classes are extremely imbalanced:
- "other" dominates at ~63% of frames
- "attack" is only ~3% of frames
- "mount" is only ~6%

A naive classifier will predict "other" for everything and get
high accuracy but ZERO macro F1 on the rare behaviors.

Your task: create a curated, BALANCED training dataset that
enables better classification of rare behaviors.

You may:
- Load your features from Step 2 (/workspace/submission/features.npz)
- Or load the raw data and compute features fresh
- Or any combination

Apply data curation strategies such as:
- Undersample the "other" class
- Oversample rare classes (random oversampling, SMOTE, etc.)
- Use sample weights to rebalance
- Remove ambiguous boundary frames (frames where the label
  changes within a short window -- these often have noisy labels)
- Use temporal context to improve label quality (majority vote
  within a sliding window)
- Combine undersampling + oversampling
- Any other strategy you believe will help

Save to /workspace/submission/curated_train.npz containing:
- "X": numpy array [n_samples, n_features] -- curated feature matrix
- "y": numpy array [n_samples] -- curated labels (0, 1, 2, 3)
- "sample_weights": numpy array [n_samples] -- per-sample weights
  (use ones if not applying weighting)
- "curation_strategy": string describing what you did

Your curated data will be evaluated by training a downstream
classifier and measuring macro F1 on hidden test sequences
(excluding "other" class from F1 computation).

Good curation should:
1. Reduce the dominance of "other"
2. Preserve enough "other" examples that the model still learns
   the class (do not remove it entirely)
3. Increase representation of rare classes (especially "attack")
4. Remove noisy labels near behavior transitions
5. Keep the total dataset size manageable (< 2M samples)

Save as: /workspace/submission/curated_train.npz

Time limit: 45 minutes recommended.
""".strip()


STEP4_INSTRUCTIONS = """
STEP 4: Full Pipeline Prediction
==================================

Final challenge: write a complete prediction pipeline.

Create a Python script at /workspace/submission/step4_predict.py
that implements:

    import numpy as np

    def predict(data_path: str) -> np.ndarray:
        \"\"\"
        Given a path to a .npy file containing CalMS21 sequences,
        return predicted class labels (int, 0-3) for every frame.

        The data file has the same structure as the training data:
        data = np.load(data_path, allow_pickle=True).item()
        sequences = data["annotator-id_0"]

        Process sequences in sorted key order. Concatenate all frames.
        Return one label per frame as a 1D numpy array.
        \"\"\"

You MAY load pre-trained models or pre-computed data from
/workspace/submission/. You MAY use any libraries available
in the container.

Your script will be called with a HIDDEN test file containing
sequences you have never seen. Your pipeline must generalize.

CONSTRAINTS:
- predict() must complete in under 10 minutes.
- Must return int labels (0, 1, 2, or 3).
- Must return exactly one label per frame.
- Process sequences in sorted key order.

TIPS:
- Train your model in this step and save it, or train inline.
- Use your features from Step 2 and balancing from Step 3.
- Random Forest or Logistic Regression are good choices.
- Temporal smoothing can help (median filter over predictions).
- Make sure your pipeline handles varying sequence lengths.

Save as: /workspace/submission/step4_predict.py

Time limit: 90 minutes recommended.
""".strip()
```

---

## 9. setup_data.py

```python
#!/usr/bin/env python3
# setup_data.py
"""
E06 CalMS21 Data Setup
=======================

Downloads CalMS21 Task 1 data from Caltech Data, splits into
env_data/ (student-visible) and scoring_data/ (hidden), and
generates all reference artifacts for scoring.

Run this ONCE before building the container image.
Requires network access.

Usage:
    uv run setup_data.py --output-dir /path/to/build/context

Output structure:
    output_dir/
    ├── env_data/
    │   ├── train/
    │   │   └── train_data.npy       (70 training sequences)
    │   ├── val/
    │   │   └── val_data.npy         (12 validation sequences with labels)
    │   └── README.txt
    ├── scoring_data/
    │   ├── test/
    │   │   └── test_data.npy        (7 hidden test sequences)
    │   ├── reference/
    │   │   ├── step1_answers.json
    │   │   ├── step2_test_features.npz
    │   │   └── test_labels.npz
    │   └── scoring_scripts/
    │       ├── judge_step1.py
    │       ├── judge_step2.py
    │       ├── judge_step3.py
    │       └── judge_step4.py
    └── tasks.py
"""

import argparse
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests


CALMS21_URL = "https://data.caltech.edu/records/s0vdx-0k302/files/task1_classic_classification.zip?download=1"
CALMS21_FILENAME = "task1_classic_classification.zip"

# After extraction, expected files:
# calms21_task1_train.npy
# calms21_task1_test.npy (test set -- keypoints only, no annotations for some)
# calms21_convert_to_npy.py (conversion script)

VOCAB = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
VOCAB_INV = {v: k for k, v in VOCAB.items()}

# How many test sequences to keep hidden vs give to student as validation
N_VAL_VISIBLE = 12
N_TEST_HIDDEN = 7

RANDOM_SEED = 42


def download_calms21(work_dir: Path):
    """Download and extract CalMS21 Task 1 data."""
    zip_path = work_dir / CALMS21_FILENAME

    if not zip_path.exists():
        print(f"Downloading CalMS21 Task 1 from {CALMS21_URL}...")
        resp = requests.get(CALMS21_URL, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="")
        print()
    else:
        print(f"Using cached {zip_path}")

    extract_dir = work_dir / "calms21_raw"
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
    else:
        print(f"Using cached extraction at {extract_dir}")

    return extract_dir


def find_npy_files(extract_dir: Path):
    """Find the train and test .npy files in the extracted directory."""
    train_path = None
    test_path = None

    for p in extract_dir.rglob("*.npy"):
        name = p.name.lower()
        if "train" in name:
            train_path = p
        elif "test" in name:
            test_path = p

    if train_path is None:
        raise FileNotFoundError(f"No train .npy found in {extract_dir}")
    if test_path is None:
        raise FileNotFoundError(f"No test .npy found in {extract_dir}")

    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    return train_path, test_path


def load_calms21_data(npy_path: Path):
    """Load a CalMS21 .npy file. Returns the annotator-id_0 sequences."""
    data = np.load(str(npy_path), allow_pickle=True).item()
    # The structure is: {"annotator-id_0": {"seq_0": {...}, "seq_1": {...}, ...}, "metadata": {...}}
    annotator_key = None
    for key in data:
        if key.startswith("annotator"):
            annotator_key = key
            break
    if annotator_key is None:
        # Try the whole dict as sequences
        annotator_key = list(data.keys())[0]

    sequences = data[annotator_key]
    metadata = data.get("metadata", {"vocab": VOCAB})
    return sequences, metadata


def split_test_sequences(test_sequences, n_val, n_hidden, seed=RANDOM_SEED):
    """Split test sequences into visible validation and hidden test sets."""
    rng = np.random.RandomState(seed)
    seq_keys = sorted(test_sequences.keys())

    total = len(seq_keys)
    if total < n_val + n_hidden:
        # Adjust: give more to hidden
        n_hidden = min(n_hidden, total // 2)
        n_val = total - n_hidden

    indices = rng.permutation(total)
    val_keys = [seq_keys[i] for i in indices[:n_val]]
    hidden_keys = [seq_keys[i] for i in indices[n_val:n_val + n_hidden]]

    val_seqs = {k: test_sequences[k] for k in sorted(val_keys)}
    hidden_seqs = {k: test_sequences[k] for k in sorted(hidden_keys)}

    print(f"  Validation: {len(val_seqs)} sequences")
    print(f"  Hidden test: {len(hidden_seqs)} sequences")

    return val_seqs, hidden_seqs


def save_as_npy(sequences, metadata, output_path: Path):
    """Save sequences in CalMS21 .npy format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "annotator-id_0": sequences,
        "metadata": metadata,
    }
    np.save(str(output_path), data, allow_pickle=True)
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


def compute_step1_answers(train_sequences, val_sequences, metadata):
    """Compute ground truth answers for Step 1."""
    vocab = metadata.get("vocab", VOCAB)
    vocab_inv = {v: k for k, v in vocab.items()}

    # Count sequences
    n_train = len(train_sequences)
    n_val = len(val_sequences)

    # Count keypoints
    first_seq = list(train_sequences.values())[0]
    kps = first_seq["keypoints"]
    n_keypoints = kps.shape[3]  # (frames, 2, 2, 7)

    # Count classes
    n_classes = len(vocab)
    class_names = ",".join(sorted(vocab.keys()))

    # Class percentages in training set
    all_labels = np.concatenate([
        seq["annotations"] for seq in train_sequences.values()
    ])
    total_frames = len(all_labels)

    pct_attack = round(100.0 * (all_labels == vocab["attack"]).sum() / total_frames, 1)
    pct_other = round(100.0 * (all_labels == vocab["other"]).sum() / total_frames, 1)

    # Most common non-self transition
    transition_counts = Counter()
    for seq in train_sequences.values():
        ann = seq["annotations"]
        for i in range(len(ann) - 1):
            if ann[i] != ann[i + 1]:
                from_name = vocab_inv[int(ann[i])]
                to_name = vocab_inv[int(ann[i + 1])]
                transition_counts[f"{from_name}->{to_name}"] += 1

    most_common = transition_counts.most_common(1)[0][0] if transition_counts else "unknown"

    answers = {
        "num_train_sequences": n_train,
        "num_val_sequences": n_val,
        "num_keypoints_per_mouse": n_keypoints,
        "num_classes": n_classes,
        "class_names": class_names,
        "pct_attack": pct_attack,
        "pct_other": pct_other,
        "most_common_transition": most_common,
    }

    print(f"  Step 1 answers: {json.dumps(answers, indent=2)}")
    return answers


def compute_reference_features(sequences):
    """Compute reference features for sequences using a canonical pipeline.

    Features computed (per frame):
    - 7 inter-mouse keypoint distances (nose-nose, etc.)
    - 2 centroid distances
    - 2 body orientations (resident, intruder)
    - 1 relative orientation
    - 2 centroid speeds (resident, intruder)
    - 2 centroid accelerations
    - Rolling mean/std of centroid distance (windows: 10, 30)

    Total: ~22 features per frame.
    """
    all_X = []
    all_y = []
    feature_names = None

    for seq_key in sorted(sequences.keys()):
        seq = sequences[seq_key]
        kps = seq["keypoints"]  # (n_frames, 2, 2, 7)
        ann = seq["annotations"]  # (n_frames,)
        n_frames = kps.shape[0]

        features = []
        names = []

        # --- Inter-mouse keypoint distances ---
        kp_names = ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base"]
        for ki, kp_name in enumerate(kp_names):
            dx = kps[:, 0, 0, ki] - kps[:, 1, 0, ki]  # resident x - intruder x
            dy = kps[:, 0, 1, ki] - kps[:, 1, 1, ki]
            dist = np.sqrt(dx**2 + dy**2)
            features.append(dist)
            names.append(f"dist_{kp_name}")

        # --- Centroid distances ---
        res_cx = kps[:, 0, 0, :].mean(axis=1)  # resident centroid x
        res_cy = kps[:, 0, 1, :].mean(axis=1)
        int_cx = kps[:, 1, 0, :].mean(axis=1)  # intruder centroid x
        int_cy = kps[:, 1, 1, :].mean(axis=1)

        centroid_dist = np.sqrt((res_cx - int_cx)**2 + (res_cy - int_cy)**2)
        features.append(centroid_dist)
        names.append("dist_centroid")

        # Nose-to-nose distance
        nose_dx = kps[:, 0, 0, 0] - kps[:, 1, 0, 0]
        nose_dy = kps[:, 0, 1, 0] - kps[:, 1, 1, 0]
        nose_dist = np.sqrt(nose_dx**2 + nose_dy**2)
        features.append(nose_dist)
        names.append("dist_nose_to_nose")

        # --- Body orientation (angle from tail to nose) ---
        for mouse_idx, mouse_name in [(0, "resident"), (1, "intruder")]:
            nose_x = kps[:, mouse_idx, 0, 0]
            nose_y = kps[:, mouse_idx, 1, 0]
            tail_x = kps[:, mouse_idx, 0, 6]
            tail_y = kps[:, mouse_idx, 1, 6]
            angle = np.arctan2(nose_y - tail_y, nose_x - tail_x)
            features.append(angle)
            names.append(f"angle_body_{mouse_name}")

        # --- Relative orientation ---
        bearing = np.arctan2(int_cy - res_cy, int_cx - res_cx)
        res_heading = features[-2]  # resident body angle
        rel_angle = bearing - res_heading
        rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle))
        features.append(rel_angle)
        names.append("angle_relative")

        # --- Centroid velocity ---
        dt = 1.0 / 30.0
        for prefix, cx, cy in [("resident", res_cx, res_cy), ("intruder", int_cx, int_cy)]:
            dx = np.diff(cx, prepend=cx[0])
            dy = np.diff(cy, prepend=cy[0])
            speed = np.sqrt(dx**2 + dy**2) / dt
            features.append(speed)
            names.append(f"speed_{prefix}")

        # --- Centroid acceleration ---
        for prefix_idx in range(2):
            speed_feat = features[-(2 - prefix_idx)]  # resident or intruder speed
            acc = np.diff(speed_feat, prepend=speed_feat[0]) / dt
            features.append(acc)
            prefix = "resident" if prefix_idx == 0 else "intruder"
            names.append(f"acceleration_{prefix}")

        # --- Temporal features: rolling mean/std of centroid distance ---
        for window in [10, 30]:
            kernel = np.ones(window) / window
            roll_mean = np.convolve(centroid_dist, kernel, mode="same")
            roll_sq_mean = np.convolve(centroid_dist**2, kernel, mode="same")
            roll_std = np.sqrt(np.maximum(roll_sq_mean - roll_mean**2, 0))
            features.append(roll_mean)
            names.append(f"roll_mean_centroid_dist_w{window}")
            features.append(roll_std)
            names.append(f"roll_std_centroid_dist_w{window}")

        X_seq = np.column_stack(features)
        all_X.append(X_seq)
        all_y.append(ann)

        if feature_names is None:
            feature_names = names

    X = np.vstack(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.int32)

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_names


def write_readme(env_data_dir: Path):
    """Write a brief README for the student."""
    readme = """CalMS21 Mouse Behavior Dataset
==============================

This dataset contains pose tracking data from resident-intruder
mouse experiments. The MARS computer vision pipeline tracked
7 keypoints per mouse at 30 Hz.

Files:
- train/train_data.npy: Training sequences with behavior annotations
- val/val_data.npy: Validation sequences with behavior annotations

Loading:
    import numpy as np
    data = np.load("train/train_data.npy", allow_pickle=True).item()
    sequences = data["annotator-id_0"]
    metadata = data["metadata"]
    vocab = metadata["vocab"]  # {"attack": 0, "investigation": 1, "mount": 2, "other": 3}

    for seq_key in sorted(sequences.keys()):
        seq = sequences[seq_key]
        keypoints = seq["keypoints"]     # (n_frames, 2, 2, 7) = [frame, mouse, xy, keypoint]
        annotations = seq["annotations"]  # (n_frames,) = class label per frame

Keypoints (per mouse):
    0: nose
    1: left_ear
    2: right_ear
    3: neck
    4: left_hip
    5: right_hip
    6: tail_base

Mouse identity:
    Mouse 0 = resident (black mouse)
    Mouse 1 = intruder (white mouse)

Image dimensions: 1024 x 570 pixels
Frame rate: 30 Hz

Behavior classes:
    0: attack (~3% of frames)
    1: investigation (~29%)
    2: mount (~6%)
    3: other (~63%)

Evaluation metric: Macro F1 on classes 0, 1, 2 (excluding "other")
"""
    (env_data_dir / "README.txt").write_text(readme)


def setup(output_dir: Path):
    """Main setup function."""
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    env_data = output_dir / "env_data"
    scoring_data = output_dir / "scoring_data"

    # Create directories
    (env_data / "train").mkdir(parents=True, exist_ok=True)
    (env_data / "val").mkdir(parents=True, exist_ok=True)
    (scoring_data / "test").mkdir(parents=True, exist_ok=True)
    (scoring_data / "reference").mkdir(parents=True, exist_ok=True)
    (scoring_data / "scoring_scripts").mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    print("=" * 60)
    print("STEP 1: Download CalMS21 data")
    print("=" * 60)
    extract_dir = download_calms21(work_dir)

    # Step 2: Find and load data files
    print("\n" + "=" * 60)
    print("STEP 2: Load data files")
    print("=" * 60)
    train_npy_path, test_npy_path = find_npy_files(extract_dir)

    train_sequences, metadata = load_calms21_data(train_npy_path)
    test_sequences, _ = load_calms21_data(test_npy_path)

    print(f"  Training: {len(train_sequences)} sequences")
    print(f"  Test: {len(test_sequences)} sequences")

    # Count total frames
    train_frames = sum(seq["keypoints"].shape[0] for seq in train_sequences.values())
    test_frames = sum(seq["keypoints"].shape[0] for seq in test_sequences.values())
    print(f"  Training frames: {train_frames:,}")
    print(f"  Test frames: {test_frames:,}")

    # Step 3: Split test into val (visible) and hidden
    print("\n" + "=" * 60)
    print("STEP 3: Split test sequences")
    print("=" * 60)
    val_sequences, hidden_sequences = split_test_sequences(
        test_sequences, N_VAL_VISIBLE, N_TEST_HIDDEN,
    )

    # Step 4: Save data files
    print("\n" + "=" * 60)
    print("STEP 4: Save data files")
    print("=" * 60)

    save_as_npy(train_sequences, metadata, env_data / "train" / "train_data.npy")
    save_as_npy(val_sequences, metadata, env_data / "val" / "val_data.npy")
    save_as_npy(hidden_sequences, metadata, scoring_data / "test" / "test_data.npy")
    write_readme(env_data)

    # Step 5: Generate Step 1 answers
    print("\n" + "=" * 60)
    print("STEP 5: Generate Step 1 reference answers")
    print("=" * 60)
    answers = compute_step1_answers(train_sequences, val_sequences, metadata)
    answers_path = scoring_data / "reference" / "step1_answers.json"
    with open(answers_path, "w") as f:
        json.dump(answers, f, indent=2)
    print(f"  Saved: {answers_path}")

    # Step 6: Compute reference features for hidden test sequences
    print("\n" + "=" * 60)
    print("STEP 6: Compute reference features for hidden test sequences")
    print("=" * 60)
    X_test, y_test, feature_names = compute_reference_features(hidden_sequences)
    print(f"  Hidden test: {X_test.shape[0]} frames, {X_test.shape[1]} features")
    print(f"  Feature names: {feature_names}")

    test_features_path = scoring_data / "reference" / "step2_test_features.npz"
    np.savez(
        str(test_features_path),
        X=X_test,
        y=y_test,
        feature_names=np.array(feature_names),
    )
    print(f"  Saved: {test_features_path}")

    # Save test labels separately (for Step 4 judge)
    test_labels_path = scoring_data / "reference" / "test_labels.npz"
    np.savez(
        str(test_labels_path),
        y=y_test,
    )
    print(f"  Saved: {test_labels_path}")

    # Step 7: Print summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"  env_data:     {env_data}")
    print(f"  scoring_data: {scoring_data}")

    # Size summary
    env_size = sum(f.stat().st_size for f in env_data.rglob("*") if f.is_file())
    score_size = sum(f.stat().st_size for f in scoring_data.rglob("*") if f.is_file())
    print(f"  env_data size:     {env_size / 1e6:.1f} MB")
    print(f"  scoring_data size: {score_size / 1e6:.1f} MB")

    # Class distribution in hidden test
    for cls in range(4):
        name = VOCAB_INV[cls]
        count = int((y_test == cls).sum())
        pct = 100.0 * count / len(y_test)
        print(f"  Hidden test {name}: {count} frames ({pct:.1f}%)")

    # Clean up work directory
    print(f"\n  Work directory (can delete): {work_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E06 CalMS21 Data Setup")
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for env_data/ and scoring_data/",
    )
    args = parser.parse_args()
    setup(Path(args.output_dir))
```

---

## 10. env_requirements.txt

```
numpy>=1.26.0,<2.0
scipy>=1.12.0
scikit-learn>=1.4.0
pandas>=2.1.0
```

These are the only dependencies needed. No GPU libraries, no deep learning frameworks. The task is pure data curation with classical ML.

---

## 11. pre_hook Implementation

```python
# pre_hook.py
"""
Pre-hook: runs before the student starts each step.
Creates the submission directory and prints step-specific hints.
"""

from pathlib import Path


def pre_hook(step_id: str) -> dict:
    """Called before each step begins."""
    submission = Path("/workspace/submission")
    submission.mkdir(parents=True, exist_ok=True)

    context = {
        "env_data_dir": "/workspace/env_data",
        "submission_dir": "/workspace/submission",
    }

    if step_id == "feature_engineering":
        # Check if Step 1 was completed
        step1_path = submission / "step1_answers.json"
        if step1_path.exists():
            context["step1_completed"] = True

    if step_id == "class_imbalance":
        # Check if Step 2 features exist
        features_path = submission / "features.npz"
        if features_path.exists():
            context["step2_features_available"] = True
            context["step2_features_path"] = str(features_path)

    if step_id == "full_pipeline":
        # Check available artifacts from previous steps
        features_path = submission / "features.npz"
        curated_path = submission / "curated_train.npz"
        context["step2_features_available"] = features_path.exists()
        context["step3_curated_available"] = curated_path.exists()

    return context
```

---

## 12. Expected Score Distributions

### 12.1 Overall Task Score (average of 4 steps)

| Score Range | Interpretation | Expected % |
|-------------|---------------|------------|
| 0.80+ | Excellent data curator -- strong features, effective balancing, good generalization | 5% |
| 0.60 -- 0.79 | Good -- solid features and some balancing | 20% |
| 0.40 -- 0.59 | Moderate -- basic features work, imbalance partially addressed | 35% |
| 0.20 -- 0.39 | Weak -- limited features, poor balancing | 25% |
| < 0.20 | Struggling -- broken pipeline or misunderstanding | 15% |

### 12.2 Benchmark Reference Points

From the CalMS21 paper and competition:

| Method | Macro F1 (excl. other) | Notes |
|--------|----------------------|-------|
| Random guessing | ~0.33 | Uniform random among 3 classes |
| Predict majority (investigation) | ~0.33 | Only one class correct |
| Raw keypoints + Logistic Regression | ~0.50 | No feature engineering |
| Basic features + Random Forest | ~0.65 | Distance + velocity features |
| NeurIPS 2021 baseline | 0.793 | Standard neural architecture |
| Baseline + TREBA features | 0.829 | Self-supervised pre-training |
| MABe 2021 top-1 | 0.864 | Winning challenge solution |

The logistic regression downstream model means the ceiling is ~0.65-0.70 with good features and balancing (limited by model capacity). Achieving 0.79+ would require exceptional features that capture almost all discriminative information.

---

## 13. Anti-Gaming Summary

| Step | Anti-Gaming Mechanism |
|------|----------------------|
| 1 | Data-dependent answers (exact percentages, transition counts) -- cannot guess |
| 2 | Feature diversity checked by name + downstream model evaluation on val -- cannot submit random features |
| 3 | Macro F1 excludes "other" on hidden test -- predicting "other" always = score 0; degenerate submissions detected |
| 4 | predict() called on hidden test data -- student cannot see test sequences; constant predictions = score 0; 10-min timeout |

### Cross-Step Anti-Gaming

1. **Hidden test sequences** -- 7 sequences from the official CalMS21 test set are never shown to the student. Different animal pairs than training.
2. **Fixed downstream model** -- student cannot overfit model choice. Logistic regression is the evaluator for Steps 2-3.
3. **Macro F1 on rare classes** -- the metric explicitly measures what's hardest: classifying attack, investigation, and mount. "Other" is excluded.
4. **Sequence-level splits** -- no frame-level data leakage between train/val/test. Different animal identities.

---

## 14. References

1. Sun, J.J. et al. (2021). **The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions.** NeurIPS Track on Datasets and Benchmarks. [arXiv:2104.02710](https://arxiv.org/abs/2104.02710)

2. Segalin, C. et al. (2021). **The Mouse Action Recognition System (MARS) software pipeline for automated analysis of social behaviors in mice.** eLife 10:e63720. [DOI:10.7554/eLife.63720](https://doi.org/10.7554/eLife.63720)

3. Sun, J.J. et al. (2021). **Task Programming: Learning Data Efficient Behavior Representations.** CVPR 2021.

4. Anderson, D.J. & Perona, P. (2014). **Toward a Science of Computational Ethology.** Neuron 84(1):18-31.

5. CalMS21 official dataset: https://data.caltech.edu/records/s0vdx-0k302

6. MABe 2021 challenge: https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications

---

*spec version: 1.0*
*last updated: 2026-02-09*
