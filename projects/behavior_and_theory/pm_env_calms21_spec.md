# CalMS21 Behavior Classification Environment – Technical Specification

> pm_env for mouse social behavior classification using the CalMS21 dataset

---

## 1. Dataset Overview

### 1.1 Source & Access

| Resource | URL |
|----------|-----|
| Official Dataset | https://data.caltech.edu/records/s0vdx-0k302 |
| Paper (NeurIPS 2021) | https://arxiv.org/abs/2104.02710 |
| MARS Pipeline Paper | https://doi.org/10.7554/eLife.63720 |
| AIcrowd Challenge | https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications |
| Baseline Code | https://gitlab.aicrowd.com/aicrowd/research/mab-e/mab-e-baselines |

### 1.2 Download

```python
import os, requests, zipfile

url = "https://data.caltech.edu/records/s0vdx-0k302/files/task1_classic_classification.zip?download=1"
fname = "task1_classic_classification.zip"

if not os.path.isfile(fname):
    r = requests.get(url)
    with open(fname, "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile(fname, "r") as z:
        z.extractall(".")
```

### 1.3 File Formats

| File Type | Format | Description |
|-----------|--------|-------------|
| Raw data | `.json` | Original CalMS21 format |
| Converted | `.npy` | Dictionary stored via `np.save(..., allow_pickle=True)` |
| Conversion script | `calms21_convert_to_npy.py` | Available from dataset URL |

---

## 2. Data Schema

### 2.1 Keypoint Definition

MARS tracks **7 anatomical keypoints** per mouse (top-view):

| Index | Body Part |
|-------|-----------|
| 0 | nose |
| 1 | left ear |
| 2 | right ear |
| 3 | neck (base of neck) |
| 4 | left hip |
| 5 | right hip |
| 6 | tail base |

### 2.2 Data Structure (per sequence)

```python
{
    "keypoints": np.ndarray,   # shape: (n_frames, 2, 2, 7)
                                # axes: [frame, mouse_id, xy_coord, keypoint]
    "scores": np.ndarray,       # shape: (n_frames, 2, 7)
                                # confidence scores per keypoint [0, 1]
    "annotations": np.ndarray,  # shape: (n_frames,)
                                # integer class labels
    "metadata": {
        "annotator-id": int,
        "vocab": {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
    }
}
```

**Notes:**
- Mouse 0 = resident (black mouse, behavior subject)
- Mouse 1 = intruder (white mouse)
- Image dimensions: 1024 x 570 pixels
- Frame rate: **30 Hz**

### 2.3 Annotation Schema (Task 1)

| Class ID | Behavior | Description |
|----------|----------|-------------|
| 0 | attack | Aggressive behavior by resident |
| 1 | investigation | Close sniffing/investigation |
| 2 | mount | Mounting behavior |
| 3 | other | No annotated behavior |

---

## 3. Class Distribution

### 3.1 Training Set (Task 1)

| Behavior | Percentage | Absolute (approx) |
|----------|------------|-------------------|
| attack | 2.76% | ~14,000 frames |
| investigation | 28.88% | ~147,000 frames |
| mount | 5.64% | ~29,000 frames |
| other | **62.72%** | ~318,000 frames |

**Total training frames:** ~507,738 frames (70 sequences)
**Total test frames:** ~262,107 frames (19 sequences)

### 3.2 Imbalance Considerations

- "other" class is **excluded** from macro F1 calculation
- effective 3-class problem for evaluation: attack, investigation, mount
- attack is the rarest class (~2.8%), critical for F1

---

## 4. Train/Test Split Protocol

### 4.1 Official Split

| Split | Sequences | Frames |
|-------|-----------|--------|
| Train | 70 | 507,738 |
| Test | 19 | 262,107 |

### 4.2 Split Strategy

**CRITICAL:** train and test sets are split by **animal identity**.

- animals in test set are **different individuals** from training
- same recording setup, date range, and annotator
- prevents identity memorization and temporal leakage

### 4.3 Validation Split (for development)

```python
from sklearn.model_selection import train_test_split

# stratify by presence of attack behavior
idx_train, idx_val = train_test_split(
    sequence_keys,
    stratify=[has_attack[k] for k in sequence_keys],
    test_size=0.2,
    random_state=42
)
```

**Rules:**
- keep entire sequences together (no frame-level splits)
- stratify on attack presence to preserve rare class

---

## 5. Evaluation Metrics

### 5.1 Primary Metric: Macro F1 Score

```python
from sklearn.metrics import f1_score

# predictions and labels are integer arrays
# exclude "other" class (id=3) from calculation
mask = labels != 3
f1 = f1_score(
    labels[mask],
    predictions[mask],
    average="macro",
    labels=[0, 1, 2]  # attack, investigation, mount only
)
```

### 5.2 Secondary Metric: Precision (tie-breaker)

```python
from sklearn.metrics import precision_score

precision = precision_score(
    labels[mask],
    predictions[mask],
    average="macro",
    labels=[0, 1, 2]
)
```

### 5.3 Additional Metrics (for analysis)

```python
from sklearn.metrics import classification_report, average_precision_score

# per-class F1
report = classification_report(
    labels[mask],
    predictions[mask],
    target_names=["attack", "investigation", "mount"],
    output_dict=True
)

# mean average precision (if probabilities available)
map_score = average_precision_score(
    one_hot_labels,
    probabilities,
    average="macro"
)
```

---

## 6. Benchmark Results

### 6.1 Official Baselines (from paper)

| Method | Macro F1 | Notes |
|--------|----------|-------|
| Baseline (neural) | 0.793 +/- 0.011 | Standard architecture |
| Baseline + Task Programming | 0.829 +/- 0.004 | TREBA features |
| **MABe 2021 Top-1** | **0.864 +/- 0.011** | Winning solution |

### 6.2 Grading Thresholds (proposed)

| Grade | Macro F1 | Interpretation |
|-------|----------|----------------|
| Fail | < 0.50 | Below random baseline for 3-class |
| Pass | 0.50 - 0.70 | Basic feature engineering works |
| Good | 0.70 - 0.79 | Solid model, competitive with baseline |
| Excellent | 0.79 - 0.86 | Near state-of-the-art |
| Outstanding | > 0.86 | Challenge-winning level |

---

## 7. Feature Engineering

### 7.1 Raw Features (28 dimensions per frame)

```python
# keypoints: (n_frames, 2, 2, 7)
# reshape to: (n_frames, 28)
raw_features = keypoints.reshape(n_frames, -1)
```

### 7.2 Essential Derived Features

#### 7.2.1 Inter-Animal Distance

```python
def compute_interanimal_distance(keypoints):
    """
    keypoints: (n_frames, 2, 2, 7) - [frame, mouse, xy, keypoint]
    """
    # centroid per mouse (average of all keypoints)
    centroids = keypoints.mean(axis=3)  # (n_frames, 2, 2)

    # euclidean distance between centroids
    diff = centroids[:, 0, :] - centroids[:, 1, :]  # (n_frames, 2)
    distance = np.sqrt((diff ** 2).sum(axis=1))  # (n_frames,)
    return distance

def compute_nose_to_nose_distance(keypoints):
    """nose keypoint is index 0"""
    nose_diff = keypoints[:, 0, :, 0] - keypoints[:, 1, :, 0]
    return np.sqrt((nose_diff ** 2).sum(axis=1))

def compute_nose_to_tail_distance(keypoints):
    """resident nose to intruder tail"""
    diff = keypoints[:, 0, :, 0] - keypoints[:, 1, :, 6]  # nose to tail
    return np.sqrt((diff ** 2).sum(axis=1))
```

#### 7.2.2 Velocity

```python
def compute_velocity(keypoints, dt=1/30):
    """
    Compute centroid velocity for each mouse
    """
    centroids = keypoints.mean(axis=3)  # (n_frames, 2, 2)

    # finite difference
    velocity = np.diff(centroids, axis=0) / dt  # (n_frames-1, 2, 2)

    # pad with zeros
    velocity = np.vstack([np.zeros((1, 2, 2)), velocity])

    # speed (magnitude)
    speed = np.sqrt((velocity ** 2).sum(axis=2))  # (n_frames, 2)
    return velocity, speed
```

#### 7.2.3 Body Orientation Angle

```python
def compute_body_angle(keypoints):
    """
    Angle from tail to neck defines body orientation
    """
    # neck (idx 3) and tail (idx 6)
    neck = keypoints[:, :, :, 3]  # (n_frames, 2, 2)
    tail = keypoints[:, :, :, 6]

    direction = neck - tail  # (n_frames, 2, 2)
    angle = np.arctan2(direction[:, :, 1], direction[:, :, 0])  # (n_frames, 2)
    return angle
```

#### 7.2.4 Relative Angle Between Mice

```python
def compute_relative_angle(keypoints):
    """
    Angle from resident to intruder relative to resident's heading
    """
    body_angle = compute_body_angle(keypoints)
    resident_angle = body_angle[:, 0]  # resident heading

    # vector from resident to intruder centroid
    centroids = keypoints.mean(axis=3)
    diff = centroids[:, 1, :] - centroids[:, 0, :]
    bearing = np.arctan2(diff[:, 1], diff[:, 0])

    # relative angle (wrap to [-pi, pi])
    rel_angle = bearing - resident_angle
    rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle))
    return rel_angle
```

#### 7.2.5 Distance to Walls

```python
def compute_wall_distance(keypoints, width=1024, height=570):
    """
    Distance from resident centroid to nearest wall
    """
    centroids = keypoints.mean(axis=3)[:, 0, :]  # resident only

    dist_left = centroids[:, 0]
    dist_right = width - centroids[:, 0]
    dist_top = centroids[:, 1]
    dist_bottom = height - centroids[:, 1]

    return np.min([dist_left, dist_right, dist_top, dist_bottom], axis=0)
```

### 7.3 Temporal Features

#### 7.3.1 Sliding Window Statistics

```python
def compute_window_features(features, window_sizes=[10, 30, 90]):
    """
    Compute mean/std over sliding windows
    window_sizes in frames (at 30Hz: 10=0.33s, 30=1s, 90=3s)
    """
    results = []
    for ws in window_sizes:
        # rolling mean
        mean = np.convolve(features, np.ones(ws)/ws, mode='same')
        # rolling std (approximate)
        sq_mean = np.convolve(features**2, np.ones(ws)/ws, mode='same')
        std = np.sqrt(np.maximum(sq_mean - mean**2, 0))
        results.extend([mean, std])
    return np.column_stack(results)
```

#### 7.3.2 MABe Winning Solution Features

From the top-1 solution:
- **egocentric coordinates:** all positions relative to resident's heading
- **PCA of pairwise distances:** 14x14 distance matrix (2 mice x 7 keypoints) reduced via PCA
- **temporal context:** 100 past + 100 future frames (skip factor 2) for Task 1

```python
def compute_pairwise_distances(keypoints):
    """
    Compute all pairwise distances between keypoints
    Returns upper triangle of 14x14 distance matrix (91 features)
    """
    n_frames = keypoints.shape[0]
    # reshape to (n_frames, 14, 2)
    all_kps = keypoints.transpose(0, 1, 3, 2).reshape(n_frames, 14, 2)

    # pairwise distances
    diff = all_kps[:, :, np.newaxis, :] - all_kps[:, np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))  # (n_frames, 14, 14)

    # upper triangle indices
    triu_idx = np.triu_indices(14, k=1)
    return dists[:, triu_idx[0], triu_idx[1]]  # (n_frames, 91)
```

---

## 8. Submission Format

### 8.1 Prediction Dictionary

```python
submission = {
    "<sequence_id_1>": [0, 1, 2, 3, ...],  # list of int, length = n_frames
    "<sequence_id_2>": [3, 3, 1, 0, ...],
    ...
}

# save as .npy
np.save("submission.npy", submission, allow_pickle=True)
```

### 8.2 Validation

```python
def validate_submission(submission, test_data, vocab):
    """Check submission format"""
    for seq_id, preds in submission.items():
        assert seq_id in test_data, f"unknown sequence: {seq_id}"
        expected_len = len(test_data[seq_id]["keypoints"])
        assert len(preds) == expected_len, f"length mismatch for {seq_id}"
        assert all(p in vocab.values() for p in preds), f"invalid class id"
    return True
```

---

## 9. Anti-Gaming Protocols

### 9.1 Temporal Leakage Prevention

```python
def enforce_sequence_split(train_sequences, test_sequences):
    """Ensure no overlap between train and test sequences"""
    train_set = set(train_sequences.keys())
    test_set = set(test_sequences.keys())
    assert train_set.isdisjoint(test_set), "sequence leak detected!"
```

### 9.2 Identity Leakage Prevention

- test set contains **different animal pairs** than training
- no frame-level splits within sequences
- validation split must also maintain sequence integrity

### 9.3 Minimum Requirements

| Requirement | Threshold | Rationale |
|-------------|-----------|-----------|
| Min train sequences | >= 50 | Statistical power |
| Min test sequences | >= 15 | Reliable evaluation |
| Min frames per class | >= 1000 | Class representation |
| Cross-validation folds | >= 5 | Variance estimation |

### 9.4 Random Baseline Check

```python
def compute_random_baseline(test_data, n_trials=100):
    """Expected F1 for random guessing"""
    from sklearn.metrics import f1_score

    all_labels = np.concatenate([
        test_data[k]["annotations"] for k in test_data
    ])
    mask = all_labels != 3  # exclude "other"
    labels = all_labels[mask]

    f1s = []
    for _ in range(n_trials):
        preds = np.random.choice([0, 1, 2], size=len(labels))
        f1s.append(f1_score(labels, preds, average="macro", labels=[0, 1, 2]))

    return np.mean(f1s), np.std(f1s)
    # expected: ~0.33 +/- 0.01
```

---

## 10. Complete Evaluation Pipeline

```python
import numpy as np
from sklearn.metrics import f1_score, precision_score, classification_report

def evaluate_calms21_submission(
    submission: dict,
    ground_truth: dict,
    vocab: dict = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
) -> dict:
    """
    Official CalMS21 Task 1 evaluation.

    Args:
        submission: dict mapping sequence_id -> list of predicted class ids
        ground_truth: dict mapping sequence_id -> {"annotations": np.array}
        vocab: class name to id mapping

    Returns:
        dict with F1, precision, and per-class metrics
    """
    all_preds = []
    all_labels = []

    for seq_id in ground_truth:
        if seq_id not in submission:
            raise ValueError(f"missing sequence: {seq_id}")

        preds = np.array(submission[seq_id])
        labels = ground_truth[seq_id]["annotations"]

        if len(preds) != len(labels):
            raise ValueError(f"length mismatch for {seq_id}")

        all_preds.extend(preds)
        all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # exclude "other" class
    mask = all_labels != vocab["other"]
    preds_filtered = all_preds[mask]
    labels_filtered = all_labels[mask]

    # primary metric: macro F1 (excluding other)
    f1_macro = f1_score(
        labels_filtered,
        preds_filtered,
        average="macro",
        labels=[vocab["attack"], vocab["investigation"], vocab["mount"]]
    )

    # secondary metric: macro precision
    precision_macro = precision_score(
        labels_filtered,
        preds_filtered,
        average="macro",
        labels=[vocab["attack"], vocab["investigation"], vocab["mount"]]
    )

    # per-class report
    report = classification_report(
        labels_filtered,
        preds_filtered,
        labels=[vocab["attack"], vocab["investigation"], vocab["mount"]],
        target_names=["attack", "investigation", "mount"],
        output_dict=True,
        zero_division=0
    )

    return {
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "per_class": {
            "attack": report["attack"],
            "investigation": report["investigation"],
            "mount": report["mount"]
        },
        "n_frames_evaluated": mask.sum(),
        "n_frames_total": len(all_labels)
    }
```

---

## 11. Student Task Description

### 11.1 Objective

Build a frame-level behavior classifier for mouse social interactions that predicts whether each frame contains **attack**, **investigation**, **mount**, or **other** behavior.

### 11.2 Requirements

1. **Data Loading:** use provided `Loading_CalMS21_data.ipynb` notebook
2. **Feature Engineering:** compute at least 3 types of derived features:
   - inter-animal distances
   - velocities
   - temporal context (rolling statistics)
3. **Model:** any sklearn/pytorch classifier (logistic regression, random forest, MLP, etc.)
4. **Evaluation:** report macro F1 score (excluding "other" class)

### 11.3 Deliverables

1. Feature extraction code (documented)
2. Model training code
3. Evaluation on held-out validation set
4. Final predictions on test set as `.npy` file
5. Brief report (~500 words) explaining approach

### 11.4 Grading Rubric

| Component | Weight | Criteria |
|-----------|--------|----------|
| Feature engineering | 30% | creativity, biological motivation |
| Model implementation | 25% | correctness, code quality |
| Evaluation | 20% | proper split, correct metrics |
| Performance (F1) | 15% | see thresholds in section 6.2 |
| Report | 10% | clarity, insight |

---

## 12. References

1. Sun, J.J. et al. (2021). **The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions.** NeurIPS Track on Datasets and Benchmarks. [arXiv:2104.02710](https://arxiv.org/abs/2104.02710)

2. Segalin, C. et al. (2021). **The Mouse Action Recognition System (MARS) software pipeline for automated analysis of social behaviors in mice.** eLife 10:e63720. [DOI:10.7554/eLife.63720](https://doi.org/10.7554/eLife.63720)

3. Sun, J.J. et al. (2021). **Task Programming: Learning Data Efficient Behavior Representations.** CVPR 2021. [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Task_Programming_Learning_Data_Efficient_Behavior_Representations_CVPR_2021_paper.html)

4. Anderson, D.J. & Perona, P. (2014). **Toward a Science of Computational Ethology.** Neuron 84(1):18-31. [DOI:10.1016/j.neuron.2014.09.005](https://doi.org/10.1016/j.neuron.2014.09.005)

---

## Appendix A: Quick Start Code

```python
# complete minimal example
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 1. load data
data = np.load("calms21_task1_train.npy", allow_pickle=True).item()
dataset = data["annotator-id_0"]
vocab = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}

# 2. extract features (simple: flatten keypoints)
X_train, y_train = [], []
for seq_id, seq in dataset.items():
    kps = seq["keypoints"].reshape(len(seq["keypoints"]), -1)
    X_train.append(kps)
    y_train.append(seq["annotations"])

X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)

# 3. train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# 4. predict and evaluate
y_pred = clf.predict(X_train)
mask = y_train != 3
f1 = f1_score(y_train[mask], y_pred[mask], average="macro", labels=[0, 1, 2])
print(f"Train F1 (excl. other): {f1:.3f}")
```

---

*spec version: 1.0*
*last updated: 2026-02-02*
