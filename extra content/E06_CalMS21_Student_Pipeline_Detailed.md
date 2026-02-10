# E06: CalMS21 – Detailed Student Pipeline

> full spec for student journey from start to evaluation
> task: E06-T1 (behavior classification from pose data)

---

## 1. What Student Sees at Start

### 1.1 Task Description (in interface)

```
TASK: Mouse Social Behavior Classification

You have access to the CalMS21 dataset – pose tracking data from resident-intruder
mouse experiments. The MARS computer vision pipeline has tracked 7 keypoints per
mouse at 30 Hz.

Your goal: Build a frame-level classifier that predicts social behavior categories
(attack, investigation, mount, or other) from the pose data.

Data format:
- Keypoints shape: [n_frames, 2, 2, 7] = [frame, mouse_id, xy, keypoint]
- Mouse 0 = resident (black), Mouse 1 = intruder (white)
- 7 keypoints: nose, left_ear, right_ear, neck, left_hip, right_hip, tail_base

Classes to predict:
- 0: attack (aggressive behavior, ~2.8% of frames)
- 1: investigation (sniffing/exploring, ~28.9% of frames)
- 2: mount (mounting behavior, ~5.6% of frames)
- 3: other (no annotated behavior, ~62.7% of frames)

Evaluation: Macro F1 score on classes 0, 1, 2 (excluding "other")
- Baseline: 0.793
- SOTA: 0.864
- Pass threshold: ≥0.50
- Excellent threshold: ≥0.79

Time limit: 4 hours
Memory limit: 4 GB
```

### 1.2 Starter Code (what student receives)

```python
# starter_code.py
"""
CalMS21 Behavior Classification Task
=====================================

Complete the `classify_behavior()` function below.

You have access to:
- numpy, pandas, scipy
- sklearn (RandomForestClassifier, SVM, MLP, etc.)
- Training data with annotations
- Test data (keypoints only, no annotations)

Your classifier should work at the FRAME level, predicting one class per frame.
"""

import numpy as np
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# ============== CONSTANTS ==============

KEYPOINT_NAMES = [
    "nose",       # 0
    "left_ear",   # 1
    "right_ear",  # 2
    "neck",       # 3
    "left_hip",   # 4
    "right_hip",  # 5
    "tail_base",  # 6
]

CLASS_VOCAB = {
    "attack": 0,
    "investigation": 1,
    "mount": 2,
    "other": 3
}

FRAME_RATE = 30  # Hz
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 570


# ============== DATA LOADING (provided) ==============

def load_train_data(path: str = "calms21_task1_train.npy") -> Dict:
    """
    Load CalMS21 training data.

    Returns:
        dict with structure:
        {
            "annotator-id_0": {
                "<sequence_id>": {
                    "keypoints": np.ndarray,   # (n_frames, 2, 2, 7)
                    "scores": np.ndarray,      # (n_frames, 2, 7)
                    "annotations": np.ndarray, # (n_frames,) int labels
                },
                ...
            }
        }
    """
    data = np.load(path, allow_pickle=True).item()
    return data


def load_test_data(path: str = "calms21_task1_test.npy") -> Dict:
    """
    Load CalMS21 test data (no annotations).

    Returns:
        dict with same structure but no "annotations" key
    """
    data = np.load(path, allow_pickle=True).item()
    return data


# ============== YOUR IMPLEMENTATION ==============

def classify_behavior(
    train_data: Dict,
    test_data: Dict,
) -> Dict[str, List[int]]:
    """
    Train a behavior classifier and generate predictions.

    Args:
        train_data: Training sequences with keypoints and annotations
        test_data: Test sequences with keypoints only

    Returns:
        predictions: dict mapping sequence_id -> list of predicted class ids
                    Each list has length = n_frames for that sequence
                    Values must be in {0, 1, 2, 3}

    Example output:
        {
            "sequence_001": [3, 3, 1, 1, 1, 0, 0, 3, ...],
            "sequence_002": [3, 3, 3, 2, 2, 2, 3, ...],
            ...
        }

    Hints:
        1. Extract features from keypoints (distances, velocities, angles)
        2. Handle class imbalance (attack is only 2.8%!)
        3. Consider temporal context (behaviors span multiple frames)
        4. Random Forest or MLP work well as baselines
    """
    # TODO: Implement this function
    raise NotImplementedError("Complete this function")


# ============== HELPER FUNCTIONS (you may use these) ==============

def compute_centroid(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute centroid (center of mass) for each mouse.

    Args:
        keypoints: (n_frames, 2, 2, 7) array

    Returns:
        centroids: (n_frames, 2, 2) array - [frame, mouse, xy]
    """
    return keypoints.mean(axis=3)  # average over 7 keypoints


def compute_distance(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance between two point arrays.

    Args:
        point1, point2: (..., 2) arrays with x, y coordinates

    Returns:
        distances: (...) array of distances
    """
    diff = point1 - point2
    return np.sqrt((diff ** 2).sum(axis=-1))


# ============== TEST YOUR SOLUTION ==============

if __name__ == "__main__":
    # Load data
    train_data = load_train_data()
    test_data = load_test_data()

    # Run classifier
    predictions = classify_behavior(
        train_data["annotator-id_0"],
        test_data["annotator-id_0"]
    )

    # Validate output format
    print(f"Number of sequences predicted: {len(predictions)}")
    for seq_id, preds in list(predictions.items())[:3]:
        print(f"  {seq_id}: {len(preds)} frames, classes: {set(preds)}")

    # Save submission
    np.save("submission.npy", predictions, allow_pickle=True)
    print("Saved submission.npy")
```

### 1.3 Reference Documentation (available to student)

```markdown
## CalMS21 Data Structure

### Keypoint Layout (top-down view)

```
        nose (0)
          |
   left_ear (1)  right_ear (2)
          |
        neck (3)
          |
  left_hip (4)  right_hip (5)
          |
      tail_base (6)
```

### Data Arrays

| Array | Shape | Description |
|-------|-------|-------------|
| keypoints | (n_frames, 2, 2, 7) | [frame, mouse, xy, keypoint] |
| scores | (n_frames, 2, 7) | confidence per keypoint [0, 1] |
| annotations | (n_frames,) | class labels {0, 1, 2, 3} |

### Axis Interpretation

- **Axis 0 (n_frames):** time, at 30 Hz
- **Axis 1 (mouse_id):** 0 = resident (black), 1 = intruder (white)
- **Axis 2 (xy):** 0 = x coordinate, 1 = y coordinate
- **Axis 3 (keypoint):** 0-6, see keypoint names above

### Coordinate System

- Origin: top-left corner of video frame
- X: increases rightward (0 to 1024)
- Y: increases downward (0 to 570)

### Class Definitions

| Class | Behavior | Typical Duration | Frequency |
|-------|----------|------------------|-----------|
| 0 | attack | 0.5-5 seconds | ~2.8% (rare!) |
| 1 | investigation | 1-10 seconds | ~28.9% |
| 2 | mount | 1-5 seconds | ~5.6% |
| 3 | other | variable | ~62.7% |

### Useful Feature Ideas

1. **Inter-animal distances:** nose-to-nose, nose-to-tail, centroid-to-centroid
2. **Velocities:** centroid speed, acceleration
3. **Body orientation:** angle from tail to neck
4. **Relative position:** angle from resident to intruder
5. **Temporal context:** rolling mean/std over windows
```

---

## 2. What Student Does (step by step)

### Step 1: Load and Explore Data

```python
# Student first explores the data structure
import numpy as np

# Load training data
data = np.load("calms21_task1_train.npy", allow_pickle=True).item()
dataset = data["annotator-id_0"]

print(f"Number of sequences: {len(dataset)}")

# Examine one sequence
seq_id = list(dataset.keys())[0]
seq = dataset[seq_id]

print(f"\nSequence: {seq_id}")
print(f"  Keypoints shape: {seq['keypoints'].shape}")
print(f"  Scores shape: {seq['scores'].shape}")
print(f"  Annotations shape: {seq['annotations'].shape}")
print(f"  Duration: {len(seq['annotations']) / 30:.1f} seconds")

# Check class distribution
from collections import Counter
all_labels = np.concatenate([s["annotations"] for s in dataset.values()])
counts = Counter(all_labels)
total = len(all_labels)

print(f"\nClass distribution:")
for cls, name in [(0, "attack"), (1, "investigation"), (2, "mount"), (3, "other")]:
    print(f"  {name}: {counts[cls]:,} ({100*counts[cls]/total:.2f}%)")
```

**Expected output:**
```
Number of sequences: 70

Sequence: sequence_001
  Keypoints shape: (12543, 2, 2, 7)
  Scores shape: (12543, 2, 7)
  Annotations shape: (12543,)
  Duration: 418.1 seconds

Class distribution:
  attack: 14,013 (2.76%)
  investigation: 146,636 (28.88%)
  mount: 28,623 (5.64%)
  other: 318,466 (62.72%)
```

### Step 2: Compute Inter-Animal Distance Features

```python
def compute_interanimal_features(keypoints):
    """
    Compute distance-based features between the two mice.

    Args:
        keypoints: (n_frames, 2, 2, 7) array

    Returns:
        features: dict of (n_frames,) arrays
    """
    features = {}

    # 1. Centroid-to-centroid distance
    centroids = keypoints.mean(axis=3)  # (n_frames, 2, 2)
    centroid_diff = centroids[:, 0, :] - centroids[:, 1, :]
    features["centroid_distance"] = np.sqrt((centroid_diff ** 2).sum(axis=1))

    # 2. Nose-to-nose distance (social investigation)
    nose_diff = keypoints[:, 0, :, 0] - keypoints[:, 1, :, 0]
    features["nose_to_nose"] = np.sqrt((nose_diff ** 2).sum(axis=1))

    # 3. Resident nose to intruder tail (often precedes mount/attack)
    res_nose = keypoints[:, 0, :, 0]   # resident nose
    int_tail = keypoints[:, 1, :, 6]   # intruder tail
    diff = res_nose - int_tail
    features["res_nose_to_int_tail"] = np.sqrt((diff ** 2).sum(axis=1))

    # 4. Intruder nose to resident tail
    int_nose = keypoints[:, 1, :, 0]
    res_tail = keypoints[:, 0, :, 6]
    diff = int_nose - res_tail
    features["int_nose_to_res_tail"] = np.sqrt((diff ** 2).sum(axis=1))

    # 5. Minimum keypoint distance (closest body parts)
    n_frames = keypoints.shape[0]
    all_kps_res = keypoints[:, 0, :, :]  # (n_frames, 2, 7)
    all_kps_int = keypoints[:, 1, :, :]  # (n_frames, 2, 7)

    min_dists = np.zeros(n_frames)
    for i in range(n_frames):
        # compute all pairwise distances between keypoints
        dists = np.sqrt(
            ((all_kps_res[i, :, :, np.newaxis] -
              all_kps_int[i, :, np.newaxis, :]) ** 2).sum(axis=0)
        )  # (7, 7)
        min_dists[i] = dists.min()
    features["min_keypoint_distance"] = min_dists

    return features
```

### Step 3: Compute Velocity Features

```python
def compute_velocity_features(keypoints, dt=1/30):
    """
    Compute velocity and acceleration features.

    Args:
        keypoints: (n_frames, 2, 2, 7) array
        dt: time step (1/30 for 30 Hz)

    Returns:
        features: dict of (n_frames,) arrays
    """
    features = {}

    # Centroid positions
    centroids = keypoints.mean(axis=3)  # (n_frames, 2, 2)

    # Velocity (first derivative)
    velocity = np.zeros_like(centroids)
    velocity[1:] = np.diff(centroids, axis=0) / dt

    # Speed (magnitude of velocity)
    speed = np.sqrt((velocity ** 2).sum(axis=2))  # (n_frames, 2)
    features["resident_speed"] = speed[:, 0]
    features["intruder_speed"] = speed[:, 1]
    features["speed_ratio"] = speed[:, 0] / (speed[:, 1] + 1e-6)

    # Acceleration (second derivative)
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = np.diff(velocity, axis=0) / dt
    accel_mag = np.sqrt((acceleration ** 2).sum(axis=2))
    features["resident_acceleration"] = accel_mag[:, 0]
    features["intruder_acceleration"] = accel_mag[:, 1]

    # Relative velocity (approaching vs separating)
    centroid_diff = centroids[:, 0, :] - centroids[:, 1, :]
    distance = np.sqrt((centroid_diff ** 2).sum(axis=1))
    # rate of change of distance (negative = approaching)
    features["approach_rate"] = np.zeros(len(distance))
    features["approach_rate"][1:] = np.diff(distance) / dt

    return features
```

### Step 4: Handle Class Imbalance

```python
def prepare_data_with_balancing(dataset, balance_method="oversample"):
    """
    Prepare training data with class imbalance handling.

    Args:
        dataset: dict of sequences
        balance_method: "oversample", "undersample", or "class_weight"

    Returns:
        X: feature matrix
        y: labels
        class_weights: dict for sklearn (if method="class_weight")
    """
    # Extract all features and labels
    X_list, y_list = [], []

    for seq_id, seq in dataset.items():
        kps = seq["keypoints"]
        labels = seq["annotations"]

        # Compute features
        dist_feats = compute_interanimal_features(kps)
        vel_feats = compute_velocity_features(kps)

        # Combine into feature matrix
        feat_arrays = list(dist_feats.values()) + list(vel_feats.values())
        X_seq = np.column_stack(feat_arrays)

        X_list.append(X_seq)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    if balance_method == "class_weight":
        # Compute weights inversely proportional to frequency
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.array([0, 1, 2, 3])
        weights = compute_class_weight("balanced", classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        return X, y, class_weights

    elif balance_method == "oversample":
        from imblearn.over_sampling import SMOTE
        # Only oversample the behavior classes, not "other"
        mask = y != 3
        X_behavior = X[mask]
        y_behavior = y[mask]

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_behavior, y_behavior)

        # Add back "other" class
        X_other = X[~mask]
        y_other = y[~mask]

        # Subsample "other" to match resampled size
        n_keep = len(X_resampled) // 3
        idx = np.random.choice(len(X_other), size=min(n_keep, len(X_other)), replace=False)

        X_final = np.vstack([X_resampled, X_other[idx]])
        y_final = np.concatenate([y_resampled, y_other[idx]])

        return X_final, y_final, None

    else:  # undersample
        from collections import Counter
        counts = Counter(y)
        min_count = min(counts[0], counts[1], counts[2])  # exclude "other"

        X_balanced, y_balanced = [], []
        for cls in [0, 1, 2, 3]:
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]

            if cls == 3:
                n_sample = min_count * 2  # keep some "other"
            else:
                n_sample = min_count

            idx = np.random.choice(len(X_cls), size=min(n_sample, len(X_cls)), replace=False)
            X_balanced.append(X_cls[idx])
            y_balanced.append(y_cls[idx])

        return np.vstack(X_balanced), np.concatenate(y_balanced), None
```

### Step 5: Train Classifier

```python
def train_classifier(X_train, y_train, class_weights=None, model_type="rf"):
    """
    Train a behavior classifier.

    Args:
        X_train: feature matrix
        y_train: labels
        class_weights: optional dict for weighted training
        model_type: "rf" (Random Forest), "svm", or "mlp"

    Returns:
        trained classifier
    """
    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=5,
            class_weight=class_weights or "balanced",
            n_jobs=-1,
            random_state=42
        )

    elif model_type == "svm":
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight=class_weights or "balanced",
                random_state=42
            ))
        ])

    elif model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                alpha=0.001,
                batch_size=256,
                learning_rate="adaptive",
                max_iter=200,
                random_state=42
            ))
        ])

    clf.fit(X_train, y_train)
    return clf
```

### Step 6: Generate Predictions

```python
def generate_predictions(clf, test_data):
    """
    Generate predictions for all test sequences.

    Args:
        clf: trained classifier
        test_data: dict of test sequences

    Returns:
        predictions: dict mapping seq_id -> list of predicted class ids
    """
    predictions = {}

    for seq_id, seq in test_data.items():
        kps = seq["keypoints"]

        # Extract same features as training
        dist_feats = compute_interanimal_features(kps)
        vel_feats = compute_velocity_features(kps)

        feat_arrays = list(dist_feats.values()) + list(vel_feats.values())
        X_test = np.column_stack(feat_arrays)

        # Handle NaN/inf
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

        # Predict
        preds = clf.predict(X_test)
        predictions[seq_id] = preds.tolist()

    return predictions


# ============== COMPLETE SOLUTION ==============

def classify_behavior(train_data, test_data):
    """Complete implementation."""

    # 1. Prepare training data with class balancing
    X_train, y_train, class_weights = prepare_data_with_balancing(
        train_data, balance_method="class_weight"
    )

    # 2. Train classifier
    clf = train_classifier(X_train, y_train, class_weights, model_type="rf")

    # 3. Generate predictions
    predictions = generate_predictions(clf, test_data)

    return predictions
```

---

## 3. Submission Format

### 3.1 Solution File

Student submits `solution.py` with the `classify_behavior()` function and returns a predictions dictionary:

```python
# solution.py
def classify_behavior(train_data, test_data):
    """Student's implementation"""
    # ... their code ...
    return predictions  # dict: {seq_id: [class_ids]}
```

### 3.2 Predictions Format

```python
predictions = {
    "sequence_001": [3, 3, 1, 1, 1, 0, 0, 3, ...],  # len = n_frames
    "sequence_002": [3, 3, 3, 2, 2, 2, 3, ...],
    ...
}

# Save as .npy
np.save("submission.npy", predictions, allow_pickle=True)
```

### 3.3 Validation Checks

```python
# validation.py - runs immediately on submission

def validate_submission(submission, test_data, vocab):
    """Validate submission format before full evaluation."""

    # Check 1: submission is dict
    assert isinstance(submission, dict), "submission must be dict"

    # Check 2: all test sequences present
    for seq_id in test_data:
        assert seq_id in submission, f"missing sequence: {seq_id}"

    # Check 3: no extra sequences
    for seq_id in submission:
        assert seq_id in test_data, f"unknown sequence: {seq_id}"

    # Check 4: correct lengths
    for seq_id, preds in submission.items():
        expected_len = len(test_data[seq_id]["keypoints"])
        assert len(preds) == expected_len, \
            f"length mismatch for {seq_id}: got {len(preds)}, expected {expected_len}"

    # Check 5: valid class ids
    valid_classes = set(vocab.values())  # {0, 1, 2, 3}
    for seq_id, preds in submission.items():
        invalid = [p for p in preds if p not in valid_classes]
        assert not invalid, f"invalid class ids in {seq_id}: {set(invalid)}"

    return True
```

---

## 4. Evaluation Pipeline (full code)

### 4.1 Evaluation Orchestrator

```python
# evaluator.py

import numpy as np
import time
import tracemalloc
from dataclasses import dataclass
from typing import Dict, List, Any
from sklearn.metrics import f1_score, precision_score, classification_report

@dataclass
class EvaluationResult:
    passed: bool
    score: float  # 0-100
    f1_macro: float
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    grade: str


def evaluate_submission(
    solution_path: str,
    train_path: str = "calms21_task1_train.npy",
    test_path: str = "calms21_task1_test.npy",
) -> EvaluationResult:
    """
    Full CalMS21 evaluation pipeline.

    1. Load student solution
    2. Run on train/test data
    3. Compute macro F1 (excluding "other")
    4. Run anti-gaming checks
    5. Compute final score
    """

    vocab = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}

    metrics = {}
    errors = []
    warnings = []

    # =========== LOAD DATA ===========

    train_data = np.load(train_path, allow_pickle=True).item()["annotator-id_0"]
    test_data = np.load(test_path, allow_pickle=True).item()["annotator-id_0"]

    # For evaluation, we need ground truth test annotations
    # (stored separately from student-visible test data)
    test_gt = np.load("calms21_task1_test_gt.npy", allow_pickle=True).item()["annotator-id_0"]

    # =========== IMPORT STUDENT SOLUTION ===========

    import importlib.util
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    solution = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(solution)
    except Exception as e:
        errors.append(f"Failed to import solution: {str(e)}")
        return EvaluationResult(
            passed=False, score=0, f1_macro=0, metrics={},
            errors=errors, warnings=[], grade="FAIL"
        )

    # =========== RUN STUDENT SOLUTION ===========

    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        predictions = solution.classify_behavior(train_data, test_data)
    except Exception as e:
        errors.append(f"Runtime error: {str(e)}")
        return EvaluationResult(
            passed=False, score=0, f1_macro=0, metrics={},
            errors=errors, warnings=[], grade="FAIL"
        )

    elapsed_time = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics["processing_time_sec"] = elapsed_time
    metrics["peak_memory_gb"] = peak_mem / 1e9

    # =========== VALIDATE OUTPUT ===========

    try:
        validate_submission(predictions, test_data, vocab)
    except AssertionError as e:
        errors.append(f"Validation failed: {str(e)}")
        return EvaluationResult(
            passed=False, score=0, f1_macro=0, metrics=metrics,
            errors=errors, warnings=[], grade="FAIL"
        )

    # =========== COMPUTE F1 SCORE ===========

    all_preds = []
    all_labels = []

    for seq_id in test_gt:
        preds = np.array(predictions[seq_id])
        labels = test_gt[seq_id]["annotations"]
        all_preds.extend(preds)
        all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Exclude "other" class from evaluation
    mask = all_labels != vocab["other"]
    preds_filtered = all_preds[mask]
    labels_filtered = all_labels[mask]

    # Primary metric: macro F1 on classes 0, 1, 2
    f1_macro = f1_score(
        labels_filtered,
        preds_filtered,
        average="macro",
        labels=[0, 1, 2]
    )

    # Secondary metric: precision
    precision_macro = precision_score(
        labels_filtered,
        preds_filtered,
        average="macro",
        labels=[0, 1, 2]
    )

    # Per-class metrics
    report = classification_report(
        labels_filtered,
        preds_filtered,
        labels=[0, 1, 2],
        target_names=["attack", "investigation", "mount"],
        output_dict=True,
        zero_division=0
    )

    metrics["f1_macro"] = f1_macro
    metrics["precision_macro"] = precision_macro
    metrics["f1_attack"] = report["attack"]["f1-score"]
    metrics["f1_investigation"] = report["investigation"]["f1-score"]
    metrics["f1_mount"] = report["mount"]["f1-score"]
    metrics["n_frames_evaluated"] = mask.sum()
    metrics["n_frames_total"] = len(all_labels)

    # =========== ANTI-GAMING CHECKS ===========

    antigaming = run_antigaming_checks(solution, predictions, test_data)

    if antigaming["constant_predictions"]:
        errors.append("FAIL: Constant predictions detected (all same class)")
    if antigaming["random_guessing"]:
        warnings.append("WARNING: Predictions appear to be random guessing")
    if antigaming["no_attack_predictions"]:
        warnings.append("WARNING: No attack class (0) predictions made")
    if antigaming["memorization_suspected"]:
        errors.append("FAIL: Memorization of test data suspected")

    # =========== COMPUTE FINAL SCORE ===========

    score, grade = compute_final_score(metrics, errors)
    passed = score >= 50 and len(errors) == 0

    return EvaluationResult(
        passed=passed,
        score=score,
        f1_macro=f1_macro,
        metrics=metrics,
        errors=errors,
        warnings=warnings,
        grade=grade
    )
```

### 4.2 Anti-Gaming Checks

```python
def run_antigaming_checks(
    solution_module,
    predictions: Dict,
    test_data: Dict
) -> Dict[str, bool]:
    """
    Detect cheating strategies.
    """
    import inspect

    results = {
        "constant_predictions": False,
        "random_guessing": False,
        "no_attack_predictions": False,
        "memorization_suspected": False,
    }

    # Collect all predictions
    all_preds = np.concatenate([predictions[k] for k in predictions])

    # Check 1: Constant predictions (all same class)
    unique_preds = np.unique(all_preds)
    if len(unique_preds) == 1:
        results["constant_predictions"] = True

    # Check 2: Missing attack predictions
    if 0 not in unique_preds:
        results["no_attack_predictions"] = True

    # Check 3: Random guessing detection
    # If class distribution is ~uniform, might be random
    from collections import Counter
    pred_counts = Counter(all_preds)
    total = len(all_preds)

    # Expected for random: ~25% each class
    expected = total / 4
    chi_sq = sum((pred_counts.get(c, 0) - expected) ** 2 / expected for c in [0, 1, 2, 3])
    # If chi-squared is very low, predictions might be random
    if chi_sq < 10 and len(unique_preds) == 4:
        results["random_guessing"] = True

    # Check 4: Memorization detection
    # Run on slightly perturbed data and check if predictions change
    try:
        source = inspect.getsource(solution_module.classify_behavior)

        # Check for hardcoded sequence IDs
        if any(seq_id in source for seq_id in test_data.keys()):
            results["memorization_suspected"] = True

        # Check for suspiciously specific numbers
        import re
        numbers = re.findall(r'\b\d{5,}\b', source)  # 5+ digit numbers
        if len(numbers) > 10:
            results["memorization_suspected"] = True

    except Exception:
        pass

    return results
```

### 4.3 Score Computation

```python
def compute_final_score(metrics: Dict, errors: List[str]) -> tuple:
    """
    Compute final score (0-100) and grade.

    Score based on:
    - F1 macro score (primary)
    - Processing time penalty
    - Memory penalty
    """

    f1 = metrics.get("f1_macro", 0)

    # Base score from F1 (0-100 scale)
    # F1 of 0.50 -> 50 points, 0.79 -> 79 points, etc.
    base_score = f1 * 100

    # Time penalty: >60s subtracts points
    time_sec = metrics.get("processing_time_sec", 0)
    if time_sec > 300:  # >5 min
        base_score -= 10
    elif time_sec > 120:  # >2 min
        base_score -= 5

    # Memory penalty: >4GB subtracts points
    mem_gb = metrics.get("peak_memory_gb", 0)
    if mem_gb > 8:
        base_score -= 10
    elif mem_gb > 4:
        base_score -= 5

    # Error penalties
    base_score -= len(errors) * 20

    score = max(0, min(100, base_score))

    # Determine grade
    if errors:
        grade = "FAIL"
    elif f1 >= 0.86:
        grade = "OUTSTANDING"
    elif f1 >= 0.79:
        grade = "EXCELLENT"
    elif f1 >= 0.70:
        grade = "GOOD"
    elif f1 >= 0.50:
        grade = "PASS"
    else:
        grade = "FAIL"

    return score, grade
```

---

## 5. Reference Solution

### 5.1 Optimal Implementation (winning solution features)

```python
# reference_solution.py
"""
Reference implementation with MABe 2021 winning solution features:
- Pairwise distances between all keypoints
- Egocentric coordinates (relative to resident heading)
- Temporal context (past + future frames)
- Balanced training with SMOTE
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


def compute_pairwise_distances(keypoints):
    """
    Compute all pairwise distances between keypoints (both mice).

    Args:
        keypoints: (n_frames, 2, 2, 7)

    Returns:
        distances: (n_frames, 91) - upper triangle of 14x14 matrix
    """
    n_frames = keypoints.shape[0]

    # Reshape to (n_frames, 14, 2): all 14 keypoints (7 per mouse)
    all_kps = keypoints.transpose(0, 1, 3, 2).reshape(n_frames, 14, 2)

    # Pairwise distance matrix
    diff = all_kps[:, :, np.newaxis, :] - all_kps[:, np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))  # (n_frames, 14, 14)

    # Extract upper triangle (91 unique distances)
    triu_idx = np.triu_indices(14, k=1)
    return dists[:, triu_idx[0], triu_idx[1]]


def compute_egocentric_coords(keypoints):
    """
    Transform keypoints to egocentric coordinates (relative to resident).

    Resident is centered at origin, facing along positive x-axis.
    """
    n_frames = keypoints.shape[0]

    # Resident body angle (from tail to neck)
    neck = keypoints[:, 0, :, 3]   # (n_frames, 2)
    tail = keypoints[:, 0, :, 6]
    direction = neck - tail
    angle = np.arctan2(direction[:, 1], direction[:, 0])  # (n_frames,)

    # Resident centroid
    res_centroid = keypoints[:, 0, :, :].mean(axis=2)  # (n_frames, 2)

    # Rotation matrix (rotate by -angle to align resident with x-axis)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    # Transform all keypoints
    ego_kps = np.zeros_like(keypoints)

    for mouse in [0, 1]:
        for kp in range(7):
            # Translate (center on resident)
            x = keypoints[:, mouse, 0, kp] - res_centroid[:, 0]
            y = keypoints[:, mouse, 1, kp] - res_centroid[:, 1]

            # Rotate
            ego_kps[:, mouse, 0, kp] = cos_a * x - sin_a * y
            ego_kps[:, mouse, 1, kp] = sin_a * x + cos_a * y

    return ego_kps


def compute_temporal_context(features, window_past=100, window_future=100, skip=2):
    """
    Add temporal context by stacking features from past and future frames.

    Args:
        features: (n_frames, n_features)
        window_past: frames to look back
        window_future: frames to look ahead
        skip: subsample factor

    Returns:
        context_features: (n_frames, n_features * n_context_frames)
    """
    n_frames, n_features = features.shape

    # Indices of context frames (relative to current)
    past_idx = list(range(-window_past, 0, skip))
    future_idx = list(range(skip, window_future + 1, skip))
    context_idx = past_idx + [0] + future_idx

    # Stack features from context frames
    context_features = np.zeros((n_frames, len(context_idx) * n_features))

    for i, offset in enumerate(context_idx):
        # Shift features by offset (pad with zeros at boundaries)
        if offset < 0:
            src_start, src_end = -offset, n_frames
            dst_start, dst_end = 0, n_frames + offset
        elif offset > 0:
            src_start, src_end = 0, n_frames - offset
            dst_start, dst_end = offset, n_frames
        else:
            src_start, src_end = 0, n_frames
            dst_start, dst_end = 0, n_frames

        context_features[dst_start:dst_end, i*n_features:(i+1)*n_features] = \
            features[src_start:src_end]

    return context_features


def extract_features_reference(keypoints):
    """
    Extract full feature set (reference implementation).
    """
    # 1. Pairwise distances (91 features)
    pairwise = compute_pairwise_distances(keypoints)

    # 2. Egocentric coordinates (flattened: 28 features)
    ego_kps = compute_egocentric_coords(keypoints)
    ego_flat = ego_kps.reshape(len(keypoints), -1)

    # 3. Velocities
    centroids = keypoints.mean(axis=3)
    velocity = np.zeros_like(centroids)
    velocity[1:] = np.diff(centroids, axis=0) * 30  # scale to pixels/second
    speed = np.sqrt((velocity ** 2).sum(axis=2))  # (n_frames, 2)

    # 4. Body angles
    neck = keypoints[:, :, :, 3]
    tail = keypoints[:, :, :, 6]
    direction = neck - tail
    angles = np.arctan2(direction[:, :, 1], direction[:, :, 0])

    # 5. Relative angle between mice
    angle_diff = angles[:, 0] - angles[:, 1]
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

    # Combine base features
    base_features = np.column_stack([
        pairwise,           # 91
        ego_flat,           # 28
        speed,              # 2
        angles,             # 2
        angle_diff[:, np.newaxis],  # 1
    ])  # Total: 124 features

    # 6. Add temporal context
    features = compute_temporal_context(
        base_features,
        window_past=60,    # 2 seconds
        window_future=60,
        skip=3
    )

    return features


def classify_behavior_reference(train_data, test_data):
    """
    Reference implementation achieving ~0.82-0.85 F1.
    """

    # Extract features for all training sequences
    X_train_list, y_train_list = [], []

    for seq_id, seq in train_data.items():
        features = extract_features_reference(seq["keypoints"])
        X_train_list.append(features)
        y_train_list.append(seq["annotations"])

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # PCA for dimensionality reduction (optional but speeds up training)
    pca = PCA(n_components=0.95)  # keep 95% variance
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Balance classes with SMOTE (only for behavior classes)
    mask = y_train != 3
    X_behavior = X_train_pca[mask]
    y_behavior = y_train[mask]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_behavior, y_behavior)

    # Subsample "other" class
    X_other = X_train_pca[~mask]
    y_other = y_train[~mask]
    n_keep = len(X_resampled) // 2
    idx = np.random.choice(len(X_other), size=min(n_keep, len(X_other)), replace=False)

    X_final = np.vstack([X_resampled, X_other[idx]])
    y_final = np.concatenate([y_resampled, y_other[idx]])

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_final, y_final)

    # Generate predictions
    predictions = {}

    for seq_id, seq in test_data.items():
        features = extract_features_reference(seq["keypoints"])
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        preds = clf.predict(features_pca)
        predictions[seq_id] = preds.tolist()

    return predictions
```

---

## 6. Feedback Examples

### 6.1 PASSED Example

```
═══════════════════════════════════════════════════════════════
                    EVALUATION RESULTS
═══════════════════════════════════════════════════════════════

Status: ✅ PASSED

Score: 76/100
Grade: GOOD

Primary Metric:
  • Macro F1 (excl. "other"): 0.762

Per-Class F1 Scores:
  ├─ attack:        0.584  ⚠️ (hardest class)
  ├─ investigation: 0.841
  └─ mount:         0.862

Performance:
  • Processing time: 42.3 seconds
  • Peak memory: 2.1 GB
  • Frames evaluated: 98,234 / 262,107

═══════════════════════════════════════════════════════════════

Suggestions for improvement:

1. ATTACK CLASS: Your attack F1 is 0.584, which is the main bottleneck.
   Attack is rare (2.8%) so the model likely under-predicts it.
   Try:
   - Oversample attack frames specifically
   - Use class_weight="balanced" in your classifier
   - Add more features that distinguish attack (high velocity, close distance)

2. TEMPORAL CONTEXT: Adding features from neighboring frames can improve
   classification since behaviors span multiple frames (0.5-5 seconds).
   Try rolling mean/std over windows of 10, 30, 90 frames.

3. FEATURE ENGINEERING: Consider adding:
   - Pairwise distances between all keypoints
   - Egocentric coordinates (relative to resident heading)
   - Body angle difference between mice

Baseline: 0.793 | Your score: 0.762 | SOTA: 0.864

═══════════════════════════════════════════════════════════════
```

### 6.2 FAILED Example

```
═══════════════════════════════════════════════════════════════
                    EVALUATION RESULTS
═══════════════════════════════════════════════════════════════

Status: ❌ FAILED

Score: 38/100
Grade: FAIL

Primary Metric:
  • Macro F1 (excl. "other"): 0.384

Errors:
  ✗ F1 score below passing threshold (0.384 < 0.50)

Per-Class F1 Scores:
  ├─ attack:        0.021  ⚠️ CRITICAL
  ├─ investigation: 0.712
  └─ mount:         0.419

Warnings:
  ⚠ Very few attack predictions made (0.3% vs expected ~2.8%)
  ⚠ Model appears to be biased toward "other" class

Performance:
  • Processing time: 12.1 seconds
  • Peak memory: 1.2 GB

═══════════════════════════════════════════════════════════════

Debug hints:

1. CRITICAL - ATTACK CLASS: Your attack F1 is nearly zero (0.021).
   This suggests your model is not predicting attack at all.

   Common causes:
   - No class balancing: attack is only 2.8% of data
   - Features don't capture attack behavior (high speed, close contact)
   - Threshold tuning needed if using probability predictions

   Quick fix: Add `class_weight="balanced"` to your classifier

2. CHECK YOUR CLASS PREDICTIONS:
   ```python
   from collections import Counter
   print(Counter(all_predictions))
   # Should see: {0: ~2.8%, 1: ~29%, 2: ~6%, 3: ~62%}
   ```

3. MOUNT CLASS: F1 of 0.419 is also low.
   Mount and attack share similar pose features (close contact).
   Try adding velocity features - attack involves faster movement.

4. VERIFY FEATURE EXTRACTION:
   - Check for NaN values in features
   - Ensure consistent feature order between train and test
   - Normalize features if using SVM/MLP

Minimum passing F1: 0.50 | Your score: 0.384

═══════════════════════════════════════════════════════════════
```

---

## 7. Visual Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     STUDENT WORKFLOW                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. LOAD & EXPLORE DATA                                     │
│     - Load calms21_task1_train.npy                         │
│     - Examine keypoints shape: (n_frames, 2, 2, 7)         │
│     - Check class distribution (attack is only 2.8%!)      │
│     - Visualize example frames                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. FEATURE ENGINEERING                                     │
│     ┌─────────────────────────────────────────────────────┐ │
│     │ Inter-Animal Distances                              │ │
│     │   • centroid-to-centroid                           │ │
│     │   • nose-to-nose (investigation)                    │ │
│     │   • nose-to-tail (mount/attack precursor)          │ │
│     │   • minimum keypoint distance                       │ │
│     └─────────────────────────────────────────────────────┘ │
│     ┌─────────────────────────────────────────────────────┐ │
│     │ Velocity & Motion                                   │ │
│     │   • centroid speed (both mice)                      │ │
│     │   • acceleration                                    │ │
│     │   • approach/separate rate                          │ │
│     └─────────────────────────────────────────────────────┘ │
│     ┌─────────────────────────────────────────────────────┐ │
│     │ Body Orientation                                    │ │
│     │   • body angle (tail → neck)                        │ │
│     │   • relative angle between mice                     │ │
│     │   • facing direction                                │ │
│     └─────────────────────────────────────────────────────┘ │
│     ┌─────────────────────────────────────────────────────┐ │
│     │ Temporal Context (optional but recommended)         │ │
│     │   • rolling mean/std over windows                   │ │
│     │   • features from past/future frames                │ │
│     └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. HANDLE CLASS IMBALANCE                                  │
│                                                             │
│     Class Distribution:                                     │
│     ┌─────────────────────────────────────────────┐        │
│     │ attack        ████  2.8%                    │        │
│     │ investigation ████████████████████ 28.9%    │        │
│     │ mount         ██████  5.6%                  │        │
│     │ other         ████████████████████████ 62.7%│        │
│     └─────────────────────────────────────────────┘        │
│                                                             │
│     Options:                                                │
│     • class_weight="balanced" in classifier                │
│     • SMOTE oversampling of minority classes               │
│     • Undersample "other" class                            │
│     • Focal loss (if using neural network)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. TRAIN CLASSIFIER                                        │
│                                                             │
│     Recommended models:                                     │
│     • Random Forest (fast, robust baseline)                │
│     • Gradient Boosting (XGBoost, LightGBM)                │
│     • MLP (if using temporal context)                      │
│                                                             │
│     Key hyperparameters:                                    │
│     • n_estimators: 100-300                                │
│     • max_depth: 15-25                                     │
│     • class_weight: "balanced"                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. GENERATE PREDICTIONS                                    │
│     - Extract same features for test data                  │
│     - Apply trained classifier                              │
│     - Return dict: {seq_id: [class_ids]}                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  6. SUBMIT solution.py + submission.npy                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  A. VALIDATION                                              │
│     - Check all test sequences present                      │
│     - Check prediction lengths match frame counts          │
│     - Check class IDs are valid {0, 1, 2, 3}               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  B. COMPUTE MACRO F1                                        │
│                                                             │
│     ┌─────────────────────────────────────────────────────┐ │
│     │ 1. Concatenate all predictions and labels           │ │
│     │ 2. EXCLUDE "other" class (id=3) from evaluation    │ │
│     │ 3. Compute F1 for each class: attack, invest, mount │ │
│     │ 4. Average → Macro F1 (unweighted mean)            │ │
│     └─────────────────────────────────────────────────────┘ │
│                                                             │
│     f1_score(labels, preds, average="macro", labels=[0,1,2])│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  C. ANTI-GAMING CHECKS                                      │
│     - No constant predictions (all same class)             │
│     - No memorization of test sequences                    │
│     - Attack class actually predicted (not zero)           │
│     - Predictions not purely random                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  D. COMPUTE FINAL SCORE & GRADE                             │
│                                                             │
│     Score = F1 × 100 - time_penalty - memory_penalty       │
│                                                             │
│     Grading:                                                │
│     ┌───────────────┬──────────┬─────────────────────────┐ │
│     │ Grade         │ F1 Range │ Interpretation          │ │
│     ├───────────────┼──────────┼─────────────────────────┤ │
│     │ FAIL          │ < 0.50   │ Below basic threshold   │ │
│     │ PASS          │ 0.50-0.70│ Basic features work     │ │
│     │ GOOD          │ 0.70-0.79│ Competitive with baseline│
│     │ EXCELLENT     │ 0.79-0.86│ Near state-of-the-art   │ │
│     │ OUTSTANDING   │ > 0.86   │ Challenge-winning level │ │
│     └───────────────┴──────────┴─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  E. RETURN FEEDBACK                                         │
│     - Pass/Fail status                                      │
│     - Score and grade                                       │
│     - Per-class F1 breakdown                               │
│     - Improvement suggestions                               │
│     - Debug hints for failures                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Key Technical Notes

### 8.1 Data Split Protocol

**CRITICAL:** Train and test sets are split by **animal identity**.

- Animals in test set are **different individuals** from training
- Same recording setup, date range, and annotator
- Prevents identity memorization and temporal leakage
- Students should NOT do frame-level validation splits within sequences

### 8.2 "Other" Class Handling

The "other" class (~62.7% of data) is **excluded from F1 computation** but should still be predicted correctly:

```python
# Correct evaluation
mask = labels != 3  # exclude "other"
f1 = f1_score(labels[mask], preds[mask], average="macro", labels=[0, 1, 2])
```

### 8.3 Common Mistakes

1. **No class balancing** → Model predicts "other" for everything
2. **Frame-level train/test split** → Temporal leakage, inflated scores
3. **Forgetting to exclude "other" in evaluation** → Wrong metric
4. **NaN values in features** → Classifier fails silently
5. **Different feature order train vs test** → Random predictions

---

*spec version: 1.0*
*last updated: 2026-02-02*
