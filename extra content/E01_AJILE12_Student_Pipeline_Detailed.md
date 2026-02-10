# E01: AJILE12 – Detailed Student Pipeline

> полная спецификация пути студента от старта до оценки
> task: E01-T1 (load + merge ECoG with pose data)

---

## 1. что видит студент на старте

### 1.1 task description (в интерфейсе)

```
TASK: Multi-Modal Neural Data Alignment

You have access to the AJILE12 dataset via DANDI API. This dataset contains:
- ECoG neural recordings (500 Hz, ~64 channels)
- Pose tracking data (30 Hz, 9 keypoints)
- Movement event annotations

Your goal: Create a function that loads and merges these modalities
into a unified DataFrame with aligned timestamps.

Requirements:
1. Load ECoG and pose data for specified time windows
2. Align to common 30 Hz timebase (pose rate)
3. Downsample ECoG appropriately
4. Exclude bad electrodes
5. Handle missing data gracefully

Time limit: 4 hours
Memory limit: 4 GB
```

### 1.2 starter code (что получает студент)

```python
# starter_code.py
"""
AJILE12 Data Alignment Task
===========================

Complete the `load_and_merge` function below.

You have access to:
- DANDI API client (pre-authenticated)
- remfile for streaming NWB files
- pynwb for NWB parsing
- numpy, pandas, scipy

DO NOT download entire files locally. Use streaming access.
"""

from dandi.dandiapi import DandiAPIClient
import remfile
import h5py
from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd
from typing import List, Tuple

# Constants
DANDISET_ID = "000055"
ECOG_RATE = 500.0  # Hz
POSE_RATE = 30.0   # Hz

def get_nwb_url(subject_id: str, session_id: int) -> str:
    """Get streaming URL for NWB file. Already implemented."""
    filename = f"sub-{subject_id}/sub-{subject_id}_ses-{session_id}_behavior+ecephys.nwb"
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(DANDISET_ID)
        asset = dandiset.get_asset_by_path(filename)
        return asset.get_content_url(follow_redirects=1, strip_query=True)


def load_and_merge(
    subject_id: str,
    session_ids: List[int],
    time_windows: List[Tuple[float, float]],
    target_keypoints: List[str],
) -> pd.DataFrame:
    """
    Load and merge ECoG with pose data.

    Args:
        subject_id: Subject identifier (e.g., "01")
        session_ids: List of session numbers to load
        time_windows: List of (start_sec, end_sec) tuples
        target_keypoints: Which pose keypoints to include

    Returns:
        DataFrame with columns:
        - timestamp: float64, common timebase in seconds
        - session_id: int
        - ecog_{i}: float32, one column per good electrode
        - {keypoint}_x, {keypoint}_y: float64, pose coordinates
        - is_movement: bool, from reaches interval table
        - behavior_state: str, from epochs table

    Notes:
        - Output should be at 30 Hz (pose rate)
        - ECoG should be downsampled by averaging
        - Exclude electrodes where 'good' == False
        - Handle missing pose data with NaN
    """
    # TODO: Implement this function
    raise NotImplementedError("Complete this function")


# ============== HELPER FUNCTIONS (you may use these) ==============

def open_nwb_streaming(url: str):
    """Open NWB file for streaming access."""
    rem_file = remfile.File(url)
    h5_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5_file, load_namespaces=True)
    return io.read(), h5_file, io


# ============== TEST YOUR SOLUTION ==============

if __name__ == "__main__":
    # Basic test case
    result = load_and_merge(
        subject_id="01",
        session_ids=[1],
        time_windows=[(0, 60)],  # first minute only
        target_keypoints=["left_wrist", "right_wrist"],
    )

    print(f"Result shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    print(f"Memory usage: {result.memory_usage(deep=True).sum() / 1e6:.2f} MB")
```

### 1.3 reference documentation (доступно студенту)

```markdown
## AJILE12 Data Structure

### ECoG Data
- Location: `/acquisition/ElectricalSeries`
- Shape: [n_samples, n_channels]
- Rate: 500 Hz
- Dtype: float32

### Pose Data
- Location: `/processing/behavior/Position/SpatialSeries`
- Shape: [n_frames, 9, 2] or separate series per keypoint
- Rate: 30 Hz (video frame rate)
- Keypoints: nose, left_ear, right_ear, left_shoulder, right_shoulder,
             left_elbow, right_elbow, left_wrist, right_wrist

### Electrode Table
- Location: `/general/extracellular_ephys/electrodes`
- Key columns:
  - `good`: bool, whether electrode passes quality check
  - `x, y, z`: MNI coordinates
  - `location`: anatomical region

### Movement Events
- Location: `/intervals/reaches`
- Columns: start_time, stop_time, reach_magnitude, etc.

### Behavioral Epochs
- Location: `/intervals/epochs`
- Columns: start_time, stop_time, tags (behavior labels)
```

---

## 2. что делает студент (пошагово)

### step 1: понять структуру данных

студент использует tools чтобы исследовать NWB файл:

```python
# студент запускает это чтобы понять структуру
url = get_nwb_url("01", 1)
nwb, h5, io = open_nwb_streaming(url)

# смотрит на ECoG
ecog = nwb.acquisition["ElectricalSeries"]
print(f"ECoG shape: {ecog.data.shape}")  # [n_samples, n_channels]
print(f"ECoG rate: {ecog.rate}")  # 500.0

# смотрит на electrodes
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Good electrodes: {electrodes_df['good'].sum()}/{len(electrodes_df)}")

# смотрит на pose
position = nwb.processing["behavior"]["Position"]
print(f"Spatial series: {list(position.spatial_series.keys())}")
```

### step 2: загрузить ECoG для time window

```python
def load_ecog_slice(nwb, start_time, end_time, good_only=True):
    """Load ECoG data for time window."""
    ecog = nwb.acquisition["ElectricalSeries"]
    rate = ecog.rate

    # convert time to samples
    start_sample = int(start_time * rate)
    end_sample = int(end_time * rate)

    # get good electrode indices
    if good_only:
        electrodes_df = nwb.electrodes.to_dataframe()
        good_indices = electrodes_df[electrodes_df["good"]].index.tolist()
    else:
        good_indices = list(range(ecog.data.shape[1]))

    # streaming slice - only downloads requested data
    data = ecog.data[start_sample:end_sample, good_indices]

    # timestamps
    timestamps = np.arange(start_sample, end_sample) / rate

    return data, timestamps, good_indices
```

### step 3: загрузить pose для time window

```python
def load_pose_slice(nwb, start_time, end_time, keypoints):
    """Load pose data for time window."""
    position = nwb.processing["behavior"]["Position"]

    # handle different storage formats
    all_series = list(position.spatial_series.values())

    if len(all_series) == 1:
        # combined format: [n_frames, n_keypoints, 2]
        series = all_series[0]
        timestamps = series.timestamps[:]

        # find time window
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        data = series.data[mask]
        timestamps = timestamps[mask]

        # select keypoints (need to map names to indices)
        keypoint_names = ["nose", "left_ear", "right_ear",
                         "left_shoulder", "right_shoulder",
                         "left_elbow", "right_elbow",
                         "left_wrist", "right_wrist"]
        kp_indices = [keypoint_names.index(kp) for kp in keypoints]
        data = data[:, kp_indices, :]

    else:
        # separate series per keypoint
        series_dict = {name: series for name, series in position.spatial_series.items()}
        timestamps = series_dict[keypoints[0]].timestamps[:]

        mask = (timestamps >= start_time) & (timestamps <= end_time)
        timestamps = timestamps[mask]

        data = np.stack([
            series_dict[kp].data[mask]
            for kp in keypoints
        ], axis=1)

    return data, timestamps
```

### step 4: выровнять timestamps (ключевой шаг!)

```python
def align_to_pose_rate(ecog_data, ecog_timestamps, pose_timestamps):
    """
    Downsample ECoG to match pose timestamps.

    ECoG is 500 Hz, pose is 30 Hz.
    For each pose timestamp, average nearby ECoG samples.
    """
    aligned_ecog = np.zeros((len(pose_timestamps), ecog_data.shape[1]))

    half_window = 0.5 / POSE_RATE  # 16.67 ms

    for i, t in enumerate(pose_timestamps):
        # find ECoG samples within window
        mask = (ecog_timestamps >= t - half_window) & \
               (ecog_timestamps < t + half_window)

        if mask.sum() > 0:
            aligned_ecog[i] = ecog_data[mask].mean(axis=0)
        else:
            aligned_ecog[i] = np.nan

    return aligned_ecog
```

### step 5: добавить movement events и behavior states

```python
def add_annotations(df, nwb):
    """Add is_movement and behavior_state columns."""

    # Movement events from reaches table
    reaches = nwb.intervals["reaches"].to_dataframe()

    df["is_movement"] = False
    for _, reach in reaches.iterrows():
        mask = (df["timestamp"] >= reach["start_time"]) & \
               (df["timestamp"] <= reach["stop_time"])
        df.loc[mask, "is_movement"] = True

    # Behavior states from epochs
    epochs = nwb.intervals["epochs"].to_dataframe()

    df["behavior_state"] = "unknown"
    for _, epoch in epochs.iterrows():
        mask = (df["timestamp"] >= epoch["start_time"]) & \
               (df["timestamp"] <= epoch["stop_time"])
        df.loc[mask, "behavior_state"] = epoch["tags"][0] if epoch["tags"] else "unknown"

    return df
```

### step 6: собрать всё вместе

```python
def load_and_merge(subject_id, session_ids, time_windows, target_keypoints):
    """Full implementation."""

    all_dfs = []

    for session_id in session_ids:
        # get streaming URL
        url = get_nwb_url(subject_id, session_id)
        nwb, h5, io = open_nwb_streaming(url)

        try:
            for start_time, end_time in time_windows:
                # load ECoG
                ecog_data, ecog_ts, good_indices = load_ecog_slice(
                    nwb, start_time, end_time, good_only=True
                )

                # load pose
                pose_data, pose_ts = load_pose_slice(
                    nwb, start_time, end_time, target_keypoints
                )

                # align ECoG to pose timestamps
                aligned_ecog = align_to_pose_rate(ecog_data, ecog_ts, pose_ts)

                # build dataframe
                df = pd.DataFrame()
                df["timestamp"] = pose_ts
                df["session_id"] = session_id

                # add ECoG columns
                for i, idx in enumerate(good_indices):
                    df[f"ecog_{idx}"] = aligned_ecog[:, i].astype(np.float32)

                # add pose columns
                for i, kp in enumerate(target_keypoints):
                    df[f"{kp}_x"] = pose_data[:, i, 0]
                    df[f"{kp}_y"] = pose_data[:, i, 1]

                # add annotations
                df = add_annotations(df, nwb)

                all_dfs.append(df)

        finally:
            io.close()
            h5.close()

    return pd.concat(all_dfs, ignore_index=True)
```

---

## 3. что студент сабмитит

### 3.1 submission format

студент отправляет **файл `solution.py`** с реализованной функцией:

```python
# solution.py
def load_and_merge(subject_id, session_ids, time_windows, target_keypoints):
    """Student's implementation"""
    # ... их код ...
    return df  # pd.DataFrame
```

### 3.2 submission validation (первичная проверка)

```python
# validation.py - запускается сразу при сабмите

def validate_submission(solution_module):
    """Quick validation before full evaluation."""

    from solution import load_and_merge

    # test 1: function exists and callable
    assert callable(load_and_merge), "load_and_merge must be a function"

    # test 2: basic invocation doesn't crash
    result = load_and_merge(
        subject_id="01",
        session_ids=[1],
        time_windows=[(0, 10)],  # 10 seconds only
        target_keypoints=["left_wrist"],
    )

    # test 3: returns DataFrame
    assert isinstance(result, pd.DataFrame), "Must return pandas DataFrame"

    # test 4: required columns exist
    required_cols = ["timestamp", "session_id", "left_wrist_x", "left_wrist_y"]
    for col in required_cols:
        assert col in result.columns, f"Missing column: {col}"

    # test 5: timestamp is sorted
    assert result["timestamp"].is_monotonic_increasing, "Timestamps must be sorted"

    # test 6: reasonable row count (30 Hz × 10 sec = 300 rows ± tolerance)
    assert 250 < len(result) < 350, f"Unexpected row count: {len(result)}"

    return True
```

---

## 4. evaluation pipeline (полный пайплайн оценки)

### 4.1 evaluation orchestrator

```python
# evaluator.py

import time
import tracemalloc
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np

@dataclass
class EvaluationResult:
    passed: bool
    score: float  # 0-100
    metrics: Dict[str, Any]
    errors: list
    warnings: list

def evaluate_submission(solution_path: str) -> EvaluationResult:
    """
    Full evaluation pipeline.

    1. Load student solution
    2. Run on multiple test cases
    3. Compare to reference outputs
    4. Compute metrics
    5. Check anti-gaming
    """

    # Import student solution
    import importlib.util
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    metrics = {}
    errors = []
    warnings = []

    # =========== TEST CASES ===========

    test_cases = [
        # Basic test
        {
            "subject_id": "01",
            "session_ids": [1],
            "time_windows": [(0, 60)],
            "target_keypoints": ["left_wrist", "right_wrist"],
        },
        # Multi-session
        {
            "subject_id": "01",
            "session_ids": [1, 2],
            "time_windows": [(0, 30)],
            "target_keypoints": ["left_wrist", "right_wrist"],
        },
        # Multi-window
        {
            "subject_id": "01",
            "session_ids": [1],
            "time_windows": [(0, 30), (3600, 3630)],  # start and 1hr mark
            "target_keypoints": ["left_wrist", "right_wrist", "nose"],
        },
        # HIDDEN: Different subject (holdout)
        {
            "subject_id": "11",  # not seen during development
            "session_ids": [1],
            "time_windows": [(0, 60)],
            "target_keypoints": ["left_wrist"],
            "hidden": True,
        },
    ]

    # =========== RUN EACH TEST ===========

    alignment_scores = []
    memory_usages = []
    processing_times = []

    for i, tc in enumerate(test_cases):
        try:
            # Start memory tracking
            tracemalloc.start()
            start_time = time.perf_counter()

            # Run student solution
            result = solution.load_and_merge(
                tc["subject_id"],
                tc["session_ids"],
                tc["time_windows"],
                tc["target_keypoints"],
            )

            # Stop timing
            elapsed = time.perf_counter() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            processing_times.append(elapsed)
            memory_usages.append(peak / 1e9)  # GB

            # Load reference output
            ref_path = f"reference_outputs/test_{i}.parquet"
            reference = pd.read_parquet(ref_path)

            # Compute alignment accuracy
            alignment = compute_alignment_accuracy(result, reference)
            alignment_scores.append(alignment)

            # Check schema
            schema_ok = check_schema(result, tc)
            if not schema_ok:
                warnings.append(f"Test {i}: schema issues")

        except Exception as e:
            errors.append(f"Test {i} failed: {str(e)}")
            alignment_scores.append(0.0)

    # =========== AGGREGATE METRICS ===========

    metrics["alignment_accuracy"] = np.mean(alignment_scores)
    metrics["alignment_min"] = np.min(alignment_scores)
    metrics["peak_memory_gb"] = np.max(memory_usages)
    metrics["avg_processing_time"] = np.mean(processing_times)
    metrics["max_processing_time"] = np.max(processing_times)

    # =========== ANTI-GAMING CHECKS ===========

    antigaming_result = run_antigaming_checks(solution)
    if antigaming_result["trivial_solution"]:
        errors.append("FAIL: Trivial solution detected")
    if antigaming_result["hardcoded_values"]:
        errors.append("FAIL: Hardcoded test-specific values detected")

    # =========== COMPUTE FINAL SCORE ===========

    score = compute_final_score(metrics, errors)
    passed = score >= 60 and len(errors) == 0

    return EvaluationResult(
        passed=passed,
        score=score,
        metrics=metrics,
        errors=errors,
        warnings=warnings,
    )
```

### 4.2 alignment accuracy computation

```python
def compute_alignment_accuracy(
    submitted: pd.DataFrame,
    reference: pd.DataFrame,
    tolerance_ms: float = 33.0,  # ~1 frame at 30 Hz
) -> float:
    """
    Compute what fraction of timestamps are correctly aligned.

    For each reference timestamp, find nearest submitted timestamp.
    Count fraction within tolerance.
    """

    submitted_ts = submitted["timestamp"].values
    reference_ts = reference["timestamp"].values

    # For each reference timestamp, find nearest in submitted
    correct = 0
    total = len(reference_ts)

    for ref_t in reference_ts:
        # Find closest submitted timestamp
        idx = np.argmin(np.abs(submitted_ts - ref_t))
        diff_ms = np.abs(submitted_ts[idx] - ref_t) * 1000

        if diff_ms <= tolerance_ms:
            correct += 1

    return correct / total


def check_data_integrity(submitted: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """
    Check that actual data values match reference.
    """

    # Check ECoG columns exist and have reasonable values
    ecog_cols = [c for c in submitted.columns if c.startswith("ecog_")]
    ref_ecog_cols = [c for c in reference.columns if c.startswith("ecog_")]

    results = {
        "ecog_columns_match": set(ecog_cols) == set(ref_ecog_cols),
        "pose_values_match": True,
        "movement_labels_match": True,
    }

    # Check pose values (allowing for interpolation differences)
    for kp in ["left_wrist", "right_wrist"]:
        for coord in ["x", "y"]:
            col = f"{kp}_{coord}"
            if col in submitted.columns and col in reference.columns:
                # correlation should be very high
                corr = np.corrcoef(
                    submitted[col].dropna(),
                    reference[col].dropna()
                )[0, 1]
                if corr < 0.99:
                    results["pose_values_match"] = False

    # Check movement labels
    if "is_movement" in submitted.columns:
        submitted_movements = submitted["is_movement"].sum()
        reference_movements = reference["is_movement"].sum()
        # should be within 5%
        if abs(submitted_movements - reference_movements) / reference_movements > 0.05:
            results["movement_labels_match"] = False

    return results
```

### 4.3 anti-gaming checks

```python
def run_antigaming_checks(solution_module) -> dict:
    """
    Detect cheating strategies.
    """

    import inspect
    source = inspect.getsource(solution_module.load_and_merge)

    results = {
        "trivial_solution": False,
        "hardcoded_values": False,
        "timestamp_copying": False,
        "session_specific_code": False,
    }

    # Check 1: Hardcoded subject/session IDs
    if '"11"' in source or "'11'" in source:  # hidden test subject
        results["hardcoded_values"] = True

    # Check 2: Trivial solution (constant output)
    test_result1 = solution_module.load_and_merge("01", [1], [(0, 10)], ["left_wrist"])
    test_result2 = solution_module.load_and_merge("01", [1], [(100, 110)], ["left_wrist"])

    if test_result1["timestamp"].equals(test_result2["timestamp"]):
        results["trivial_solution"] = True

    # Check 3: Timestamp precision check
    # If timestamps are suspiciously round numbers, something is wrong
    ts_diff = test_result1["timestamp"].diff().dropna()
    if ts_diff.std() < 1e-10:  # perfectly uniform
        results["timestamp_copying"] = True

    # Check 4: Session-specific branching
    if "if session_id ==" in source or "if subject_id ==" in source:
        results["session_specific_code"] = True

    return results
```

### 4.4 final score computation

```python
def compute_final_score(metrics: dict, errors: list) -> float:
    """
    Compute final score 0-100.

    Weights:
    - alignment_accuracy: 40%
    - schema_correctness: 20%
    - memory_efficiency: 20%
    - processing_time: 20%

    Penalties:
    - Each error: -20 points
    """

    score = 0.0

    # Alignment (40 points max)
    # 95% = 30 points, 99% = 40 points
    alignment = metrics["alignment_accuracy"]
    if alignment >= 0.99:
        score += 40
    elif alignment >= 0.95:
        score += 30 + (alignment - 0.95) / 0.04 * 10
    elif alignment >= 0.90:
        score += 20 + (alignment - 0.90) / 0.05 * 10
    else:
        score += max(0, alignment * 22)

    # Schema (20 points)
    # Binary: either schema is correct or not
    # (from per-test schema checks aggregated)
    score += 20  # assume correct for now

    # Memory (20 points max)
    # <2GB = 20, <4GB = 15, <8GB = 10, else 0
    mem = metrics["peak_memory_gb"]
    if mem < 2:
        score += 20
    elif mem < 4:
        score += 15
    elif mem < 8:
        score += 10
    else:
        score += 0

    # Processing time (20 points max)
    # <20s = 20, <60s = 15, <120s = 10, else 0
    time = metrics["avg_processing_time"]
    if time < 20:
        score += 20
    elif time < 60:
        score += 15
    elif time < 120:
        score += 10
    else:
        score += 0

    # Penalties
    score -= len(errors) * 20

    return max(0, min(100, score))
```

---

## 5. reference outputs (ground truth)

### 5.1 как генерируются reference outputs

```python
# generate_reference.py
# Запускается offline, один раз

from reference_solution import load_and_merge_reference

test_cases = [...]  # same as evaluation

for i, tc in enumerate(test_cases):
    result = load_and_merge_reference(
        tc["subject_id"],
        tc["session_ids"],
        tc["time_windows"],
        tc["target_keypoints"],
    )

    # save as parquet (efficient, preserves dtypes)
    result.to_parquet(f"reference_outputs/test_{i}.parquet")

    # also save checksums
    checksums = {
        "row_count": len(result),
        "timestamp_hash": hash(tuple(result["timestamp"].values)),
        "ecog_mean": result[[c for c in result.columns if "ecog" in c]].mean().mean(),
    }
    json.dump(checksums, open(f"reference_outputs/test_{i}_checksums.json", "w"))
```

### 5.2 reference solution (идеальное решение)

```python
# reference_solution.py
# Это "золотой стандарт" - оптимальная имплементация

def load_and_merge_reference(subject_id, session_ids, time_windows, target_keypoints):
    """
    Reference implementation with:
    - Efficient streaming (remfile)
    - Proper timestamp alignment
    - Memory-efficient chunking
    - Correct handling of edge cases
    """

    all_dfs = []

    for session_id in session_ids:
        url = get_nwb_url(subject_id, session_id)

        # Use disk cache for repeated access
        cache = remfile.DiskCache("/tmp/nwb_cache")
        rem_file = remfile.File(url, disk_cache=cache)

        with h5py.File(rem_file, "r") as h5:
            with NWBHDF5IO(file=h5, load_namespaces=True) as io:
                nwb = io.read()

                # Get electrode info once
                electrodes = nwb.electrodes.to_dataframe()
                good_mask = electrodes["good"]
                good_indices = electrodes[good_mask].index.tolist()

                # Get reaches and epochs once
                reaches = nwb.intervals["reaches"].to_dataframe()
                epochs = nwb.intervals["epochs"].to_dataframe()

                for start_time, end_time in time_windows:
                    df = process_window(
                        nwb,
                        start_time,
                        end_time,
                        good_indices,
                        target_keypoints,
                        reaches,
                        epochs,
                        session_id,
                    )
                    all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def process_window(nwb, start, end, good_indices, keypoints, reaches, epochs, session_id):
    """Process single time window efficiently."""

    # === Load pose (defines output timestamps) ===
    position = nwb.processing["behavior"]["Position"]
    series = list(position.spatial_series.values())[0]

    pose_ts = series.timestamps[:]
    pose_mask = (pose_ts >= start) & (pose_ts <= end)
    pose_ts = pose_ts[pose_mask]
    pose_data = series.data[pose_mask]

    # === Load ECoG ===
    ecog = nwb.acquisition["ElectricalSeries"]
    ecog_rate = ecog.rate

    start_sample = int(start * ecog_rate)
    end_sample = int(end * ecog_rate)

    # Only load good electrodes
    ecog_data = ecog.data[start_sample:end_sample, good_indices]
    ecog_ts = np.arange(start_sample, end_sample) / ecog_rate

    # === Align ECoG to pose timestamps ===
    # Use vectorized approach for speed
    aligned_ecog = np.zeros((len(pose_ts), len(good_indices)), dtype=np.float32)

    half_window = 0.5 / POSE_RATE

    for i, t in enumerate(pose_ts):
        mask = (ecog_ts >= t - half_window) & (ecog_ts < t + half_window)
        if mask.sum() > 0:
            aligned_ecog[i] = ecog_data[mask].mean(axis=0)
        else:
            aligned_ecog[i] = np.nan

    # === Build DataFrame ===
    df = pd.DataFrame()
    df["timestamp"] = pose_ts
    df["session_id"] = session_id

    # ECoG columns
    for i, idx in enumerate(good_indices):
        df[f"ecog_{idx}"] = aligned_ecog[:, i]

    # Pose columns
    keypoint_names = ["nose", "left_ear", "right_ear",
                     "left_shoulder", "right_shoulder",
                     "left_elbow", "right_elbow",
                     "left_wrist", "right_wrist"]

    for kp in keypoints:
        kp_idx = keypoint_names.index(kp)
        df[f"{kp}_x"] = pose_data[:, kp_idx, 0]
        df[f"{kp}_y"] = pose_data[:, kp_idx, 1]

    # === Add annotations ===
    df["is_movement"] = False
    for _, reach in reaches.iterrows():
        if reach["stop_time"] < start or reach["start_time"] > end:
            continue
        mask = (df["timestamp"] >= reach["start_time"]) & \
               (df["timestamp"] <= reach["stop_time"])
        df.loc[mask, "is_movement"] = True

    df["behavior_state"] = "unknown"
    for _, epoch in epochs.iterrows():
        if epoch["stop_time"] < start or epoch["start_time"] > end:
            continue
        mask = (df["timestamp"] >= epoch["start_time"]) & \
               (df["timestamp"] <= epoch["stop_time"])
        tags = epoch["tags"]
        df.loc[mask, "behavior_state"] = tags[0] if tags else "unknown"

    return df
```

---

## 6. feedback to student (что видит студент после оценки)

### 6.1 если PASSED

```
═══════════════════════════════════════════════════════════════
                    EVALUATION RESULTS
═══════════════════════════════════════════════════════════════

Status: ✅ PASSED

Score: 78/100

Metrics:
  • Alignment accuracy: 96.3% (threshold: 95%)
  • Peak memory usage: 2.8 GB (threshold: 4 GB)
  • Processing time: 45.2s avg (threshold: 60s)
  • Schema correctness: ✅

Breakdown:
  ├─ Alignment:       34/40 points
  ├─ Schema:          20/20 points
  ├─ Memory:          15/20 points
  └─ Time:            15/20 points

Grade: GOOD

Suggestions for improvement:
  • Your alignment could be improved by using proper interpolation
    instead of nearest-neighbor matching
  • Memory usage could be reduced by processing in smaller chunks
  • Consider caching electrode metadata across time windows

═══════════════════════════════════════════════════════════════
```

### 6.2 если FAILED

```
═══════════════════════════════════════════════════════════════
                    EVALUATION RESULTS
═══════════════════════════════════════════════════════════════

Status: ❌ FAILED

Score: 42/100

Errors:
  ✗ Test 3 failed: KeyError: 'nose_x'
  ✗ Alignment accuracy below threshold: 72% < 95%

Metrics:
  • Alignment accuracy: 72.1% (threshold: 95%)
  • Peak memory usage: 6.2 GB (threshold: 4 GB) ⚠️
  • Processing time: 180.5s avg (threshold: 60s) ⚠️

Warnings:
  • Missing keypoint columns in test 3
  • Memory exceeds limit

Debug hints:
  • Check that all requested keypoints are included in output
  • Your alignment might be off - verify timestamp conversion
  • Consider using streaming instead of loading entire files

Failed test case details:
  Test 3: subject="01", sessions=[1], windows=[(0,30),(3600,3630)]
  Expected columns: ['nose_x', 'nose_y', 'left_wrist_x', ...]
  Got columns: ['left_wrist_x', 'left_wrist_y', ...]

═══════════════════════════════════════════════════════════════
```

---

## 7. summary: full pipeline visualization

```
┌─────────────────────────────────────────────────────────────┐
│                     STUDENT WORKFLOW                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. READ TASK DESCRIPTION                                   │
│     - Understand input format (subject, sessions, windows)  │
│     - Understand output format (DataFrame schema)           │
│     - Note constraints (memory, time)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. EXPLORE DATA (using provided tools)                     │
│     - Open NWB file via streaming                          │
│     - Inspect ECoG: shape, rate, electrodes                │
│     - Inspect Pose: keypoints, timestamps                   │
│     - Understand electrode quality flags                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. IMPLEMENT SOLUTION                                      │
│     - Load ECoG slice (streaming, only good electrodes)    │
│     - Load Pose slice (handle storage format variants)     │
│     - Align timestamps (downsample ECoG to pose rate)      │
│     - Add movement/behavior annotations                     │
│     - Build DataFrame with correct schema                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. LOCAL TESTING                                           │
│     - Run on small test case (10 seconds)                  │
│     - Check output shape and columns                        │
│     - Verify memory usage                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. SUBMIT solution.py                                      │
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
│     - Function exists and callable                         │
│     - Basic invocation doesn't crash                        │
│     - Returns DataFrame with required columns               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  B. RUN TEST CASES                                          │
│     - Basic: single session, single window                 │
│     - Multi-session: multiple sessions                      │
│     - Multi-window: discontinuous time ranges               │
│     - HIDDEN: different subject (holdout)                   │
│                                                             │
│     For each test:                                          │
│     - Measure processing time                               │
│     - Track peak memory                                     │
│     - Compare to reference output                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  C. COMPUTE METRICS                                         │
│     - Alignment accuracy (timestamps within tolerance)      │
│     - Data integrity (values match reference)               │
│     - Memory efficiency (peak usage)                        │
│     - Processing time                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  D. ANTI-GAMING CHECKS                                      │
│     - No hardcoded values for hidden test                   │
│     - No trivial/constant solutions                         │
│     - Timestamps actually computed (not copied)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  E. COMPUTE FINAL SCORE                                     │
│     Alignment (40%) + Schema (20%) +                        │
│     Memory (20%) + Time (20%) - Penalties                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  F. RETURN FEEDBACK                                         │
│     - Pass/Fail status                                      │
│     - Detailed metrics                                      │
│     - Improvement suggestions                               │
│     - Debug hints for failures                              │
└─────────────────────────────────────────────────────────────┘
```

---

*end of detailed pipeline specification*
