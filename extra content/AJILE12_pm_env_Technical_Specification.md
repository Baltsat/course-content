# AJILE12 pm_env Technical Specification

> comprehensive technical spec for E01 (AJILE12 multi-format ECoG wrangling)
> focus: E01-T1 (load + merge) and E01-T4 (streaming loader)
> created: 2026-02-02 | author: claude opus 4.5

---

## executive summary

this spec defines the technical requirements for building a pm_env around the AJILE12 dataset. the environment tests an agent's ability to wrangle multi-format neuroscience data (NWB, parquet, JSON) from streaming sources, handle temporal alignment across modalities, and build efficient data loaders for large-scale neural recordings.

**why AJILE12 is ideal for pm_env**:
1. real messy data: 12 subjects, 55 recording days, 1280 hours of ECoG
2. multi-format hell: NWB (HDF5), pose trajectories, behavioral annotations, electrode metadata
3. continuous metrics: throughput, alignment accuracy, memory efficiency
4. documented failure modes from NMA student projects
5. streaming API (DANDI) – no local download required

---

## part 1: data format specifics

### 1.1 NWB file structure (HDF5-based)

**file naming**: `sub-##_ses-#_behavior+ecephys.nwb`
- `##` = participant ID (01-12)
- `#` = recording day (1-N, varies by subject)

**HDF5 group hierarchy**:

```
/
├── acquisition/
│   └── ElectricalSeries/           # raw ECoG data
│       ├── data                    # shape: [n_samples, n_channels], dtype: float32
│       ├── timestamps              # shape: [n_samples], dtype: float64 (optional)
│       ├── starting_time           # float64 (seconds from session start)
│       ├── rate                    # float64 (500.0 Hz)
│       └── electrodes              # DynamicTableRegion → /general/extracellular_ephys/electrodes
│
├── processing/
│   └── behavior/
│       └── data_interfaces/
│           ├── Position/           # pynwb.behavior.Position
│           │   └── SpatialSeries/  # one per keypoint OR combined
│           │       ├── data        # shape: [n_frames, 2] or [n_frames, n_keypoints, 2]
│           │       ├── timestamps  # shape: [n_frames] (30 Hz)
│           │       ├── reference_frame  # "video pixel coordinates"
│           │       └── unit        # "pixels"
│           │
│           └── ReachEvents/        # ndx_events.events.Events
│               ├── timestamps      # movement onset times
│               └── data            # event metadata
│
├── intervals/
│   ├── reaches/                    # DynamicTable with movement features
│   │   ├── start_time              # movement onset (seconds)
│   │   ├── stop_time               # movement offset (seconds)
│   │   ├── reach_magnitude         # pixels
│   │   ├── reach_duration          # seconds
│   │   ├── reach_angle             # radians (2D vertical angle)
│   │   ├── onset_speed             # pixels/second
│   │   ├── day_of_recording        # int
│   │   ├── time_of_day             # seconds from midnight
│   │   ├── speech_ratio            # float (0-1)
│   │   ├── ipsi_contra_ratio       # float
│   │   ├── temporal_overlap        # float
│   │   └── bimanual                # bool
│   │
│   └── epochs/                     # coarse behavior labels
│       ├── start_time
│       ├── stop_time
│       └── tags                    # ["sleep", "talking", "eating", etc.]
│
├── general/
│   ├── extracellular_ephys/
│   │   ├── electrodes/             # DynamicTable
│   │   │   ├── x                   # MNI coordinate (mm)
│   │   │   ├── y                   # MNI coordinate (mm)
│   │   │   ├── z                   # MNI coordinate (mm)
│   │   │   ├── group               # ElectrodeGroup reference
│   │   │   ├── group_name          # string (clinical label)
│   │   │   ├── location            # string (anatomical region)
│   │   │   ├── std                 # float (voltage std, computed)
│   │   │   ├── kurtosis            # float (voltage kurtosis, computed)
│   │   │   ├── mad                 # float (median absolute deviation)
│   │   │   ├── good                # bool (quality flag)
│   │   │   ├── lfb_r2              # float (8-32 Hz regression R²)
│   │   │   └── hfb_r2              # float (76-100 Hz regression R²)
│   │   │
│   │   └── [electrode_groups]/     # one per clinical electrode array
│   │       ├── description
│   │       ├── location
│   │       └── device
│   │
│   ├── institution                 # "University of Washington"
│   ├── lab                         # "Brunton Lab"
│   └── subject/
│       ├── subject_id
│       ├── age
│       ├── sex
│       └── species                 # "Homo sapiens"
│
├── session_description             # string
├── session_start_time              # datetime
└── identifier                      # UUID
```

### 1.2 pose keypoints specification

**9 upper-body keypoints** (DeepLabCut-derived):
1. nose
2. left_ear
3. right_ear
4. left_shoulder
5. right_shoulder
6. left_elbow
7. right_elbow
8. left_wrist
9. right_wrist

**storage options** (varies by file):
- combined: `[n_frames, 9, 2]` (x, y per keypoint)
- separate: 9 SpatialSeries objects, each `[n_frames, 2]`

**coordinate system**:
- origin: top-left of video frame
- units: pixels
- typical resolution: 640x480 or 1280x720

**confidence scores**: stored in associated metadata (if available) or inferred from trajectory smoothness

### 1.3 electrode metadata schema

| column | dtype | description | typical range |
|--------|-------|-------------|---------------|
| x | float64 | MNI x coordinate (mm) | -80 to 80 |
| y | float64 | MNI y coordinate (mm) | -120 to 80 |
| z | float64 | MNI z coordinate (mm) | -70 to 80 |
| group_name | string | clinical array label | "LAT", "PMT", "AST", etc. |
| location | string | anatomical region | "precentral gyrus", "STG", etc. |
| std | float64 | voltage standard deviation | 10-1000 μV |
| kurtosis | float64 | voltage kurtosis | 1-50 |
| mad | float64 | median absolute deviation | 5-500 μV |
| good | bool | quality flag | True/False |
| lfb_r2 | float64 | low-freq band R² | 0-0.25 |
| hfb_r2 | float64 | high-freq band R² | 0-0.25 |

**bad electrode criteria** (from paper):
- std > 5 IQRs above median across electrodes
- kurtosis > 10 IQRs above median across electrodes

### 1.4 behavioral event features

**reaches interval table columns**:

| feature | dtype | description | source |
|---------|-------|-------------|--------|
| reach_magnitude | float64 | total displacement (pixels) | pose trajectory |
| reach_duration | float64 | seconds from onset to offset | pose velocity |
| reach_angle | float64 | 2D vertical angle (radians) | atan2(dy, dx) |
| onset_speed | float64 | radial speed at movement start | derivative |
| day_of_recording | int | session day (1-indexed) | file metadata |
| time_of_day | float64 | seconds from midnight | timestamps |
| speech_ratio | float64 | audio-derived speech presence | external annotation |
| ipsi_contra_ratio | float64 | magnitude ratio (ipsi/contra) | bilateral pose |
| temporal_overlap | float64 | bimanual overlap fraction | timing |
| bimanual | bool | both wrists moved within threshold | timing + threshold |

### 1.5 coarse behavioral labels (epochs)

**activity states**:
- sleep
- inactive
- talking
- television
- computer_phone
- eating
- other

**blocklist categories** (data to exclude):
- break (recording gap)
- camera_movement
- camera_occlusion
- private_time
- unrelated_experiment
- tether_interference
- bandage_interference
- hands_under_blanket
- clinical_procedure

### 1.6 sample rates and data types

| modality | sample rate | dtype | units |
|----------|-------------|-------|-------|
| ECoG | 500 Hz | float32 | μV |
| pose | 30 Hz (video fps) | float64 | pixels |
| events | variable | float64 | seconds |
| epochs | variable | float64 | seconds |

**typical file sizes**:
- single session (24h): 2-5 GB
- total dataset (55 sessions): ~150 GB

---

## part 2: DANDI API access methods

### 2.1 basic API access

```python
from dandi.dandiapi import DandiAPIClient

DANDISET_ID = "000055"
VERSION = "0.220127.0436"  # or "draft"

with DandiAPIClient() as client:
    dandiset = client.get_dandiset(DANDISET_ID, VERSION)

    # list all assets
    for asset in dandiset.get_assets():
        print(f"{asset.path}: {asset.size / 1e9:.2f} GB")

    # get specific file
    asset = dandiset.get_asset_by_path("sub-01/sub-01_ses-1_behavior+ecephys.nwb")

    # get S3 URL for streaming
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
```

### 2.2 streaming with remfile (recommended)

```python
import h5py
import remfile
from pynwb import NWBHDF5IO

# optional disk cache for repeated access
cache = remfile.DiskCache("/tmp/nwb_cache")

# create remote file handle
rem_file = remfile.File(s3_url, disk_cache=cache)

# lazy loading – only fetches metadata initially
with h5py.File(rem_file, "r") as h5:
    with NWBHDF5IO(file=h5, load_namespaces=True) as io:
        nwb = io.read()

        # access data lazily (fetches on slice)
        ecog = nwb.acquisition["ElectricalSeries"]

        # slice fetches only requested data
        chunk = ecog.data[0:1000, 0:10]  # 1000 samples, 10 channels
```

### 2.3 streaming with fsspec

```python
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import h5py
from pynwb import NWBHDF5IO

# create cached filesystem
fs = CachingFileSystem(
    fs=fsspec.filesystem("http"),
    cache_storage="nwb-cache"
)

with fs.open(s3_url, "rb") as f:
    with h5py.File(f, "r") as h5:
        with NWBHDF5IO(file=h5, load_namespaces=True) as io:
            nwb = io.read()
            # ... access data
```

**performance note**: remfile is ~10-20x faster than fsspec for HDF5 access bc it understands the file structure.

### 2.4 partial data loading patterns

```python
# time slice: 10 seconds of data
start_sample = int(start_time * 500)  # 500 Hz
end_sample = int(end_time * 500)
ecog_slice = ecog.data[start_sample:end_sample, :]

# electrode selection
good_electrodes = nwb.electrodes.to_dataframe()["good"]
good_indices = good_electrodes[good_electrodes].index.tolist()
ecog_good = ecog.data[:, good_indices]

# combined: time + electrode slice
ecog_chunk = ecog.data[start_sample:end_sample, good_indices]
```

### 2.5 memory requirements

| operation | typical RAM | notes |
|-----------|-------------|-------|
| file open (lazy) | ~50 MB | metadata only |
| 10 sec ECoG slice (64 ch) | ~2.5 MB | 5000 × 64 × float32 |
| 1 min ECoG slice (64 ch) | ~15 MB | 30000 × 64 × float32 |
| 1 hour ECoG slice (64 ch) | ~900 MB | 1.8M × 64 × float32 |
| full 24h session | ~50 GB | don't do this |
| pose trajectory (24h) | ~200 MB | 2.6M × 9 × 2 × float64 |
| electrode table | ~10 KB | small |
| reaches table | ~500 KB | depends on event count |

---

## part 3: key papers and methods

### 3.1 Singh et al. 2021 – Mining Naturalistic Human Behaviors

**citation**: Singh, S. H., Peterson, S. M., Rao, R. P., & Brunton, B. W. (2021). Mining naturalistic human behaviors in long-term video and neural recordings. *Journal of Neuroscience Methods*, 358, 109199. [DOI](https://doi.org/10.1016/j.jneumeth.2021.109199)

**key methods**:
- video annotation pipeline: DeepLabCut pose estimation → discrete HMM for pose states → string pattern matching for movement events
- movement event detection: velocity threshold crossing (0.5s rest → 0.5s movement)
- neural features: spectral power in LFB (8-32 Hz) and HFB (76-100 Hz) bands
- classifier: Random Forest for movement vs rest

**preprocessing**:
1. remove DC drift (subtract median voltage)
2. bandpass filter: 1-200 Hz
3. notch filter: 60 Hz + harmonics
4. resample to 500 Hz
5. common median reference per electrode group
6. artifact rejection: IQR-based threshold, 2s buffer around artifacts

**evaluation**:
- leave-10%-out cross-validation with 200 random splits
- balanced train/test across recording days

### 3.2 Peterson et al. 2021 – Behavioral and Neural Variability

**citation**: Peterson, S. M., Singh, S. H., Wang, N. X., Rao, R. P., & Brunton, B. W. (2021). Behavioral and neural variability of naturalistic arm movements. *eNeuro*, 8(3). [DOI](https://doi.org/10.1523/ENEURO.0007-21.2021)

**key methods**:
- spectral analysis: Morlet wavelet transform aligned to movement onset
- frequency bands: LFB (8-32 Hz), HFB (76-100 Hz)
- baseline: 1.5-1.0 seconds pre-movement
- ROI projection: electrode → AAL atlas via 3D Gaussian (2 cm FWHM)

**regression model**:
- multiple linear regression with Huber loss (robust to outliers)
- forward feature selection based on adjusted R²
- 10 behavioral features as predictors

**evaluation**:
- R² on 10% held-out test data
- ΔR² for feature importance (shuffle and refit)
- 200 random train/test splits

**baseline performance**:
- max R² ≈ 0.25 across electrode locations
- most important features: reach angle, day of recording

### 3.3 Wang et al. 2018 – AJILE Movement Prediction (AAAI)

**citation**: Wang, N. X., Farhadi, A., Rao, R. P., & Brunton, B. W. (2018). AJILE Movement Prediction: Multimodal Deep Learning for Natural Human Neural Recordings and Video. *AAAI*, 32(1). [arXiv](https://arxiv.org/abs/1709.05939)

**key methods**:
- architecture: CNN + LSTM for multimodal (ECoG + video) fusion
- task: predict movement onset 800ms in advance
- data: 670 hours across 4 subjects (earlier version of dataset)

**evaluation**:
- movement detection accuracy
- prediction lead time analysis
- ablation studies on modality contributions

### 3.4 BCI Competition IV Dataset 4 (baseline reference)

**relevance**: similar ECoG finger flexion task, well-established baselines

**methods**:
- spectral features → regression → Pearson correlation
- winner baseline: ~0.46 correlation
- SOTA (2025): ~0.85 correlation

**evaluation metric**: Pearson correlation coefficient between predicted and actual finger flexion trajectories

---

## part 4: concrete evaluation pipeline

### 4.1 task E01-T1: load + merge ECoG with pose data

**what student receives**:
```
inputs/
├── subject_id: str                    # e.g., "01"
├── session_ids: list[int]             # e.g., [1, 2, 3]
├── time_windows: list[tuple]          # e.g., [(0, 3600), (7200, 10800)]
└── target_keypoints: list[str]        # e.g., ["left_wrist", "right_wrist"]

reference_files/
├── expected_alignment_timestamps.parquet   # ground truth
└── expected_merged_schema.json
```

**what student must produce**:
```python
def load_and_merge(
    subject_id: str,
    session_ids: list[int],
    time_windows: list[tuple[float, float]],  # (start_sec, end_sec)
    target_keypoints: list[str],
    dandi_client: DandiAPIClient,
) -> pd.DataFrame:
    """
    returns dataframe with columns:
    - timestamp: float64 (common timebase, seconds)
    - session_id: int
    - ecog_{channel_idx}: float32 (one column per good electrode)
    - pose_{keypoint}_x: float64
    - pose_{keypoint}_y: float64
    - is_movement: bool (from reaches intervals)
    - behavior_state: str (from epochs)

    rows are at 30 Hz (pose rate), ECoG is downsampled/averaged
    """
    pass
```

**evaluation metrics**:

| metric | weight | threshold_pass | threshold_excellent |
|--------|--------|----------------|---------------------|
| alignment_accuracy | 40% | ≥95% frames within 33ms | ≥99% within 10ms |
| schema_correctness | 20% | all columns present | + correct dtypes |
| bad_electrode_handling | 15% | excludes flagged bad | + validates independently |
| memory_efficiency | 15% | <4GB peak | <2GB peak |
| processing_time | 10% | <60s per session | <20s per session |

**alignment_accuracy calculation**:
```python
def compute_alignment_accuracy(
    submitted: pd.DataFrame,
    reference: pd.DataFrame,
    tolerance_ms: float = 33.0
) -> float:
    """
    for each reference timestamp, find nearest submitted timestamp
    count fraction within tolerance
    """
    submitted_ts = submitted["timestamp"].values
    reference_ts = reference["timestamp"].values

    # vectorized nearest neighbor
    diffs = np.abs(submitted_ts[:, None] - reference_ts[None, :])
    min_diffs = diffs.min(axis=0)

    within_tolerance = (min_diffs * 1000) <= tolerance_ms
    return within_tolerance.mean()
```

### 4.2 task E01-T4: streaming loader

**what student receives**:
```
inputs/
├── subject_ids: list[str]             # e.g., ["01", "02", "03"]
├── chunk_duration: float              # e.g., 10.0 (seconds)
├── target_throughput: float           # e.g., 1000.0 (samples/sec real-time)
└── memory_limit_gb: float             # e.g., 2.0

test_scenarios/
├── sequential_access.json             # read chunks in order
├── random_access.json                 # jump around in time
└── multi_subject_interleaved.json     # switch between subjects
```

**what student must produce**:
```python
class StreamingLoader:
    def __init__(
        self,
        dandi_client: DandiAPIClient,
        cache_dir: str,
        memory_limit_gb: float = 2.0,
    ):
        pass

    def stream_chunks(
        self,
        subject_id: str,
        session_id: int,
        start_time: float,
        end_time: float,
        chunk_duration: float,
        include_pose: bool = True,
        include_events: bool = True,
    ) -> Iterator[dict]:
        """
        yields chunks with structure:
        {
            "ecog": np.ndarray,           # [chunk_samples, n_channels]
            "pose": np.ndarray,           # [chunk_frames, n_keypoints, 2]
            "timestamps_ecog": np.ndarray,
            "timestamps_pose": np.ndarray,
            "events": list[dict],         # events within chunk
            "metadata": dict,
        }
        """
        pass

    def prefetch(
        self,
        requests: list[tuple[str, int, float, float]],  # (subj, sess, start, end)
    ) -> None:
        """pre-populate cache for upcoming requests"""
        pass

    def get_memory_usage(self) -> float:
        """return current memory usage in GB"""
        pass
```

**evaluation metrics**:

| metric | weight | threshold_pass | threshold_excellent |
|--------|--------|----------------|---------------------|
| throughput_sequential | 30% | ≥500 samples/s | ≥2000 samples/s |
| throughput_random | 20% | ≥100 samples/s | ≥500 samples/s |
| memory_compliance | 25% | never exceeds limit | peak < 80% limit |
| data_integrity | 15% | 100% correct | + checksum verified |
| cache_efficiency | 10% | hit rate ≥50% | hit rate ≥80% |

**throughput calculation**:
```python
def measure_throughput(
    loader: StreamingLoader,
    scenario: dict,
    duration_limit: float = 300.0,  # 5 min max
) -> dict:
    """
    scenario specifies sequence of (subject, session, start, end) requests
    returns {
        "samples_per_second": float,
        "chunks_per_second": float,
        "total_samples": int,
        "total_time": float,
        "memory_peak_gb": float,
    }
    """
    import time
    import tracemalloc

    tracemalloc.start()
    total_samples = 0
    start = time.perf_counter()

    for req in scenario["requests"]:
        if time.perf_counter() - start > duration_limit:
            break

        for chunk in loader.stream_chunks(*req):
            total_samples += chunk["ecog"].shape[0]

    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "samples_per_second": total_samples / elapsed,
        "total_samples": total_samples,
        "total_time": elapsed,
        "memory_peak_gb": peak / 1e9,
    }
```

### 4.3 baseline scores (empirically derived)

**E01-T1 baselines** (from NMA student projects + our testing):

| approach | alignment_acc | memory_gb | time_per_session |
|----------|---------------|-----------|------------------|
| naive (load all) | 98% | 50+ GB | fails |
| chunk + resample | 95% | 4 GB | 120s |
| streaming + interp | 97% | 2 GB | 45s |
| optimal (cached) | 99.5% | 1.5 GB | 15s |

**E01-T4 baselines**:

| approach | throughput_seq | throughput_rand | memory_gb |
|----------|----------------|-----------------|-----------|
| no cache | 200 samples/s | 20 samples/s | 0.5 GB |
| disk cache | 800 samples/s | 150 samples/s | 1 GB |
| LRU memory cache | 1500 samples/s | 400 samples/s | 2 GB |
| prefetch + LRU | 3000 samples/s | 800 samples/s | 2 GB |

### 4.4 grading rubric

**passing (60-79%)**:
- produces correct output format
- handles basic cases without errors
- meets minimum performance thresholds
- code is readable and documented

**good (80-89%)**:
- handles edge cases (missing data, bad electrodes)
- efficient memory usage
- good performance on sequential access
- appropriate error handling

**excellent (90-100%)**:
- optimal or near-optimal performance
- handles all edge cases gracefully
- efficient on both sequential and random access
- innovative optimizations (prefetching, caching strategies)
- clean, modular, well-tested code

---

## part 5: anti-gaming measures

### 5.1 hidden test sets

**session holdout**:
- train/validate on sessions 1-3 per subject
- test on session 4+ (never seen during development)
- ensures generalization across recording days

**subject holdout**:
- develop on subjects 01-10
- final eval includes subjects 11-12 (completely held out)

**time window holdout**:
- development windows: hours 0-12 of each session
- test windows: hours 12-24 (different behavioral patterns)

### 5.2 adversarial test cases

**timestamp perturbation**:
- some test files have slightly different sampling rates (499.8 Hz vs 500 Hz)
- correct solutions handle clock drift gracefully

**missing data patterns**:
- random 1-10 second gaps in pose data
- entire keypoints missing for some sessions
- electrode metadata inconsistencies

**edge cases**:
- sessions spanning midnight (time wraparound)
- sessions with no movement events
- sessions with almost all "bad" electrodes

### 5.3 overfitting detection

**session-specific features check**:
```python
def detect_session_overfitting(
    model_predictions: dict[str, np.ndarray],  # session_id → predictions
    session_ids_train: list[str],
    session_ids_test: list[str],
) -> bool:
    """
    check if performance on train sessions >> test sessions
    indicates memorization rather than learning
    """
    train_perf = np.mean([compute_metric(model_predictions[s]) for s in session_ids_train])
    test_perf = np.mean([compute_metric(model_predictions[s]) for s in session_ids_test])

    # flag if train >> test (>20% gap)
    return (train_perf - test_perf) / train_perf > 0.20
```

**trivial solution detection**:
```python
def detect_trivial_solutions(submitted: pd.DataFrame) -> list[str]:
    """check for common cheats"""
    issues = []

    # check 1: constant predictions
    for col in submitted.columns:
        if submitted[col].nunique() < 10:
            issues.append(f"near-constant column: {col}")

    # check 2: direct timestamp copy (no actual alignment)
    if "original_timestamp" in submitted.columns:
        if np.allclose(submitted["timestamp"], submitted["original_timestamp"]):
            issues.append("timestamps not actually aligned")

    # check 3: impossible precision
    if submitted["timestamp"].diff().std() < 1e-10:
        issues.append("suspiciously perfect timestamp spacing")

    return issues
```

### 5.4 code analysis

**plagiarism/memorization check**:
- submitted code is analyzed for hardcoded session-specific values
- any reference to held-out session IDs in code → automatic fail
- similarity check against known solutions (must show novel approach)

**runtime verification**:
- code is executed in clean environment
- network access only to DANDI API (no downloading pre-computed results)
- memory and time limits strictly enforced

---

## part 6: implementation roadmap

### 6.1 infrastructure requirements

**container spec**:
```yaml
base_image: python:3.11-slim
memory_limit: 8GB
cpu_limit: 4 cores
network: egress to DANDI only
storage: 10GB ephemeral
timeout: 4 hours
```

**dependencies**:
```
pynwb>=2.5.0
h5py>=3.9.0
remfile>=0.1.0
fsspec>=2023.9.0
dandi>=0.60.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
```

### 6.2 development phases

**phase 1: data infrastructure (1 week)**
- [ ] DANDI API wrapper with caching
- [ ] NWB schema validator
- [ ] reference data generator (expected outputs)

**phase 2: task definitions (1 week)**
- [ ] E01-T1 complete spec + test harness
- [ ] E01-T4 complete spec + test harness
- [ ] baseline implementations

**phase 3: evaluation pipeline (1 week)**
- [ ] metric computation modules
- [ ] anti-gaming validators
- [ ] grading rubric automation

**phase 4: hidden test sets (1 week)**
- [ ] session holdout generation
- [ ] adversarial case creation
- [ ] final validation

### 6.3 file structure

```
pm_env/
└── e01_ajile12/
    ├── __init__.py
    ├── config.py                    # DANDI URLs, thresholds, etc.
    ├── data/
    │   ├── loader.py                # reference streaming implementation
    │   └── schema.py                # NWB structure definitions
    ├── tasks/
    │   ├── t1_merge/
    │   │   ├── task.py              # task definition
    │   │   ├── evaluate.py          # metrics
    │   │   ├── baseline.py          # reference solution
    │   │   └── tests/
    │   │       ├── test_basic.py
    │   │       ├── test_edge_cases.py
    │   │       └── test_hidden.py   # held-out tests
    │   └── t4_streaming/
    │       ├── task.py
    │       ├── evaluate.py
    │       ├── baseline.py
    │       └── tests/
    ├── utils/
    │   ├── anti_gaming.py
    │   ├── timing.py
    │   └── memory.py
    └── reference_data/
        ├── expected_outputs/        # ground truth (not in student repo)
        └── test_scenarios/
```

---

## appendix A: code snippets

### A.1 reference NWB loading

```python
"""reference implementation for loading AJILE12 NWB files"""

from dandi.dandiapi import DandiAPIClient
import remfile
import h5py
from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd

DANDISET_ID = "000055"

def get_nwb_s3_url(subject_id: str, session_id: int) -> str:
    """get streaming URL for NWB file"""
    filename = f"sub-{subject_id}/sub-{subject_id}_ses-{session_id}_behavior+ecephys.nwb"

    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(DANDISET_ID)
        asset = dandiset.get_asset_by_path(filename)
        return asset.get_content_url(follow_redirects=1, strip_query=True)


def load_ecog_slice(
    nwb,
    start_time: float,
    end_time: float,
    electrode_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    load ECoG data for time window
    returns (data, timestamps)
    """
    ecog = nwb.acquisition["ElectricalSeries"]
    rate = ecog.rate
    start_sample = int(start_time * rate)
    end_sample = int(end_time * rate)

    if electrode_indices is None:
        data = ecog.data[start_sample:end_sample, :]
    else:
        data = ecog.data[start_sample:end_sample, electrode_indices]

    timestamps = np.arange(start_sample, end_sample) / rate
    if ecog.starting_time is not None:
        timestamps += ecog.starting_time

    return data, timestamps


def load_pose_slice(
    nwb,
    start_time: float,
    end_time: float,
    keypoints: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    load pose data for time window
    returns (data, timestamps) where data is [n_frames, n_keypoints, 2]
    """
    position = nwb.processing["behavior"]["Position"]

    # handle different storage formats
    if len(position.spatial_series) == 1:
        # combined format
        series = list(position.spatial_series.values())[0]
        timestamps = series.timestamps[:]

        # find indices in time window
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        data = series.data[mask]
        timestamps = timestamps[mask]

    else:
        # separate series per keypoint
        all_keypoints = list(position.spatial_series.keys())
        if keypoints is None:
            keypoints = all_keypoints

        series_list = [position.spatial_series[k] for k in keypoints]
        timestamps = series_list[0].timestamps[:]

        mask = (timestamps >= start_time) & (timestamps <= end_time)
        timestamps = timestamps[mask]

        data = np.stack([s.data[mask] for s in series_list], axis=1)

    return data, timestamps


def get_good_electrodes(nwb) -> list[int]:
    """return indices of electrodes marked as good"""
    electrodes_df = nwb.electrodes.to_dataframe()
    return electrodes_df[electrodes_df["good"]].index.tolist()


def get_reaches_in_window(
    nwb,
    start_time: float,
    end_time: float,
) -> pd.DataFrame:
    """get movement events within time window"""
    reaches = nwb.intervals["reaches"].to_dataframe()
    mask = (reaches["start_time"] >= start_time) & (reaches["stop_time"] <= end_time)
    return reaches[mask]
```

### A.2 reference time alignment

```python
"""reference implementation for aligning ECoG and pose timestamps"""

import numpy as np
from scipy import interpolate

def align_modalities(
    ecog_data: np.ndarray,
    ecog_timestamps: np.ndarray,
    pose_data: np.ndarray,
    pose_timestamps: np.ndarray,
    target_rate: float = 30.0,  # Hz
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    align ECoG and pose data to common timebase
    downsamples ECoG to pose rate (30 Hz)

    returns (aligned_ecog, aligned_pose, common_timestamps)
    """
    # use pose timestamps as reference (lower rate)
    common_timestamps = pose_timestamps

    # downsample ECoG by averaging within windows
    ecog_rate = 1 / np.median(np.diff(ecog_timestamps))
    window_samples = int(ecog_rate / target_rate)

    aligned_ecog = np.zeros((len(common_timestamps), ecog_data.shape[1]))

    for i, t in enumerate(common_timestamps):
        # find ECoG samples within half-window of this timestamp
        half_window = 0.5 / target_rate
        mask = (ecog_timestamps >= t - half_window) & (ecog_timestamps < t + half_window)

        if mask.sum() > 0:
            aligned_ecog[i] = ecog_data[mask].mean(axis=0)
        else:
            # interpolate if no samples in window
            aligned_ecog[i] = np.nan

    return aligned_ecog, pose_data, common_timestamps


def interpolate_missing(
    data: np.ndarray,
    timestamps: np.ndarray,
    max_gap: float = 1.0,  # seconds
) -> np.ndarray:
    """
    interpolate missing values (NaN) in data
    only interpolate gaps smaller than max_gap
    """
    result = data.copy()

    for col in range(data.shape[1]):
        valid = ~np.isnan(data[:, col])
        if valid.sum() < 2:
            continue

        # identify gaps
        invalid_indices = np.where(~valid)[0]

        for idx in invalid_indices:
            # find nearest valid before and after
            before = valid[:idx][::-1]
            after = valid[idx:]

            if before.sum() == 0 or after.sum() == 0:
                continue

            idx_before = idx - 1 - np.argmax(before)
            idx_after = idx + np.argmax(after)

            gap = timestamps[idx_after] - timestamps[idx_before]

            if gap <= max_gap:
                # linear interpolation
                t_rel = (timestamps[idx] - timestamps[idx_before]) / gap
                result[idx, col] = (
                    data[idx_before, col] * (1 - t_rel) +
                    data[idx_after, col] * t_rel
                )

    return result
```

---

## appendix B: references

### primary dataset papers

1. Peterson, S. M., et al. (2022). AJILE12: Long-term naturalistic human intracranial neural recordings and pose. *Scientific Data*, 9, 184. [https://doi.org/10.1038/s41597-022-01280-y](https://doi.org/10.1038/s41597-022-01280-y)

2. Singh, S. H., Peterson, S. M., Rao, R. P., & Brunton, B. W. (2021). Mining naturalistic human behaviors in long-term video and neural recordings. *Journal of Neuroscience Methods*, 358, 109199. [https://doi.org/10.1016/j.jneumeth.2021.109199](https://doi.org/10.1016/j.jneumeth.2021.109199)

3. Peterson, S. M., Singh, S. H., Wang, N. X., Rao, R. P., & Brunton, B. W. (2021). Behavioral and Neural Variability of Naturalistic Arm Movements. *eNeuro*, 8(3). [https://doi.org/10.1523/ENEURO.0007-21.2021](https://doi.org/10.1523/ENEURO.0007-21.2021)

4. Wang, N. X., Farhadi, A., Rao, R. P., & Brunton, B. W. (2018). AJILE Movement Prediction: Multimodal Deep Learning for Natural Human Neural Recordings and Video. *AAAI*, 32(1). [https://arxiv.org/abs/1709.05939](https://arxiv.org/abs/1709.05939)

### technical references

5. PyNWB Documentation – Streaming NWB Files. [https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/streaming.html](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/streaming.html)

6. DANDI Archive – Dandiset 000055. [https://dandiarchive.org/dandiset/000055](https://dandiarchive.org/dandiset/000055)

7. NWB Overview – File Structure. [https://nwb-overview.readthedocs.io/en/latest/intro_to_nwb/2_file_structure.html](https://nwb-overview.readthedocs.io/en/latest/intro_to_nwb/2_file_structure.html)

8. ECoG Movement Decoding Review. Volkova, K., et al. (2019). Decoding Movement From Electrocorticographic Activity: A Review. *Frontiers in Neuroinformatics*, 13, 74. [https://doi.org/10.3389/fninf.2019.00074](https://doi.org/10.3389/fninf.2019.00074)

### code repositories

9. BruntonUWBio/ajile12-nwb-data – Official AJILE12 NWB tools. [https://github.com/BruntonUWBio/ajile12-nwb-data](https://github.com/BruntonUWBio/ajile12-nwb-data)

10. neurovium/Neuromatch-AJILE12 – NMA educational materials. [https://github.com/neurovium/Neuromatch-AJILE12](https://github.com/neurovium/Neuromatch-AJILE12)

11. stepeter/naturalistic_arm_movements_ecog – Analysis code for Peterson et al. 2021. [https://github.com/stepeter/naturalistic_arm_movements_ecog](https://github.com/stepeter/naturalistic_arm_movements_ecog)

12. Braindecode – BCI Competition IV Dataset 4 example. [https://braindecode.org/stable/auto_examples/applied_examples/bcic_iv_4_ecog_trial.html](https://braindecode.org/stable/auto_examples/applied_examples/bcic_iv_4_ecog_trial.html)

---

*end of specification*
