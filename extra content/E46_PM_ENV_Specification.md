# E46: Steinmetz Multi-Area Neural Data Curation – pm_env Specification

> complete pm_env spec for E46: steinmetz neuropixels data curation + cross-area decoding
> dataset: steinmetz et al. 2019 (nature), neuropixels recordings from 42 brain regions
> pattern: data curation → downstream model performance
> created: 2026-02-09 | author: claude opus 4.6

---

## executive summary

this environment tests an agent's ability to curate multi-area neuropixels data from the steinmetz et al. (2019) visual decision-making experiment. the core loop: student curates neural data (quality control, feature engineering, cross-session normalization) → system trains a fixed downstream decoder → score = decoder performance on hidden test sessions from different mice.

**why steinmetz for pm_env**:
1. real messy data: ~30,000 neurons across 42 brain regions, 39 sessions, 10 mice
2. natural quality heterogeneity: low-firing units, multi-unit contamination, unstable recordings
3. continuous metrics: balanced accuracy is naturally continuous and informative
4. cross-session generalization: different mice = hardest neural decoding challenge
5. published nature paper, freely available preprocessed data on OSF/figshare
6. already used in NMA curriculum (load_steinmetz_decisions.ipynb)

---

## 1. dataset overview

### 1.1 source & access

| property | value |
|----------|-------|
| **paper** | steinmetz et al., "distributed coding of choice, action and engagement across the mouse brain" (nature, 2019) |
| **figshare** | https://figshare.com/articles/dataset/steinmetz/9598406 |
| **NMA preprocessed** | OSF: https://osf.io/agvxh/download (part0), https://osf.io/uv3mw/download (part1), https://osf.io/ehmw2/download (part2) |
| **NMA extra** | OSF: spike times (https://osf.io/4bjns/download), waveforms (https://osf.io/ugm9v/download), LFP (https://osf.io/kx3v9/download) |
| **sessions** | 39 sessions, 10 mice |
| **neurons** | ~30,000 total across all sessions |
| **brain regions** | 42 unique areas across 7 groups |
| **trial structure** | visual decision-making: contrast left/right → wheel turn → feedback |

### 1.2 data schema (NMA preprocessed format)

each session `dat = alldat[k]` contains:

| field | shape / type | description |
|-------|-------------|-------------|
| `spks` | `[n_neurons, n_trials, n_timebins]` | binned spike counts (10ms bins) |
| `brain_area` | `[n_neurons]` string | brain region label per neuron |
| `ccf` | `[n_neurons, 3]` float | allen CCF coordinates |
| `contrast_right` | `[n_trials]` float | right stimulus contrast (0, 0.25, 0.5, 1.0) |
| `contrast_left` | `[n_trials]` float | left stimulus contrast |
| `response` | `[n_trials]` int | choice: -1 (right), 0 (no-go), 1 (left) |
| `response_time` | `[n_trials]` float | response timing |
| `feedback_type` | `[n_trials]` int | +1 (correct/reward), -1 (incorrect/noise) |
| `gocue` | `[n_trials]` float | go cue timing |
| `mouse_name` | string | mouse identifier |
| `date_exp` | string | session date |
| `bin_size` | float | 0.01 (10ms) |
| `pupil` | `[3, n_trials, n_timebins]` | pupil area + x,y position |
| `wheel` | `[1, n_trials, n_timebins]` | wheel velocity |
| `face` | `[1, n_trials, n_timebins]` | face motion energy |
| `licks` | `[1, n_trials, n_timebins]` | lick detection |
| `trough_to_peak` | `[n_neurons]` float | waveform width (<=10 = fast-spiking) |
| `reaction_time` | `[n_trials, 2]` | RT + direction |
| `prev_reward` | `[n_trials]` float | previous trial reward time |

**temporal structure**: 250 time bins per trial (2.5 seconds), starting 500ms before stimulus onset (bin 0 = -0.5s, bin 50 = stimulus onset, bin 250 = +2.0s).

### 1.3 extra data files

| file | contents | shape per session |
|------|----------|-------------------|
| `steinmetz_st.npz` | exact spike times (not binned) | `ss[neuron, trial]` = list of times |
| `steinmetz_wav.npz` | waveform shapes | `waveform_w`, `waveform_u` |
| `steinmetz_lfp.npz` | local field potential per area | `lfp[n_areas, n_trials, n_timebins]` |

### 1.4 brain region groups

| group | label | areas |
|-------|-------|-------|
| visual cortex | vis ctx | VISa, VISam, VISl, VISp, VISpm, VISrl |
| thalamus | thal | CL, LD, LGd, LH, LP, MD, MG, PO, POL, PT, RT, SPF, TH, VAL, VPL, VPM |
| hippocampus | hipp | CA, CA1, CA2, CA3, DG, SUB, POST |
| non-visual cortex | other ctx | ACA, AUD, COA, DP, ILA, MOp, MOs, OLF, ORB, ORBm, PIR, PL, SSp, SSs, RSP, TT |
| midbrain | midbrain | APN, IC, MB, MRN, NB, PAG, RN, SCs, SCm, SCig, SCsg, ZI |
| basal ganglia | basal ganglia | ACB, CP, GPe, LS, LSc, LSr, MS, OT, SNr, SI |
| cortical subplate | cortical subplate | BLA, BMA, EP, EPd, MEA |

### 1.5 data messiness (why curation matters)

1. **unit quality heterogeneity**: many neurons have very low firing rates (<0.5 Hz) — these are likely noise or poorly isolated units
2. **waveform width variability**: `trough_to_peak` distinguishes putative excitatory (broad) from inhibitory (narrow) neurons, but some values are ambiguous
3. **brain region coverage varies by session**: different probe placements → different region combinations
4. **behavioral state confounds**: pupil dilation, face motion energy, and wheel velocity all correlate with neural activity independent of the visual stimulus
5. **lapse trials**: some trials show random-seeming choices unrelated to contrast — the mouse "lapses" in attention
6. **cross-session variability**: different mice, different neurons, different absolute firing rates
7. **passive trials**: end-of-session passive replay trials have different neural dynamics

### 1.6 train / validation / test split

| split | sessions | description |
|-------|----------|-------------|
| training | 0–29 (30 sessions) | available in `env_data/sessions/` |
| validation | 30–34 (5 sessions) | available in `env_data/sessions/` |
| hidden test | 35–38 (4 sessions) | neural data in `env_data/test_sessions_unlabeled/`, labels in `scoring_data/` |

**critical**: hidden test sessions come from DIFFERENT MICE than training sessions. this forces cross-subject generalization.

---

## 2. container structure

### 2.1 directory layout

```
/workdir/                           # student's working directory
├── env_data/
│   ├── sessions/                   # training + validation sessions
│   │   ├── session_00.npz
│   │   ├── session_01.npz
│   │   ├── ...
│   │   └── session_34.npz
│   ├── test_sessions_unlabeled/    # hidden test neural data (NO behavioral labels)
│   │   ├── session_35.npz
│   │   ├── session_36.npz
│   │   ├── session_37.npz
│   │   └── session_38.npz
│   ├── session_metadata.json       # mouse name, date, neuron counts per session
│   └── brain_region_groups.json    # region → group mapping
│
├── scoring_data/                   # HIDDEN from student
│   ├── test_labels/                # behavioral labels for test sessions
│   │   ├── session_35_labels.npz
│   │   ├── session_36_labels.npz
│   │   ├── session_37_labels.npz
│   │   └── session_38_labels.npz
│   ├── scoring_step2.py
│   ├── scoring_step3.py
│   └── scoring_step4.py
│
└── (student's work products)
```

### 2.2 session file format (env_data/sessions/)

each `session_XX.npz` contains:

```python
{
    "spks": np.ndarray,           # [n_neurons, n_trials, 250] — binned spikes
    "brain_area": np.ndarray,     # [n_neurons] — string labels
    "contrast_left": np.ndarray,  # [n_trials] — 0, 0.25, 0.5, 1.0
    "contrast_right": np.ndarray, # [n_trials]
    "response": np.ndarray,       # [n_trials] — -1, 0, 1
    "feedback_type": np.ndarray,  # [n_trials] — -1, 1
    "response_time": np.ndarray,  # [n_trials]
    "gocue": np.ndarray,          # [n_trials]
    "mouse_name": str,
    "date_exp": str,
    "trough_to_peak": np.ndarray, # [n_neurons] — waveform width
    "pupil": np.ndarray,          # [3, n_trials, 250]
    "wheel": np.ndarray,          # [1, n_trials, 250]
    "face": np.ndarray,           # [1, n_trials, 250]
    "licks": np.ndarray,          # [1, n_trials, 250]
    "reaction_time": np.ndarray,  # [n_trials, 2]
}
```

### 2.3 unlabeled test session format (env_data/test_sessions_unlabeled/)

same as above but **without** `response`, `feedback_type`, `response_time`, and `reaction_time`. these fields are in `scoring_data/test_labels/`.

### 2.4 session_metadata.json

```json
{
    "sessions": [
        {
            "session_id": 0,
            "mouse_name": "Cori",
            "date_exp": "2016-12-14",
            "n_neurons": 698,
            "n_trials": 340,
            "brain_areas": ["VISp", "VISl", "VISrl", "VISam", "VISpm", "LGd", "LP", "DG", "CA1", "SUB"],
            "split": "train"
        },
        ...
    ],
    "mice": {
        "Cori": {"sessions": [0, 1, 2], "split": "train"},
        "Forssmann": {"sessions": [3, 4, 5, 6], "split": "train"},
        ...
    }
}
```

---

## 3. task specification: 4 progressive steps

### 3.1 step 1: data exploration & quality assessment

**judge**: ExactMatchJudge

**instructions**:
```
You have access to neural recording data in {env_data_dir}/sessions/. This is the
Steinmetz et al. (2019) Neuropixels dataset — simultaneous recordings from multiple
brain areas in mice performing a visual decision-making task.

Each session file (session_XX.npz) contains spike data, brain area labels,
stimulus contrasts, behavioral responses, and auxiliary signals (pupil, wheel, face).

Explore the data and answer the following questions using submit_answers:

1. "num_sessions": How many total sessions are available (training + validation)?
2. "total_neurons": Total number of neurons across all available sessions.
3. "num_brain_areas": How many unique brain areas appear across all sessions?
4. "top3_areas": The 3 brain areas with the most neurons across all sessions
   (comma-separated, descending order, e.g. "VISp,MOs,DG").
5. "num_stimulus_conditions": How many unique (contrast_left, contrast_right)
   combinations exist? Count each unique pair.
6. "mean_firing_rate": Mean firing rate across ALL neurons in ALL sessions,
   in Hz. The data is binned at 10ms. Round to 1 decimal place.
7. "pct_low_firing": Percentage of neurons with mean firing rate < 0.5 Hz
   across all time bins and trials. Round to 1 decimal place.
8. "num_correct_trials_session0": Number of correct trials in session 0.
   A trial is correct if feedback_type == 1.
```

**answer computation** (for setup_data.py to precompute):
```python
import numpy as np
import os
import json

def compute_step1_answers(sessions_dir):
    sessions = sorted([f for f in os.listdir(sessions_dir) if f.endswith(".npz")])

    total_neurons = 0
    all_areas = set()
    area_counts = {}
    all_firing_rates = []
    n_correct_session0 = None
    stimulus_conditions = set()

    for i, fname in enumerate(sessions):
        data = np.load(os.path.join(sessions_dir, fname), allow_pickle=True)
        spks = data["spks"]
        n_neurons = spks.shape[0]
        total_neurons += n_neurons

        areas = data["brain_area"]
        for a in areas:
            a_str = str(a)
            all_areas.add(a_str)
            area_counts[a_str] = area_counts.get(a_str, 0) + 1

        # firing rate: spks is [n_neurons, n_trials, n_timebins], bins are 10ms
        # mean across trials and timebins, convert to Hz (divide by 0.01)
        mean_fr_per_neuron = spks.mean(axis=(1, 2)) / 0.01  # spikes per 10ms bin → Hz
        all_firing_rates.extend(mean_fr_per_neuron.tolist())

        # stimulus conditions
        cl = data["contrast_left"]
        cr = data["contrast_right"]
        for j in range(len(cl)):
            stimulus_conditions.add((float(cl[j]), float(cr[j])))

        # correct trials in session 0
        if i == 0:
            n_correct_session0 = int((data["feedback_type"] == 1).sum())

    # top 3 areas by neuron count
    sorted_areas = sorted(area_counts.items(), key=lambda x: -x[1])
    top3 = ",".join([a[0] for a in sorted_areas[:3]])

    mean_fr = round(np.mean(all_firing_rates), 1)
    pct_low = round(100 * np.mean(np.array(all_firing_rates) < 0.5), 1)

    return {
        "num_sessions": len(sessions),
        "total_neurons": total_neurons,
        "num_brain_areas": len(all_areas),
        "top3_areas": top3,
        "num_stimulus_conditions": len(stimulus_conditions),
        "mean_firing_rate": mean_fr,
        "pct_low_firing": pct_low,
        "num_correct_trials_session0": n_correct_session0,
    }
```

### 3.2 step 2: neural data quality control

**judge**: ExecutableJudge (continuous)

**instructions**:
```
Many neurons in the dataset are low-quality: they fire too rarely, have unstable
activity across the session, or show signs of multi-unit contamination. Your task
is to curate the neural data by assigning quality labels to each neuron.

For each session, produce a quality assessment.

Save to /workdir/unit_quality.npz:
- "quality_labels": a dict (pickled) mapping session_id (int) to a numpy array
  of shape [n_neurons] with integer values:
    0 = bad (exclude from downstream analysis)
    1 = acceptable (include with caveats)
    2 = good (high-quality unit)
- "quality_criteria": a string describing your criteria

Criteria to consider:
- Firing rate: neurons with < 0.5 Hz mean firing rate across all trials may be noise
- Stability: firing rate should not change dramatically between the first half
  and second half of trials (non-stationarity = drift/electrode movement)
- Trial presence: neuron should fire on a reasonable fraction of trials
  (silent on >80% of trials is suspicious)
- Waveform width: use trough_to_peak to distinguish cell types;
  very narrow or very wide waveforms may indicate artifacts
- Task modulation: neurons that show NO difference in firing rate between
  stimulus conditions may not be task-relevant

Your quality labels will be used to filter neurons for downstream decoding.
Better filtering → better decoding accuracy → higher score.
```

**scoring script** (`scoring_step2.py`):
```python
#!/usr/bin/env python3
"""
Scoring for Step 2: Neural Data Quality Control

Score = improvement in decoding accuracy when using student's filtered neurons
vs using ALL neurons (no filtering).

Logic:
1. Load student's quality labels
2. For each session: decode left/right choice using student's "good" neurons
3. Compare to baseline decoder using ALL neurons
4. Score = f(improvement)
"""
import numpy as np
import os
import sys
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def load_session(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def decode_session(spks, response, neuron_mask, time_start=50, time_end=150):
    """
    Decode left vs right choice from neural activity in stimulus window.

    Args:
        spks: [n_neurons, n_trials, n_timebins]
        response: [n_trials] with values -1, 0, 1
        neuron_mask: boolean [n_neurons]
        time_start, time_end: time bin range (50-150 = 0-1s post-stimulus)

    Returns:
        mean balanced accuracy from 5-fold CV, or None if insufficient data
    """
    # exclude no-go trials
    valid = response != 0
    if valid.sum() < 20:
        return None

    y = (response[valid] > 0).astype(int)  # 1=left, 0=right

    if neuron_mask.sum() < 3:
        return None

    # features: mean firing rate in stimulus window per neuron
    X = spks[neuron_mask][:, valid, time_start:time_end].mean(axis=2).T
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if X.shape[0] < 20 or X.shape[1] < 3:
        return None

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    try:
        cv_scores = cross_val_score(
            model, X_s, y, cv=min(5, min(y.sum(), (1-y).sum())),
            scoring="balanced_accuracy"
        )
        return float(np.mean(cv_scores))
    except Exception:
        return None


def score_quality(quality_path, sessions_dir, output_path):
    # load student's quality labels
    try:
        data = np.load(quality_path, allow_pickle=True)
        quality_labels = data["quality_labels"].item()
    except Exception as e:
        json.dump(
            {"score": 0.0, "metadata": {"error": f"Failed to load quality file: {e}"}},
            open(output_path, "w")
        )
        return

    session_files = sorted([
        f for f in os.listdir(sessions_dir) if f.endswith(".npz")
    ])

    filtered_scores = []
    baseline_scores = []

    for fname in session_files:
        session_id = int(fname.replace("session_", "").replace(".npz", ""))
        session = load_session(os.path.join(sessions_dir, fname))
        spks = session["spks"]
        response = session["response"]
        n_neurons = spks.shape[0]

        # baseline: all neurons
        all_mask = np.ones(n_neurons, dtype=bool)
        baseline_acc = decode_session(spks, response, all_mask)
        if baseline_acc is not None:
            baseline_scores.append(baseline_acc)

        # student's filtering
        if session_id in quality_labels:
            labels = np.array(quality_labels[session_id])
            if len(labels) != n_neurons:
                # wrong size — treat as all-pass
                good_mask = all_mask
            else:
                good_mask = labels >= 1  # include "acceptable" and "good"
        else:
            good_mask = all_mask

        if good_mask.sum() < 3:
            # too aggressive filtering
            filtered_scores.append(0.5)  # chance
        else:
            filtered_acc = decode_session(spks, response, good_mask)
            if filtered_acc is not None:
                filtered_scores.append(filtered_acc)

    if len(filtered_scores) == 0 or len(baseline_scores) == 0:
        json.dump(
            {"score": 0.0, "metadata": {"error": "No sessions could be scored"}},
            open(output_path, "w")
        )
        return

    mean_filtered = np.mean(filtered_scores)
    mean_baseline = np.mean(baseline_scores)
    improvement = mean_filtered - mean_baseline

    # score mapping:
    #   improvement < -0.05 → 0.0 (filtering made things much worse)
    #   improvement = -0.05 → 0.0
    #   improvement = 0.0   → 0.3  (no improvement but no harm)
    #   improvement = 0.05  → 0.6  (meaningful improvement)
    #   improvement >= 0.10 → 1.0  (excellent filtering)
    if improvement < -0.05:
        score = 0.0
    elif improvement < 0.0:
        score = (improvement + 0.05) / 0.05 * 0.3
    elif improvement < 0.05:
        score = 0.3 + (improvement / 0.05) * 0.3
    else:
        score = min(0.6 + (improvement - 0.05) / 0.05 * 0.4, 1.0)

    result = {
        "score": round(score, 4),
        "metadata": {
            "mean_accuracy_filtered": round(mean_filtered, 4),
            "mean_accuracy_baseline": round(mean_baseline, 4),
            "improvement": round(improvement, 4),
            "n_sessions_scored": len(filtered_scores),
            "per_session_filtered": [round(s, 4) for s in filtered_scores],
            "per_session_baseline": [round(s, 4) for s in baseline_scores],
        }
    }
    json.dump(result, open(output_path, "w"))


if __name__ == "__main__":
    score_quality(sys.argv[1], sys.argv[2], sys.argv[3])
```

### 3.3 step 3: feature engineering for cross-area decoding

**judge**: ExecutableJudge (continuous)

**instructions**:
```
Build features for decoding the mouse's decision from neural activity across
brain regions.

The key challenge: different brain regions encode decisions at different times
and with different dynamics. Visual cortex (VISp, VISl) responds early to the
stimulus. Frontal motor areas (MOs, MOp) encode the decision later. Midbrain
areas (SC) integrate information. A good feature pipeline must handle this
temporal heterogeneity.

Save to /workdir/neural_features.npz:
- "X_train": numpy array [n_trials, n_features] — features from training sessions
- "y_train": numpy array [n_trials] — decision labels (0=right, 1=left)
- "X_val": numpy array [n_trials_val, n_features] — features from validation sessions
- "y_val": numpy array [n_trials_val] — validation labels
- "feature_names": numpy array of strings describing each feature
- "session_ids_train": numpy array [n_trials] — which session each training trial came from
- "session_ids_val": numpy array [n_trials_val] — which session each validation trial came from

IMPORTANT:
- Exclude no-go trials (response == 0)
- Labels: 0 = right choice (response == -1), 1 = left choice (response == 1)
- Features must be the SAME dimensionality for all trials regardless of
  which brain areas are present in a session

Feature ideas to explore:
- Per-region mean firing rates in different time windows:
  * pre-stimulus (bins 0-50 = -0.5 to 0s)
  * early stimulus (bins 50-100 = 0 to 0.5s)
  * late stimulus (bins 100-150 = 0.5 to 1.0s)
  * response window (bins 150-200 = 1.0 to 1.5s)
- Stimulus contrast features (contrast_left, contrast_right, contrast_diff)
- Population activity: PCA on population vectors per time window
- Single-neuron selectivity indices (choice preference per neuron)
- Cross-region correlation patterns
- Behavioral signals: pupil, wheel, face motion

HINT: The contrast difference (contrast_left - contrast_right) is the strongest
single predictor of choice. But neural activity provides additional information,
especially on ambiguous trials where contrasts are equal on both sides.

Your features will be evaluated by training a logistic regression on X_train/y_train
and testing on hidden sessions.
```

**scoring script** (`scoring_step3.py`):
```python
#!/usr/bin/env python3
"""
Scoring for Step 3: Feature Engineering

Score = balanced accuracy of logistic regression trained on student's features,
evaluated on hidden test sessions.

The student provides X_train, y_train. We:
1. Train LogisticRegression on the student's features
2. Apply the SAME feature extraction to hidden test sessions
   (this is the tricky part — we need the student's feature pipeline to work
    on unseen sessions, so we evaluate on validation features they provided)
3. Score = balanced accuracy on validation set (which we verify against
   hidden test set scores for anti-gaming)
"""
import numpy as np
import os
import sys
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score


def score_features(features_path, hidden_sessions_dir, hidden_labels_dir, output_path):
    try:
        data = np.load(features_path, allow_pickle=True)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
    except Exception as e:
        json.dump(
            {"score": 0.0, "metadata": {"error": f"Failed to load features: {e}"}},
            open(output_path, "w")
        )
        return

    # validate shapes
    if X_train.ndim != 2 or y_train.ndim != 1:
        json.dump(
            {"score": 0.0, "metadata": {"error": "X_train must be 2D, y_train must be 1D"}},
            open(output_path, "w")
        )
        return

    if X_train.shape[0] != y_train.shape[0]:
        json.dump(
            {"score": 0.0, "metadata": {"error": "X_train and y_train row count mismatch"}},
            open(output_path, "w")
        )
        return

    if X_val.shape[1] != X_train.shape[1]:
        json.dump(
            {"score": 0.0, "metadata": {"error": "X_train and X_val feature count mismatch"}},
            open(output_path, "w")
        )
        return

    # check for degenerate inputs
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    if X_train.shape[0] < 50:
        json.dump(
            {"score": 0.0, "metadata": {"error": "Too few training samples"}},
            open(output_path, "w")
        )
        return

    # anti-gaming: check for constant predictions / trivial features
    n_unique_labels = len(np.unique(y_train))
    if n_unique_labels < 2:
        json.dump(
            {"score": 0.0, "metadata": {"error": "y_train must contain both classes"}},
            open(output_path, "w")
        )
        return

    # check feature variance — at least some features must vary
    feature_vars = np.var(X_train, axis=0)
    n_nonzero_var = (feature_vars > 1e-10).sum()
    if n_nonzero_var < 1:
        json.dump(
            {"score": 0.0, "metadata": {"error": "All features are constant"}},
            open(output_path, "w")
        )
        return

    # train model on student's features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    try:
        model.fit(X_train_s, y_train)
    except Exception as e:
        json.dump(
            {"score": 0.0, "metadata": {"error": f"Model training failed: {e}"}},
            open(output_path, "w")
        )
        return

    # evaluate on validation set
    y_pred_val = model.predict(X_val_s)
    val_acc = balanced_accuracy_score(y_val, y_pred_val)

    # also compute train accuracy for overfitting check
    y_pred_train = model.predict(X_train_s)
    train_acc = balanced_accuracy_score(y_train, y_pred_train)

    # score mapping:
    #   balanced_accuracy = 0.50 (chance) → score 0.0
    #   balanced_accuracy = 0.60 → score 0.2 (basic features work)
    #   balanced_accuracy = 0.70 → score 0.5 (good features)
    #   balanced_accuracy = 0.80 → score 0.8 (excellent features)
    #   balanced_accuracy = 0.85+ → score 1.0 (near-optimal)

    raw = val_acc - 0.5  # above chance
    if raw <= 0:
        score = 0.0
    elif raw < 0.10:
        score = raw / 0.10 * 0.2
    elif raw < 0.20:
        score = 0.2 + (raw - 0.10) / 0.10 * 0.3
    elif raw < 0.30:
        score = 0.5 + (raw - 0.20) / 0.10 * 0.3
    else:
        score = min(0.8 + (raw - 0.30) / 0.05 * 0.2, 1.0)

    # penalize extreme overfitting (train >> val by >15%)
    overfit_gap = train_acc - val_acc
    if overfit_gap > 0.15:
        penalty = min((overfit_gap - 0.15) / 0.15, 0.5)  # up to 50% penalty
        score = score * (1 - penalty)

    result = {
        "score": round(score, 4),
        "metadata": {
            "val_balanced_accuracy": round(val_acc, 4),
            "train_balanced_accuracy": round(train_acc, 4),
            "overfit_gap": round(overfit_gap, 4),
            "n_features": int(X_train.shape[1]),
            "n_train_samples": int(X_train.shape[0]),
            "n_val_samples": int(X_val.shape[0]),
            "n_nonzero_variance_features": int(n_nonzero_var),
        }
    }
    json.dump(result, open(output_path, "w"))


if __name__ == "__main__":
    score_features(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
```

### 3.4 step 4: cross-session generalization (FINAL)

**judge**: ExecutableJudge (continuous)

**instructions**:
```
Final challenge: produce predictions for hidden test sessions from DIFFERENT MICE.

The hardest problem in neural decoding is generalizing across subjects. Each mouse
has different neurons in different positions. Your pipeline must extract features
that are invariant to the specific neurons recorded.

Test session data is available in {env_data_dir}/test_sessions_unlabeled/. These
files contain neural data (spks, brain_area, contrast_left, contrast_right, etc.)
but NO behavioral labels (response, feedback_type).

Save to /workdir/cross_session_predictions.npz:
- "predictions": a dict (pickled) mapping session_id (int) to numpy array of
  predicted decisions. Each array has shape [n_trials] with values:
    0 = predicted right choice
    1 = predicted left choice
  Include predictions for ALL trials (including what would be no-go).
- "prediction_method": a string describing your approach

Strategies to consider:
- Region-level features: aggregate neural activity within each brain area so that
  the specific neurons don't matter, only which areas are present
- Cross-session normalization: z-score firing rates per session so absolute
  magnitudes don't matter
- Contrast as a feature: contrast_left and contrast_right are available in the
  test sessions and are highly predictive of choice
- Multi-region temporal features: combine early visual activity with late
  motor/frontal activity
- Train on ALL available labeled sessions, predict on unlabeled test sessions

HINT: A model using ONLY contrast difference achieves ~65-70% balanced accuracy.
Neural features should push this above 75%. The best solutions combine contrast
information with region-level neural features and cross-session normalization.

Your predictions will be scored against the true behavioral responses on the
hidden test sessions.
```

**scoring script** (`scoring_step4.py`):
```python
#!/usr/bin/env python3
"""
Scoring for Step 4: Cross-Session Generalization

Score = balanced accuracy of student's predictions on hidden test sessions.

Anti-gaming checks:
- Constant predictions → score 0
- Predictions don't vary across sessions → score penalty
- Must predict for all test sessions
"""
import numpy as np
import os
import sys
import json
from sklearn.metrics import balanced_accuracy_score


def score_predictions(predictions_path, test_labels_dir, output_path):
    try:
        data = np.load(predictions_path, allow_pickle=True)
        predictions = data["predictions"].item()
    except Exception as e:
        json.dump(
            {"score": 0.0, "metadata": {"error": f"Failed to load predictions: {e}"}},
            open(output_path, "w")
        )
        return

    if not isinstance(predictions, dict):
        json.dump(
            {"score": 0.0, "metadata": {"error": "predictions must be a dict"}},
            open(output_path, "w")
        )
        return

    # load true labels
    label_files = sorted([
        f for f in os.listdir(test_labels_dir) if f.endswith(".npz")
    ])

    if len(label_files) == 0:
        json.dump(
            {"score": 0.0, "metadata": {"error": "No label files found"}},
            open(output_path, "w")
        )
        return

    all_preds = []
    all_labels = []
    per_session_results = {}

    for fname in label_files:
        session_id = int(
            fname.replace("session_", "").replace("_labels.npz", "")
        )

        labels_data = np.load(os.path.join(test_labels_dir, fname), allow_pickle=True)
        true_response = labels_data["response"]

        if session_id not in predictions:
            per_session_results[session_id] = {
                "status": "missing",
                "balanced_accuracy": 0.0,
            }
            continue

        pred = np.array(predictions[session_id])

        if len(pred) != len(true_response):
            per_session_results[session_id] = {
                "status": f"length_mismatch: pred={len(pred)}, true={len(true_response)}",
                "balanced_accuracy": 0.0,
            }
            continue

        # convert true responses: -1 → 0 (right), 1 → 1 (left), 0 → skip
        valid = true_response != 0
        if valid.sum() < 10:
            per_session_results[session_id] = {
                "status": "too_few_valid_trials",
                "balanced_accuracy": 0.5,
            }
            continue

        true_binary = (true_response[valid] > 0).astype(int)
        pred_valid = np.array(pred[valid]).astype(int)

        # clamp predictions to {0, 1}
        pred_valid = np.clip(pred_valid, 0, 1)

        acc = balanced_accuracy_score(true_binary, pred_valid)
        per_session_results[session_id] = {
            "status": "scored",
            "balanced_accuracy": round(float(acc), 4),
            "n_valid_trials": int(valid.sum()),
            "pred_mean": round(float(pred_valid.mean()), 4),
        }

        all_preds.extend(pred_valid.tolist())
        all_labels.extend(true_binary.tolist())

    if len(all_preds) == 0:
        json.dump(
            {"score": 0.0, "metadata": {
                "error": "No valid predictions",
                "per_session": per_session_results,
            }},
            open(output_path, "w")
        )
        return

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    overall_acc = balanced_accuracy_score(all_labels, all_preds)

    # anti-gaming: check for constant predictions
    n_unique_preds = len(np.unique(all_preds))
    if n_unique_preds < 2:
        json.dump(
            {"score": 0.0, "metadata": {
                "error": "Constant predictions detected",
                "pred_unique_values": int(n_unique_preds),
                "overall_balanced_accuracy": round(overall_acc, 4),
            }},
            open(output_path, "w")
        )
        return

    # anti-gaming: check prediction balance (shouldn't be >90% one class)
    pred_mean = all_preds.mean()
    if pred_mean < 0.05 or pred_mean > 0.95:
        # extreme bias — partial penalty
        bias_penalty = 0.3
    else:
        bias_penalty = 0.0

    # score mapping:
    #   balanced_accuracy = 0.50 (chance) → score 0.0
    #   balanced_accuracy = 0.60 → score 0.2
    #   balanced_accuracy = 0.70 → score 0.4
    #   balanced_accuracy = 0.75 → score 0.6
    #   balanced_accuracy = 0.80 → score 0.8
    #   balanced_accuracy = 0.85+ → score 1.0

    raw = overall_acc - 0.5
    if raw <= 0:
        score = 0.0
    elif raw < 0.10:
        score = raw / 0.10 * 0.2
    elif raw < 0.20:
        score = 0.2 + (raw - 0.10) / 0.10 * 0.2
    elif raw < 0.25:
        score = 0.4 + (raw - 0.20) / 0.05 * 0.2
    elif raw < 0.30:
        score = 0.6 + (raw - 0.25) / 0.05 * 0.2
    else:
        score = min(0.8 + (raw - 0.30) / 0.05 * 0.2, 1.0)

    score = max(0.0, score - bias_penalty)

    n_sessions_scored = sum(
        1 for v in per_session_results.values() if v.get("status") == "scored"
    )

    result = {
        "score": round(score, 4),
        "metadata": {
            "overall_balanced_accuracy": round(overall_acc, 4),
            "n_sessions_scored": n_sessions_scored,
            "n_sessions_total": len(label_files),
            "n_trials_total": len(all_preds),
            "prediction_mean": round(float(pred_mean), 4),
            "bias_penalty": bias_penalty,
            "per_session": per_session_results,
        }
    }
    json.dump(result, open(output_path, "w"))


if __name__ == "__main__":
    score_predictions(sys.argv[1], sys.argv[2], sys.argv[3])
```

---

## 4. setup_data.py

```python
#!/usr/bin/env python3
"""
Setup script for E46: Steinmetz Multi-Area Neural Data Curation

Downloads the preprocessed Steinmetz dataset from OSF, splits into
train/validation/hidden-test, and prepares the container filesystem.

Usage:
    python setup_data.py --output_dir /path/to/container/root
"""
import argparse
import json
import os
import sys
import shutil
import numpy as np

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)


# ============================================================
# DATA SOURCES
# ============================================================

MAIN_DATA_URLS = [
    ("steinmetz_part0.npz", "https://osf.io/agvxh/download"),
    ("steinmetz_part1.npz", "https://osf.io/uv3mw/download"),
    ("steinmetz_part2.npz", "https://osf.io/ehmw2/download"),
]

EXTRA_DATA_URLS = [
    ("steinmetz_wav.npz", "https://osf.io/ugm9v/download"),
]

TRAIN_SESSIONS = list(range(0, 30))      # sessions 0-29
VAL_SESSIONS = list(range(30, 35))        # sessions 30-34
TEST_SESSIONS = list(range(35, 39))       # sessions 35-38

BRAIN_REGION_GROUPS = {
    "visual_cortex": ["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"],
    "thalamus": [
        "CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL",
        "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM",
    ],
    "hippocampus": ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"],
    "non_visual_cortex": [
        "ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF",
        "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP", "TT",
    ],
    "midbrain": [
        "APN", "IC", "MB", "MRN", "NB", "PAG", "RN",
        "SCs", "SCm", "SCig", "SCsg", "ZI",
    ],
    "basal_ganglia": [
        "ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI",
    ],
    "cortical_subplate": ["BLA", "BMA", "EP", "EPd", "MEA"],
}


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(url, dest_path):
    """Download file with progress indication."""
    if os.path.isfile(dest_path):
        print(f"  [skip] {dest_path} already exists")
        return
    print(f"  [download] {url}")
    print(f"           → {dest_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  [{pct:5.1f}%] {downloaded}/{total} bytes", end="")
    print()


def download_all(cache_dir):
    """Download all data files to cache directory."""
    os.makedirs(cache_dir, exist_ok=True)

    for fname, url in MAIN_DATA_URLS + EXTRA_DATA_URLS:
        download_file(url, os.path.join(cache_dir, fname))


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_all_sessions(cache_dir):
    """Load all 39 sessions from the 3-part NPZ files."""
    alldat = np.array([])
    for fname, _ in MAIN_DATA_URLS:
        path = os.path.join(cache_dir, fname)
        alldat = np.hstack((
            alldat,
            np.load(path, allow_pickle=True)["dat"]
        ))
    print(f"  loaded {len(alldat)} sessions")

    # also load waveform data for trough_to_peak
    wav_path = os.path.join(cache_dir, "steinmetz_wav.npz")
    dat_wav = np.load(wav_path, allow_pickle=True)["dat"]

    return alldat, dat_wav


# ============================================================
# SAVE SESSION FILES
# ============================================================

def save_session(dat, dat_wav_session, session_id, output_path, include_labels=True):
    """Save a single session as a standardized NPZ file."""
    save_dict = {
        "spks": dat["spks"],
        "brain_area": dat["brain_area"],
        "contrast_left": dat["contrast_left"],
        "contrast_right": dat["contrast_right"],
        "gocue": dat["gocue"],
        "mouse_name": str(dat["mouse_name"]),
        "date_exp": str(dat["date_exp"]),
        "trough_to_peak": dat_wav_session["trough_to_peak"],
        "pupil": dat["pupil"],
        "wheel": dat["wheel"],
        "face": dat["face"],
        "licks": dat["licks"],
    }

    if include_labels:
        save_dict["response"] = dat["response"]
        save_dict["feedback_type"] = dat["feedback_type"]
        save_dict["response_time"] = dat["response_time"]
        save_dict["reaction_time"] = dat["reaction_time"]

    np.savez_compressed(output_path, **save_dict)


def save_labels(dat, output_path):
    """Save just the behavioral labels for hidden test sessions."""
    np.savez_compressed(output_path,
        response=dat["response"],
        feedback_type=dat["feedback_type"],
        response_time=dat["response_time"],
        reaction_time=dat["reaction_time"],
    )


# ============================================================
# BUILD METADATA
# ============================================================

def build_metadata(alldat):
    """Build session_metadata.json."""
    sessions_meta = []
    mice_meta = {}

    for i, dat in enumerate(alldat):
        mouse = str(dat["mouse_name"])
        areas = list(np.unique(dat["brain_area"]).astype(str))

        if i in TRAIN_SESSIONS:
            split = "train"
        elif i in VAL_SESSIONS:
            split = "validation"
        else:
            split = "test"

        session_info = {
            "session_id": i,
            "mouse_name": mouse,
            "date_exp": str(dat["date_exp"]),
            "n_neurons": int(dat["spks"].shape[0]),
            "n_trials": int(dat["spks"].shape[1]),
            "brain_areas": areas,
            "split": split,
        }
        sessions_meta.append(session_info)

        if mouse not in mice_meta:
            mice_meta[mouse] = {"sessions": [], "split": set()}
        mice_meta[mouse]["sessions"].append(i)
        mice_meta[mouse]["split"].add(split)

    # convert sets to lists for JSON serialization
    for mouse in mice_meta:
        mice_meta[mouse]["split"] = sorted(mice_meta[mouse]["split"])

    return {"sessions": sessions_meta, "mice": mice_meta}


# ============================================================
# PRECOMPUTE STEP 1 ANSWERS
# ============================================================

def compute_reference_answers(alldat, sessions_dir):
    """Compute exact answers for Step 1."""
    # only count sessions available to student (train + val = 0-34)
    available_sessions = TRAIN_SESSIONS + VAL_SESSIONS

    total_neurons = 0
    all_areas = set()
    area_counts = {}
    all_firing_rates = []
    stimulus_conditions = set()
    n_correct_session0 = None

    for i in available_sessions:
        dat = alldat[i]
        spks = dat["spks"]
        n_neurons = spks.shape[0]
        total_neurons += n_neurons

        for a in dat["brain_area"]:
            a_str = str(a)
            all_areas.add(a_str)
            area_counts[a_str] = area_counts.get(a_str, 0) + 1

        # firing rate: spks is [n_neurons, n_trials, n_timebins]
        # each bin = 10ms, so mean spike count per bin / 0.01 = Hz
        mean_fr = spks.mean(axis=(1, 2)) / 0.01
        all_firing_rates.extend(mean_fr.tolist())

        cl = dat["contrast_left"]
        cr = dat["contrast_right"]
        for j in range(len(cl)):
            stimulus_conditions.add((round(float(cl[j]), 2), round(float(cr[j]), 2)))

        if i == 0:
            n_correct_session0 = int((dat["feedback_type"] == 1).sum())

    sorted_areas = sorted(area_counts.items(), key=lambda x: -x[1])
    top3 = ",".join([a[0] for a in sorted_areas[:3]])

    fr_array = np.array(all_firing_rates)
    mean_fr = round(float(np.mean(fr_array)), 1)
    pct_low = round(float(100 * np.mean(fr_array < 0.5)), 1)

    answers = {
        "num_sessions": str(len(available_sessions)),
        "total_neurons": str(total_neurons),
        "num_brain_areas": str(len(all_areas)),
        "top3_areas": top3,
        "num_stimulus_conditions": str(len(stimulus_conditions)),
        "mean_firing_rate": str(mean_fr),
        "pct_low_firing": str(pct_low),
        "num_correct_trials_session0": str(n_correct_session0),
    }

    return answers


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Setup E46 Steinmetz pm_env data"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Root directory for the container filesystem"
    )
    parser.add_argument(
        "--cache_dir", default="/tmp/steinmetz_cache",
        help="Directory to cache downloaded files"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    cache_dir = args.cache_dir

    # create directory structure
    env_data_dir = os.path.join(output_dir, "env_data")
    sessions_dir = os.path.join(env_data_dir, "sessions")
    test_unlabeled_dir = os.path.join(env_data_dir, "test_sessions_unlabeled")
    scoring_dir = os.path.join(output_dir, "scoring_data")
    test_labels_dir = os.path.join(scoring_dir, "test_labels")

    for d in [sessions_dir, test_unlabeled_dir, scoring_dir, test_labels_dir]:
        os.makedirs(d, exist_ok=True)

    # step 1: download
    print("=== downloading data ===")
    download_all(cache_dir)

    # step 2: load
    print("=== loading sessions ===")
    alldat, dat_wav = load_all_sessions(cache_dir)

    # step 3: save training + validation sessions
    print("=== saving training + validation sessions ===")
    for i in TRAIN_SESSIONS + VAL_SESSIONS:
        dest = os.path.join(sessions_dir, f"session_{i:02d}.npz")
        save_session(alldat[i], dat_wav[i], i, dest, include_labels=True)
        print(f"  session {i:02d}: {alldat[i]['spks'].shape[0]} neurons, "
              f"{alldat[i]['spks'].shape[1]} trials")

    # step 4: save test sessions (unlabeled for student, labels for scoring)
    print("=== saving test sessions ===")
    for i in TEST_SESSIONS:
        # unlabeled version → env_data
        unlabeled_dest = os.path.join(
            test_unlabeled_dir, f"session_{i:02d}.npz"
        )
        save_session(
            alldat[i], dat_wav[i], i, unlabeled_dest, include_labels=False
        )

        # labels → scoring_data
        labels_dest = os.path.join(
            test_labels_dir, f"session_{i:02d}_labels.npz"
        )
        save_labels(alldat[i], labels_dest)

        print(f"  session {i:02d}: {alldat[i]['spks'].shape[0]} neurons, "
              f"{alldat[i]['spks'].shape[1]} trials (test, unlabeled)")

    # step 5: save metadata
    print("=== building metadata ===")
    metadata = build_metadata(alldat)
    with open(os.path.join(env_data_dir, "session_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # step 6: save brain region groups
    with open(os.path.join(env_data_dir, "brain_region_groups.json"), "w") as f:
        json.dump(BRAIN_REGION_GROUPS, f, indent=2)

    # step 7: compute reference answers for step 1
    print("=== computing step 1 reference answers ===")
    answers = compute_reference_answers(alldat, sessions_dir)
    with open(os.path.join(scoring_dir, "step1_answers.json"), "w") as f:
        json.dump(answers, f, indent=2)
    print(f"  answers: {json.dumps(answers, indent=2)}")

    # step 8: copy scoring scripts
    print("=== setup complete ===")

    # print summary
    total_neurons = sum(
        alldat[i]["spks"].shape[0]
        for i in TRAIN_SESSIONS + VAL_SESSIONS
    )
    total_test_neurons = sum(
        alldat[i]["spks"].shape[0] for i in TEST_SESSIONS
    )
    print(f"\n  training+val sessions: {len(TRAIN_SESSIONS) + len(VAL_SESSIONS)}")
    print(f"  hidden test sessions: {len(TEST_SESSIONS)}")
    print(f"  total neurons (train+val): {total_neurons}")
    print(f"  total neurons (test): {total_test_neurons}")
    print(f"  env_data: {env_data_dir}")
    print(f"  scoring_data: {scoring_dir}")


if __name__ == "__main__":
    main()
```

---

## 5. tasks.py

```python
#!/usr/bin/env python3
"""
Task definitions for E46: Steinmetz Multi-Area Neural Data Curation

Defines the 4 progressive steps with their judges and scoring.
"""

TASK_CONFIG = {
    "environment_id": "E46",
    "environment_name": "Steinmetz Multi-Area Neural Data Curation",
    "description": (
        "Curate multi-area Neuropixels recordings from the Steinmetz et al. (2019) "
        "visual decision-making experiment. Progressive steps: explore data, "
        "quality-control neurons, engineer cross-area features, and generalize "
        "decoding across mice."
    ),
    "time_limit_seconds": 14400,  # 4 hours
    "hardware": "cpu",
    "network": False,
    "container_image": "e46-steinmetz:latest",
    "steps": [
        # ====================================================
        # STEP 1: Data Exploration & Quality Assessment
        # ====================================================
        {
            "step_id": 1,
            "name": "Data Exploration & Quality Assessment",
            "instructions": (
                "You have access to neural recording data in {env_data_dir}/sessions/. "
                "This is the Steinmetz et al. (2019) Neuropixels dataset — simultaneous "
                "recordings from multiple brain areas in mice performing a visual "
                "decision-making task.\n\n"
                "Each session file (session_XX.npz) contains spike data, brain area labels, "
                "stimulus contrasts, behavioral responses, and auxiliary signals.\n\n"
                "Explore the data and answer the following questions using submit_answers:\n\n"
                '1. "num_sessions": How many total sessions are available (training + validation)?\n'
                '2. "total_neurons": Total number of neurons across all available sessions.\n'
                '3. "num_brain_areas": How many unique brain areas appear across all sessions?\n'
                '4. "top3_areas": The 3 brain areas with the most neurons across all sessions '
                '(comma-separated, descending order, e.g. "VISp,MOs,DG").\n'
                '5. "num_stimulus_conditions": How many unique (contrast_left, contrast_right) '
                'combinations exist?\n'
                '6. "mean_firing_rate": Mean firing rate across ALL neurons in ALL sessions, '
                'in Hz. Data is binned at 10ms. Round to 1 decimal place.\n'
                '7. "pct_low_firing": Percentage of neurons with mean firing rate < 0.5 Hz. '
                'Round to 1 decimal place.\n'
                '8. "num_correct_trials_session0": Number of correct trials in session 0 '
                '(feedback_type == 1).\n'
            ),
            "judge": {
                "type": "ExactMatchJudge",
                "answers_file": "scoring_data/step1_answers.json",
            },
        },
        # ====================================================
        # STEP 2: Neural Data Quality Control
        # ====================================================
        {
            "step_id": 2,
            "name": "Neural Data Quality Control",
            "instructions": (
                "Many neurons in the dataset are low-quality: they fire too rarely, "
                "have unstable activity across the session, or show signs of multi-unit "
                "contamination. Your task is to curate the neural data by assigning "
                "quality labels to each neuron.\n\n"
                "Save to /workdir/unit_quality.npz:\n"
                '- "quality_labels": dict mapping session_id (int) → numpy array '
                "[n_neurons] with values: 0=bad, 1=acceptable, 2=good\n"
                '- "quality_criteria": string describing your criteria\n\n'
                "Criteria to consider:\n"
                "- Firing rate: neurons with < 0.5 Hz mean may be noise\n"
                "- Stability: firing rate should not change dramatically between "
                "first and second half of trials\n"
                "- Trial presence: neuron should fire on a reasonable fraction of trials\n"
                "- Waveform width: use trough_to_peak to distinguish cell types\n"
                "- Task modulation: neurons showing stimulus/choice selectivity "
                "are more informative\n\n"
                "Your quality labels will filter neurons for downstream decoding. "
                "Better filtering → better decoding → higher score."
            ),
            "judge": {
                "type": "ExecutableJudge",
                "score_type": "continuous",
                "script": "scoring_data/scoring_step2.py",
                "command": (
                    "python scoring_data/scoring_step2.py "
                    "/workdir/unit_quality.npz "
                    "env_data/sessions "
                    "/workdir/step2_score.json"
                ),
                "output_file": "/workdir/step2_score.json",
            },
        },
        # ====================================================
        # STEP 3: Feature Engineering for Cross-Area Decoding
        # ====================================================
        {
            "step_id": 3,
            "name": "Feature Engineering for Cross-Area Decoding",
            "instructions": (
                "Build features for decoding the mouse's decision from neural activity "
                "across brain regions.\n\n"
                "The key challenge: different brain regions encode decisions at different "
                "times and with different dynamics. Visual cortex responds early. "
                "Frontal motor areas encode the decision later.\n\n"
                "Save to /workdir/neural_features.npz:\n"
                '- "X_train": [n_trials, n_features] — features from training sessions\n'
                '- "y_train": [n_trials] — decision labels (0=right, 1=left)\n'
                '- "X_val": [n_trials_val, n_features] — features from validation sessions\n'
                '- "y_val": [n_trials_val] — validation labels\n'
                '- "feature_names": array of strings\n'
                '- "session_ids_train": session ID per training trial\n'
                '- "session_ids_val": session ID per validation trial\n\n'
                "Exclude no-go trials (response==0). Labels: 0=right, 1=left.\n"
                "Features must have the SAME dimensionality for all trials.\n\n"
                "Feature ideas: per-region mean firing rates in time windows, "
                "contrast features, PCA on population activity, selectivity indices, "
                "cross-region correlations, behavioral signals.\n\n"
                "HINT: contrast difference is the strongest single predictor. "
                "Neural features should add information above and beyond contrast."
            ),
            "judge": {
                "type": "ExecutableJudge",
                "score_type": "continuous",
                "script": "scoring_data/scoring_step3.py",
                "command": (
                    "python scoring_data/scoring_step3.py "
                    "/workdir/neural_features.npz "
                    "env_data/test_sessions_unlabeled "
                    "scoring_data/test_labels "
                    "/workdir/step3_score.json"
                ),
                "output_file": "/workdir/step3_score.json",
            },
        },
        # ====================================================
        # STEP 4: Cross-Session Generalization (FINAL)
        # ====================================================
        {
            "step_id": 4,
            "name": "Cross-Session Generalization",
            "instructions": (
                "Final challenge: produce predictions for hidden test sessions "
                "from DIFFERENT MICE.\n\n"
                "Test session data is in {env_data_dir}/test_sessions_unlabeled/. "
                "These have neural data but NO behavioral labels.\n\n"
                "Save to /workdir/cross_session_predictions.npz:\n"
                '- "predictions": dict mapping session_id (int) → numpy array '
                "[n_trials] with values 0=right, 1=left (for ALL trials)\n"
                '- "prediction_method": string describing your approach\n\n'
                "Strategies:\n"
                "- Region-level features (aggregate within brain area)\n"
                "- Cross-session normalization (z-score per session)\n"
                "- Contrast features (available in test data)\n"
                "- Multi-region temporal features\n\n"
                "HINT: Contrast-only achieves ~65-70%. Neural features should "
                "push above 75%. Best solutions combine contrast with "
                "region-level neural features."
            ),
            "judge": {
                "type": "ExecutableJudge",
                "score_type": "continuous",
                "script": "scoring_data/scoring_step4.py",
                "command": (
                    "python scoring_data/scoring_step4.py "
                    "/workdir/cross_session_predictions.npz "
                    "scoring_data/test_labels "
                    "/workdir/step4_score.json"
                ),
                "output_file": "/workdir/step4_score.json",
            },
        },
    ],
}
```

---

## 6. env_requirements.txt

```
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
```

---

## 7. anti-gaming analysis

| step | attack vector | defense |
|------|--------------|---------|
| 1 | guessing answers | data-dependent values (neuron counts, firing rates) require actual computation |
| 1 | hardcoding from memory | 35 sessions × many fields = too many values to memorize |
| 2 | include all neurons (no filtering) | score = IMPROVEMENT over all-neurons baseline; doing nothing = score ~0.3 |
| 2 | exclude all neurons | minimum neuron check (≥3 per session) |
| 2 | random labels | random filtering is unlikely to beat systematic quality control |
| 3 | use only contrast (no neural features) | contrast-only baseline achieves ~0.65-0.70, which maps to score ~0.3-0.4; need neural features for higher scores |
| 3 | overfit to training data | overfitting penalty: score reduced if train-val gap > 15% |
| 3 | constant features | variance check on features |
| 4 | constant predictions | explicit check: constant predictions = score 0 |
| 4 | extreme bias (>90% one class) | bias penalty reduces score by 0.3 |
| 4 | memorize train distribution | test sessions are from DIFFERENT MICE, so memorizing train statistics doesn't help |
| 4 | only use contrast | contrast-only gets ~0.65-0.70 on test mice; need neural features for higher |

### why cross-mouse generalization is hard

the fundamental challenge: each mouse has different neurons recorded from different exact locations. session 35 might have 400 neurons in VISp, MOs, and CA1. session 36 might have 600 neurons in VISl, ACA, and SCm. the student's pipeline must:

1. aggregate at the brain-region level (not individual neuron level)
2. handle missing regions (not all sessions have all areas)
3. normalize across sessions (different mice have different baseline rates)
4. combine neural information with contrast to beat contrast-only baseline

---

## 8. baseline performance estimates

### 8.1 step 2 baselines

| approach | improvement over all-neurons | expected score |
|----------|---------------------------|----------------|
| no filtering (submit all ones) | 0.0 | ~0.3 |
| firing rate threshold only (>0.5 Hz) | +0.01 to +0.03 | ~0.35-0.45 |
| FR + stability + trial presence | +0.03 to +0.06 | ~0.45-0.65 |
| FR + stability + presence + selectivity | +0.05 to +0.10 | ~0.60-0.85 |
| optimal (multi-criteria with tuning) | +0.08 to +0.12 | ~0.75-1.0 |

### 8.2 step 3 baselines

| feature approach | val balanced accuracy | expected score |
|-----------------|----------------------|----------------|
| raw flattened spikes | ~0.55-0.60 | ~0.1-0.2 |
| contrast only | ~0.65-0.70 | ~0.3-0.4 |
| per-region mean rates | ~0.70-0.75 | ~0.4-0.6 |
| multi-window per-region + contrast | ~0.75-0.80 | ~0.6-0.8 |
| above + PCA + behavioral signals | ~0.80-0.85 | ~0.8-1.0 |

### 8.3 step 4 baselines

| approach | test balanced accuracy | expected score |
|----------|----------------------|----------------|
| random | ~0.50 | 0.0 |
| majority class | ~0.50 | 0.0 |
| contrast only | ~0.65-0.70 | ~0.3-0.4 |
| region-level features | ~0.70-0.75 | ~0.4-0.6 |
| region-level + contrast + normalization | ~0.75-0.80 | ~0.6-0.8 |
| optimal pipeline | ~0.80-0.85 | ~0.8-1.0 |

---

## 9. score mapping summary

| step | metric | 0.0 | 0.2 | 0.5 | 0.8 | 1.0 |
|------|--------|-----|-----|-----|-----|-----|
| 1 | exact match | 0 correct | - | - | - | all 8 correct |
| 2 | decoding improvement | <-0.05 | ~0.0 | ~+0.03 | ~+0.08 | ≥+0.10 |
| 3 | val balanced accuracy | 0.50 | 0.60 | 0.70 | 0.80 | ≥0.85 |
| 4 | test balanced accuracy | 0.50 | 0.60 | 0.70 | 0.80 | ≥0.85 |

**continuous reward interpretation**:
- 0.0 = total failure (worse than chance or broken submission)
- 0.2 = below median (basic attempt that partially works)
- 0.5 = median (reasonable approach, standard features)
- 0.8 = human-level (good neuroscience reasoning + feature engineering)
- 1.0 = excellent (near-optimal pipeline combining QC, features, and generalization)

---

## 10. references

### primary dataset

1. Steinmetz, N. A., Zatka-Haas, P., Carandini, M., & Harris, K. D. (2019). Distributed coding of choice, action and engagement across the mouse brain. *Nature*, 576, 266-273. https://doi.org/10.1038/s41586-019-1787-x

2. Steinmetz, N. A., et al. (2019). Dataset from Steinmetz et al. 2019. *figshare*. https://figshare.com/articles/dataset/steinmetz/9598406

### NMA materials

3. Neuromatch Academy. load_steinmetz_decisions.ipynb. https://github.com/NeuromatchAcademy/course-content/blob/main/projects/neurons/load_steinmetz_decisions.ipynb

4. Neuromatch Academy. load_steinmetz_extra.ipynb. https://github.com/NeuromatchAcademy/course-content/blob/main/projects/neurons/load_steinmetz_extra.ipynb

### related analysis repos

5. shir-openu/steinmetz-neural-decision-signals. https://github.com/shir-openu/steinmetz-neural-decision-signals

6. NMA2022Chaosfan/HMM_SteinmetzDataset. https://github.com/NMA2022Chaosfan/HMM_SteinmetzDataset

### neural decoding methods

7. Glaser, J. I., et al. (2020). Machine learning for neural decoding. *eNeuro*, 7(4). https://doi.org/10.1523/ENEURO.0506-19.2020

8. Stringer, C., et al. (2019). Spontaneous behaviors drive multidimensional, brainwide activity. *Science*, 364(6437). https://doi.org/10.1126/science.aav7893

### spike sorting quality

9. Hill, D. N., Mehta, S. B., & Kleinfeld, D. (2011). Quality metrics to accompany spike sorting of extracellular signals. *Journal of Neuroscience*, 31(24), 8699-8705. https://doi.org/10.1523/JNEUROSCI.0971-11.2011

---

*end of specification*
