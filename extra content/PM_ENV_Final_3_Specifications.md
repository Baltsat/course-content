# pm_env — neuroscience data curation environments (final)

> 2 confirmed + 1 optional. updated: 2026-02-09
> all environments: **student curates data** → **system trains fixed LogReg** → **score = downstream performance**

---

## executive summary

**confirmed:**
- **ENV 1: CalMS21** — mouse behavior classification from pose keypoints. feature/label curation.
- **ENV 2: Steinmetz** — cross-mouse neural decision decoding from Neuropixels. sample/quality curation.

**optional 3rd (maximum diversification):**
- **ENV 3: HCP Task fMRI** — cognitive task decoding from parcellated fMRI. spatial/region curation.

three modalities (pose keypoints, electrophysiology, fMRI). three curation types (feature engineering, quality filtering, region selection). three metrics (macro F1, balanced accuracy, balanced accuracy). three generalization challenges (cross-animal, cross-mouse, cross-subject).

---
---

# ENV 1: CalMS21 — mouse behavior classification

### the idea in one paragraph

student receives 28-dimensional keypoint trajectories of two interacting mice with noisy behavior labels.
63% of frames are "other" (no behavior). the evaluation metric (macro F1) IGNORES "other" — so a model
that always predicts "other" scores exactly 0.0. to score well, the student must: (1) engineer features
that distinguish attack/mount/investigation, (2) handle extreme class imbalance via resampling or class weights,
(3) deal with annotation noise at behavior boundaries. system trains LogReg on curated features,
evaluates macro F1 on hidden test sequences from DIFFERENT mice.

### references

- Sun et al. (2021) "The Multi-Agent Behavior Dataset" NeurIPS. arxiv:2104.02710
- Segalin et al. (2021) "MARS pipeline" eLife 10:e63720. doi:10.7554/eLife.63720
- MABe 2021 challenge: 402 participants, top-1 = 0.864 F1. https://sites.google.com/view/mabe21/challenge
- FERAL (2025): 94.5 mAP using V-JEPA2 video foundation model. bioRxiv 2025.11.16.688666
- MARS feature extractor: 270 hand-crafted spatiotemporal features. https://github.com/neuroethology/MARS

### data

    source:         CalMS21 Task 1 — https://data.caltech.edu/records/s0vdx-0k302
    location:       env_data/train/ (70 sequences), env_data/val/ (split from train)
    format:         .npy dict per sequence (allow_pickle=True)
    per frame:      keypoints [n_frames, 2, 2, 7] = [frames, mice, xy, keypoints]
                    + confidence scores [n_frames, 2, 7]
                    + annotation: 0=attack, 1=investigation, 2=mount, 3=other
    total frames:   ~508K train, ~262K test
    frame rate:     30 Hz
    class balance:  attack 2.76% | investigation 28.88% | mount 5.64% | other 62.72%

    7 keypoints per mouse: nose, left_ear, right_ear, neck, left_hip, right_hip, tail_base
    mouse 0 = resident (black), mouse 1 = intruder (white)
    image: 1024 x 570 pixels

**data messiness:**
- extreme class imbalance (attack = 2.8% vs other = 62.6%)
- keypoint tracking failures when mice overlap (~3% of frames)
- annotation noise at behavior boundaries (+-5 frames inter-annotator disagreement)
- sequences vary in length (900-3000 frames at 30 fps)
- mouse identity (resident vs intruder) matters for interpretation
- train/test split by animal identity (different mice)

**MARS feature catalog** (from neuroethology/MARS GitHub):
270 hand-crafted features organized as:
- inter-mouse distances (nose-nose, nose-tail, centroid, all 91 pairwise keypoint distances)
- per-mouse velocities (centroid speed, angular velocity)
- body orientation angles (heading, relative bearing, facing angle)
- posture features (body length, body width, ear-to-ear distance)
- temporal features (rolling mean/std at multiple window sizes)

### tools

    bash            600s/call, 14400s cumulative
    submit_answers  key-value dict (step 1 exploration)

    python pkgs:    numpy, pandas, scipy, scikit-learn, matplotlib

### what student submits

    step 1:  submit_answers — 8 exploration answers (class counts, keypoint names, etc.)
    step 2:  /workdir/features.npz — engineered feature matrix X_train, y_train, feature_names
    step 3:  /workdir/curated_train.npz — balanced/curated training set + sample_weights
    step 4:  /workdir/predictions.npz — frame-level predictions for all sequences

### evaluation

**step 1 — exploration (ExactMatchJudge):**
fraction of 8 answers correct → 0.0 to 1.0

**step 2 — feature quality:**
system trains LogReg(C=0.1, multinomial) on student's features.
evaluates macro F1 on student's val split (excluding "other").
feature diversity bonus: has_distance, has_velocity, has_orientation, has_temporal.
score = 0.6 * f1_mapped(macro_f1) + 0.4 * diversity

**step 3 — curation quality:**
system trains LogReg on student's CURATED train set.
evaluates on HIDDEN test sequences using REFERENCE features (not student's).
score = f1_mapped(macro_f1)

**step 4 — final (student's predictions directly):**
student's frame-level predictions on hidden test sequences.
score = f1_mapped(macro_f1)

**F1-to-score mapping** (calibrated from MABe 2021 challenge, 402 participants):

    F1 <= 0.33  →  0.0   (random guessing on 3-class)
    F1  = 0.45  →  0.2   (raw keypoints + LogReg, no imbalance handling)
    F1  = 0.55  →  0.4   (basic distance features, class_weight="balanced")
    F1  = 0.65  →  0.6   (MARS-style features + resampling)
    F1  = 0.72  →  0.75  (temporal features + SMOTE + boundary removal)
    F1  = 0.79  →  0.9   (MABe baseline = 0.793 ± 0.011 — near LogReg ceiling)
    F1 >= 0.86  →  1.0   (MABe top-1 = 0.864 ± 0.011 — requires model change)

**published baselines** (from Sun et al. 2021 + MABe challenge):

    raw keypoints + LogReg:          ~0.40-0.50 macro F1
    MARS 270 features + LogReg:      ~0.60-0.65 macro F1
    TREBA features + LogReg:         ~0.70 macro F1
    neural baseline (1D CNN):        0.793 ± 0.011
    task programming (TREBA):        0.829 ± 0.004
    MABe top-1 (team solution):      0.864 ± 0.011
    FERAL (V-JEPA2, 2025):           0.945 mAP (different metric but frontier)

**curation gap: +0.20-0.30 F1 purely from data curation decisions** (confirmed by MABe challenge)

### anti-hack

| threat | defense |
|--------|---------|
| predict "other" always | macro F1 excludes "other" → score = 0 |
| train on test data | test sequences in scoring_data/, inaccessible |
| memorize validation labels | hidden sequences mixed into "validation" |
| constant class predictions | require >= 3 unique classes in predictions |
| ignore class imbalance | naive on imbalanced data → F1 ~ 0.15 |
| submit MARS features directly | no network access; MARS code not in container |

### why it WILL work

1. **curation gap is proven**: raw → 0.45 F1. curated → 0.65-0.70 F1.
   this +0.20 gap is ENTIRELY from data curation, not model architecture.

2. **anti-gaming is natural**: predicting "other" = 0. random = 0.33.
   you CANNOT game this metric without genuinely understanding the data.

3. **multiple failure points**: wrong features (raw coords), no imbalance handling,
   include boundary frames, ignore tracking failures. each independently fixable.

4. **continuous improvement**: class_weight → +0.10. distances → +0.05.
   velocities → +0.05. temporal → +0.05. boundary removal → +0.02.

5. **real challenge**: same dataset/metric as MABe 2021 challenge (402 participants, 28 teams).

### progressive difficulty

    v1:  full annotations, 70 train sequences                    expected median: 0.30
    v2:  50% annotations removed (semi-supervised propagation)    expected median: 0.20
    v3:  add annotation noise (5% random flips)                   expected median: 0.15
    v4:  fine-grained subtypes (sniff-face vs sniff-body)         expected median: 0.10

---
---

# ENV 2: Steinmetz — cross-mouse neural decision decoding

### the idea in one paragraph

student receives Neuropixels recordings from 42 brain areas in mice performing a visual choice task
(turn wheel left or right based on stimulus contrast). 39 sessions from 10 mice, ~30K neurons total.
the key challenge: decode the mouse's CHOICE from neural activity, then generalize to 4 completely
new sessions from mice the student has never seen. this requires: (1) filtering bad neurons (25% are noise),
(2) discovering which brain regions encode choice (frontal/midbrain, NOT visual cortex),
(3) engineering region-level features that transfer across mice with completely different neurons,
(4) normalizing across sessions. system trains LogReg on curated features, evaluates balanced accuracy
on hidden sessions from new mice.

### references

- Steinmetz et al. (2019) "Distributed coding of choice, action and engagement across the mouse brain"
  Nature 576:266-273. doi:10.1038/s41586-019-1787-x
- NMA preprocessed data: https://osf.io/agvxh/ (part0), https://osf.io/uv3mw/ (part1), https://osf.io/ehmw2/ (part2)
- NMA tutorial W1D3 Tutorial 2: LogReg decoding on Steinmetz data
- NEDS (2025): Neural Encoding and Decoding at Scale. 91% cross-animal choice decoding. arXiv:2504.08201
- Multi-task Masking (2024): "universal translator" for neural data. NeurIPS 2024. arXiv:2407.14668

### data

    source:         Steinmetz et al. 2019 — figshare.com/articles/dataset/steinmetz/9598406
    NMA download:   osf.io/agvxh/ (part0), osf.io/uv3mw/ (part1), osf.io/ehmw2/ (part2)
    location:       env_data/sessions/ (35 visible) + env_data/test_unlabeled/ (4 hidden)
    format:         .npz per session
    per session:    spks [n_neurons, n_trials, 250] — spike counts, 10ms bins, 2.5s/trial
                    brain_area [n_neurons] — "VISp", "MOs", "ACA", "SC", ...
                    contrast_left, contrast_right [n_trials] — 0, 0.25, 0.5, 1.0
                    response [n_trials] — -1=right, 0=nogo, 1=left
                    feedback_type [n_trials] — +1=correct, -1=incorrect
                    trough_to_peak [n_neurons] — waveform width for unit QC
                    mouse_name, date_exp — session metadata
    total:          39 sessions, 10 mice, ~30K neurons, 42 brain areas
    extra:          spike times (osf.io/4bjns), waveforms (osf.io/ugm9v), LFP (osf.io/kx3v9)

    time structure (250 bins = 2.5s per trial):
        bins 0-49:    pre-stimulus (-0.5s to 0s)
        bins 50-99:   stimulus + early decision (0s to 0.5s)
        bins 100-149: decision + response (0.5s to 1.0s)
        bins 150-199: post-response + feedback (1.0s to 1.5s)
        bins 200-249: post-trial (1.5s to 2.0s)

    brain region groups (42 areas, 7 groups):
        visual cortex:   VISa, VISam, VISl, VISp, VISpm, VISrl
        thalamus:        CL, LD, LGd, LH, LP, MD, MG, PO, POL, PT, RT, SPF, TH, VAL, VPL, VPM
        hippocampal:     CA, CA1, CA3, DG, POST, SUB
        frontal motor:   ACA, MOs, MOp, PL, ILA, ORB    ← CHOICE ENCODING HERE
        midbrain:        SCm, MRN, APN, IC, SNr          ← AND HERE
        striatum:        ACB, CP, GPe, LS, OLF, SI
        other cortex:    RSP, SSp, SSs, TEa, VISC

**data messiness:**
- ~25% of neurons have mean firing rate < 0.5 Hz (likely noise or lost units)
- waveform widths < 0.2 ms suggest multi-unit contamination
- different sessions record different brain region combinations
- trial-level noise: ~15% lapse trials (random choice ignoring stimulus)
- cross-mouse: completely different neurons (no unit matching possible)
- neuron counts vary: 50-1500+ per session

**key neuroscience insight (the trap):**
choice signals are in frontal motor cortex (MOs, ACA) and midbrain (SCm, MRN),
NOT in primary visual cortex (VISp). VISp encodes stimulus contrast, not choice.
including ALL brain regions dilutes the choice signal with stimulus encoding.
the student must discover this from the data — it's the core curation insight.
(confirmed by Steinmetz 2019 Fig 4, NEDS 2025 cross-area analysis)

### tools

    bash            600s/call, 14400s cumulative
    submit_answers  key-value dict (step 1 exploration)

    python pkgs:    numpy, pandas, scipy, scikit-learn, matplotlib

### what student submits

    step 1:  submit_answers — 8 exploration answers (neuron counts, brain areas, trial types)
    step 2:  /workdir/unit_quality.npz — quality labels per neuron per session (0=bad, 1=ok, 2=good)
    step 3:  /workdir/neural_features.npz — trial-level feature matrix from curated neurons
    step 4:  /workdir/cross_session_predictions.npz — left/right predictions for 4 hidden sessions

### evaluation

**step 1 — exploration (ExactMatchJudge):**
fraction correct → 0.0 to 1.0

**step 2 — neural QC:**
for each training session: decode using ONLY student's "good" neurons vs ALL neurons (baseline).
score = mapped improvement over baseline. negative improvement → 0.0.

**step 3 — feature quality:**
system trains LogReg(C=0.1) on student's features.
evaluates balanced accuracy on student's val split.
overfitting penalty: if train_acc - val_acc > 0.15, reduce by 20%.
score = acc_mapped(bal_acc) * overfit_penalty

**step 4 — final (cross-mouse):**
compare student's predictions for sessions 36-39 to true (hidden) response labels.
per-session balanced accuracy, then mean across 4 sessions.
score = acc_mapped(mean_bal_acc)

**accuracy-to-score mapping** (calibrated from Steinmetz 2019 + NMA tutorials + NEDS):

    bal_acc <= 0.50  →  0.0   (chance)
    bal_acc  = 0.60  →  0.2   (contrast-only prediction, no neural data)
    bal_acc  = 0.70  →  0.4   (basic neural features, one region)
    bal_acc  = 0.75  →  0.55  (multi-region + proper time windows)
    bal_acc  = 0.80  →  0.7   (region-level aggregation + cross-session normalization)
    bal_acc  = 0.85  →  0.85  (near-optimal for LogReg)
    bal_acc >= 0.90  →  1.0   (requires foundation model — NEDS achieves 91%)

**published baselines:**

    contrast-only (no neural data):         ~65-70% balanced accuracy
    all neurons, mean rate + LogReg:        ~55-60% (worse than contrast — overfitting)
    VISp neurons only, LogReg:              ~60-65% (stimulus encoding, not choice)
    MOs/frontal neurons, proper windows:    ~75-80%
    multi-region + normalization + LogReg:  ~80-85%
    NEDS transformer (2025):                91% cross-animal
    MtM universal translator (2024):        ~85% cross-session

**curation gap: +15-20pp balanced accuracy from neural feature curation** (over contrast-only baseline)

### anti-hack

| threat | defense |
|--------|---------|
| use response labels from test sessions | response removed from test_unlabeled/ |
| predict using only contrast (no neural) | contrast alone = ~0.60; neural needed for higher |
| constant predictions | check std(preds) > 0, class balance not extreme |
| overfit to training sessions | overfitting penalty in step 3; cross-mouse eval in step 4 |
| bad QC makes things worse | step 2 score = relative to all-neurons baseline |
| include all brain areas | dilutes choice signal → hurts cross-mouse generalization |

### why it WILL work

1. **cross-mouse is genuinely hard**: different mice have completely different neurons.
   the student MUST invent features that are mouse-independent (region-level aggregation).

2. **contrast-only floor**: using only contrast_left - contrast_right = ~65-70%.
   neural features must ADD information above this baseline.

3. **neuroscience knowledge required**: choice → MOs/frontal/midbrain (Steinmetz 2019 Fig 4).
   VISp encodes stimulus, not choice. this is a genuine curation insight.

4. **multiple independent axes**: unit QC (firing rate, waveform), region selection,
   time window (decision period vs post-response), normalization strategy.

5. **well-calibrated difficulty**: 10 mice with varying brain area coverage.
   cross-mouse generalization extensively studied (IBL, NEDS, MtM).

### progressive difficulty

    v1:  35 train, 4 hidden test sessions                        expected median: 0.30
    v2:  restrict to 10 best sessions (student selects)           expected median: 0.25
    v3:  remove brain_area labels — discover from activity        expected median: 0.15
    v4:  add decoy neurons (20% random Poisson spikes)            expected median: 0.10

---
---

# ENV 3: HCP Task fMRI — cognitive task decoding (optional)

### the idea in one paragraph

student receives pre-parcellated fMRI data (360 Glasser atlas regions) from 80 human subjects
performing 7 cognitive tasks (motor, working memory, emotion, gambling, language, relational, social).
each task has multiple conditions with known timing (EV files). the student must: (1) extract
condition-level features from raw BOLD time series (which temporal window? mean vs contrast?),
(2) select informative brain parcels (360 is too many for LogReg — must curate to ~50-100),
(3) normalize across subjects (different BOLD baselines), (4) handle variable trial counts across tasks.
system trains LogReg on curated parcel features, evaluates balanced accuracy on 20 hidden subjects.

### references

- Glasser et al. (2016) "A multi-modal parcellation of human cerebral cortex"
  Nature 536:171-178. doi:10.1038/nature18933
- Ji et al. (2019) "Mapping the human brain's cortical-subcortical functional network organization"
  NeuroImage 185:35-57. doi:10.1016/j.neuroimage.2018.10.006 (12 network parcellation)
- HCP Reference Manual: https://www.humanconnectome.org/storage/app/media/documentation/s1200/
  HCP_S1200_Release_Reference_Manual.pdf (task descriptions pp. 45-54)
- NMA data: https://osf.io/2y3fw/download (hcp_task.tgz, pre-parcellated)
- Barch et al. (2013) "Function in the HCP" NeuroImage 80:169-189
- NMA student project E48: task state decoding ~90% with FC + RF. https://github.com/seannoah/nma2020-HCP-project
- NMA student project E52: GNN-LSTM for WM demand. https://github.com/saamehsanaaee/WMAD-Montbretia_Cabinet-ISP

### data

    source:         Human Connectome Project S1200 release
    NMA download:   osf.io/2y3fw/download (hcp_task.tgz)
    location:       env_data/hcp_task/subjects/{subjectID}/{TASK}/{RUN}/
    format:         data.npy per run (360 x n_timepoints) + EVs/ folder
    subjects:       339 total in NMA dataset (use 100-subject behavioral subset:
                    80 train, 20 hidden test)
    parcels:        360 Glasser atlas ROIs (180 per hemisphere)
    TR:             0.72 seconds
    runs:           2 per task per subject (LR, RL)
    scan duration:  ~284 timepoints per run (~204 seconds)

    7 tasks with conditions:
        MOTOR:      lf, rf, lh, rh, t, cue (6 conditions)
        WM:         0bk_body, 0bk_faces, 0bk_places, 0bk_tools,
                    2bk_body, 2bk_faces, 2bk_places, 2bk_tools (8 conditions)
        EMOTION:    fear, neut (2 conditions)
        GAMBLING:   loss, win (2 conditions)
        LANGUAGE:   math, story (2 conditions)
        RELATIONAL: match, relation (2 conditions)
        SOCIAL:     ment, rnd (2 conditions)

    per run:
        data.npy:   [360, n_timepoints] float — parcellated BOLD signal, mean-removed
        EVs/:       {condition}.txt — onset, duration, amplitude per trial
                    Stats.txt — behavioral data (RT, accuracy) averaged per run

    region info:
        regions.npy — [name, network] per parcel (360 regions, 12 networks)
        networks: somatomotor, cingulo-opercular, auditory, default mode,
                  visual1, visual2, frontoparietal, language, posterior multimodal,
                  ventral multimodal, orbito-affective, dorsal attention

**data messiness:**
- BOLD signal is slow (hemodynamic delay ~5s) — trial timing is imprecise
- different tasks have different scan durations (176-405 timepoints)
- 2 runs per task have opposite phase encoding (LR vs RL) — systematic signal differences
- cross-subject variability: same parcel, same task, different activation magnitude
- some subjects have missing runs or behavioral data
- 360 parcels is too many features for small sample sizes → overfitting risk

**key neuroscience insight (curation opportunity):**
each task activates SPECIFIC brain networks:
- MOTOR → somatomotor network (lateralized: left hand = right hemisphere)
- WM → frontoparietal + dorsal attention
- EMOTION → default mode + orbito-affective
- LANGUAGE → language network (left-lateralized temporal/frontal)
- RELATIONAL → frontoparietal
selecting task-relevant parcels DRAMATICALLY improves decoding vs using all 360.
(confirmed by NMA notebook: group-level contrasts show clear network specificity)

### tools

    bash            600s/call, 14400s cumulative
    submit_answers  key-value dict (step 1 exploration)

    python pkgs:    numpy, pandas, scipy, scikit-learn, matplotlib, nilearn

### what student submits

    step 1:  submit_answers — 8 exploration answers (subject count, parcel count, task list, etc.)
    step 2:  /workdir/parcel_selection.npz — informativeness labels per parcel + selection criteria
    step 3:  /workdir/task_features.npz — trial-level features from curated parcels
    step 4:  /workdir/task_predictions.npz — 7-class task predictions for hidden subjects

### evaluation

**step 1 — exploration (ExactMatchJudge):**
fraction correct → 0.0 to 1.0

**step 2 — parcel selection quality:**
compare student's parcel selection to reference (parcels with highest t-statistics
from group-level task contrasts, computed on all 100 subjects).
selection precision + recall. network-coverage bonus (selected from >= 5 distinct networks).
score = 0.7 * selection_f1 + 0.3 * network_coverage

**step 3 — feature quality:**
system trains LogReg(C=0.1, multinomial, 7-class) on student's task features.
evaluates balanced accuracy on student's val split.
score = acc_mapped(bal_acc)

**step 4 — final (cross-subject):**
student's 7-class task predictions for 20 hidden subjects.
score = acc_mapped(bal_acc_7class)

**accuracy-to-score mapping** (7-class balanced accuracy):

    bal_acc <= 0.14  →  0.0   (chance = 1/7)
    bal_acc  = 0.40  →  0.2   (raw mean parcels, no selection)
    bal_acc  = 0.60  →  0.4   (basic parcel features, some normalization)
    bal_acc  = 0.75  →  0.6   (network-level aggregation + z-scoring)
    bal_acc  = 0.85  →  0.8   (curated parcels + condition contrasts + proper normalization)
    bal_acc >= 0.95  →  1.0   (near-optimal — deep learning territory)

**published baselines:**

    raw parcel means + LogReg (360 features):       ~60-70% (overfitting risk)
    FC patterns + RF/SVM (E48 student project):      ~90% (complex model, not LogReg)
    network-level aggregation + LogReg:              ~80-85% (12 network features)
    curated parcels + condition contrasts + LogReg:  ~85-90%
    GNN-LSTM (E52 project, WM only):                cross-task transfer demonstrated
    deep learning approaches:                        95%+ on HCP task classification

**curation gap: +20-30pp balanced accuracy from parcel/feature curation** (LogReg: raw ~60% → curated ~85%)
note: E48 achieves ~90% with FC + RF, but with LogReg constraint, curation is critical

### anti-hack

| threat | defense |
|--------|---------|
| use all 360 parcels (no curation) | overfitting with p > n → poor cross-subject |
| use scan length to identify task | features must be trial-level (condition-averaged), not scan-level |
| peek at hidden subject data | hidden subjects not in env_data/ |
| constant predictions | require >= 5 unique classes, balanced across tasks |
| ignore cross-subject normalization | different BOLD baselines → poor generalization |

### why it WILL work

1. **parcel curation is essential**: 360 parcels but only ~50-100 are task-discriminative.
   using all → overfitting. using random subset → poor accuracy. genuine selection needed.

2. **neuroscience knowledge helps**: knowing that motor = somatomotor, language = temporal/frontal,
   WM = frontoparietal gives a massive head start. discoverable from data exploration.

3. **different from other envs**: fMRI modality, spatial curation (which brain regions),
   human subjects (vs mouse), cognitive tasks (vs behavior/decision).

4. **cross-subject generalization**: same conceptual challenge as cross-mouse in Steinmetz
   but in human fMRI domain. different normalization strategies needed.

5. **clean data, hard problem**: NMA pre-parcellated data is clean (no preprocessing needed),
   but cross-subject cognitive decoding from parcels is genuinely challenging.

### progressive difficulty

    v1:  80 train subjects, 20 hidden, 7-class task decoding       expected median: 0.25
    v2:  within-task condition decoding (e.g., 5-class MOTOR)       expected median: 0.20
    v3:  reduce to 50 train subjects (smaller sample)               expected median: 0.15
    v4:  remove network labels — discover parcel grouping            expected median: 0.10

---
---

# cross-environment comparison

### numbers side-by-side

                            CalMS21                 Steinmetz               HCP fMRI
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    paper                   Sun et al 2021          Steinmetz et al 2019    Glasser et al 2016
                            NeurIPS                 Nature                  Nature
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    species                 mouse (pairs)           mouse (single)          human
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    modality                pose keypoints          Neuropixels spikes      parcellated fMRI
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    curation type           feature engineering     quality filtering       region selection
                            + label balancing       + cross-session norm    + feature extraction
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    data size               ~500 MB                 ~2 GB                   ~1.5 GB
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    features per sample     28 raw → 50-270         ~30K neurons            360 parcels
                            engineered              → region-level          → curated subset
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    downstream task         3-class behavior        binary choice           7-class cognitive
                            (excl "other")          (left vs right)         task
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    metric                  macro F1                balanced accuracy       balanced accuracy
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    naive LogReg            0.40-0.50 F1            50-60% bal acc          ~60-70% (raw parcels)
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    curated LogReg          0.65-0.70 F1            80-85% bal acc          ~85-90%
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    SOTA (any model)        0.864 F1 (MABe top-1)  91% (NEDS)             ~90% FC+RF (E48)
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    curation gap            +0.20 F1 (HUGE)         +15-20pp (STRONG)      +20-30pp (STRONG)
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    unique trap             "other" = 0 F1          contrast-only = 70%    scan length varies
                                                    (need neural to beat)  (can't use as feature)
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    generalization          cross-animal            cross-mouse             cross-subject
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    NMA support             loading notebook +      loading notebook +      loading notebook +
                            pm_env spec (645 lines) tutorials (W1D3)       tutorials (fMRI)
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────

### portfolio rationale

- CalMS21 + Steinmetz = **confirmed pair**. complementary modalities, curation types, and challenges.
- HCP fMRI = **optional 3rd** for maximum diversification. human fMRI is a completely different domain
  from mouse electrophysiology and pose keypoints. adds spatial curation (brain region selection)
  as a new axis not covered by the other two.

### implementation priority

| priority | environment | rationale |
|----------|-------------|-----------|
| **1st** | CalMS21 | smallest data (500 MB), fastest iteration, largest curation gap, spec 80% done |
| **2nd** | Steinmetz | cross-mouse is compelling, well-documented, data in NMA repo |
| **3rd** | HCP fMRI | clean data, different domain, optional for diversification |

---

*end of specification*
