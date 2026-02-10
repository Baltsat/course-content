# pm_env — 3 neuroscience data curation environments

> research-backed specifications, feb 2026

---

## executive summary

three candidate environments for data curation tasks on neuroscience datasets.
each follows the pattern: **student curates data/features → system trains fixed LogReg → score = downstream performance**.

honest quality ratings:

- **CalMS21** — A+. strongest curation story, largest impact, spec already 80% done in NMA repo
- **Steinmetz** — A. genuine cross-mouse challenge, multiple curation axes, well-documented
- **Stringer orientations** — B. science shows curation barely helps linear decoders on this data. works only with artificial constraint (neuron budget). continuous metric is unique

if picking ONE environment: CalMS21.
if picking TWO: CalMS21 + Steinmetz.
if picking THREE: all three work, but Stringer is the weakest link.

---

## comparison

### what does the student actually DO?

**CalMS21 — curate LABELS + FEATURES for behavior classification**

the student gets video-derived keypoints (28 values/frame) of two mice interacting.
annotations say which behavior is happening: attack (2.8%), investigation (29%), mount (5.6%), other (63%).
the student must engineer features (distances, velocities, angles between mice),
deal with extreme class imbalance (predicting "other" always = 0 F1 on eval metric),
and handle annotation noise at behavior boundaries.

why curation matters: raw keypoints + LogReg = 0.45 F1. expert-curated features + balanced weights = 0.65-0.70 F1.
that's a 0.20 F1 gap purely from data curation.

**Steinmetz — curate NEURONS + FEATURES for decision decoding across mice**

the student gets Neuropixels recordings from 42 brain areas in mice doing a visual task (left/right choice).
~30K neurons total, 39 sessions, 10 mice.
the student must filter bad neurons (25% are noise), select informative brain regions
(choice lives in MOs/frontal, NOT in VISp), engineer time-window features,
and normalize across sessions so the decoder works on NEW MICE it's never seen.

why curation matters: contrast-only baseline = 65-70% balanced acc. with neural features + proper
region selection + cross-mouse normalization = 80-85%. that's +15-20% from curation.

**Stringer — curate NEURONS from massive population for orientation decoding**

the student gets ~23,600 neurons from mouse visual cortex, each with responses to ~4,600 oriented stimuli.
the student must select which neurons to use for decoding the stimulus orientation (continuous, 0-360 degrees).
the trap: using ALL neurons with LogReg(C=1.0) → massive overfitting (23K features, 4.6K samples).
student must reduce dimensionality via neuron selection and/or PCA.

why curation matters (CAVEAT): the original paper (Stringer 2021 Cell) shows that with ridge regression,
curation doesn't help — the decoder handles everything. the curation challenge ONLY exists because
we force a fixed LogReg as downstream model. this makes the curation feel somewhat artificial —
the student is solving "how to make LogReg work with p >> n" rather than "how to curate data."

### numbers side-by-side

                            CalMS21                 Steinmetz               Stringer orientations
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    paper                   Sun et al 2021          Steinmetz et al 2019    Stringer et al 2021
                            NeurIPS                 Nature                  Cell
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    modality                pose keypoints          Neuropixels spikes      2-photon calcium
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    data size               ~500 MB                 ~2 GB                   ~400 MB
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    features per sample     28 raw → 50-100         ~30K neurons            ~23K neurons
                            engineered              → region-level          → subset
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    downstream task         3-class behavior        binary choice           orientation regression
                            (excl "other")          (left vs right)         (continuous 0-360 deg)
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    metric                  macro F1                balanced accuracy       R2 / angular error
                            (discrete)              (discrete)              (continuous)
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    naive LogReg            0.45-0.50 F1            65-70% bal acc          ~50% (overfit)
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    curated LogReg          0.65-0.70 F1            80-85% bal acc          TBD (~70-80%?)
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    SOTA (any model)        0.864 F1 (MABe top-1)   ~85-90% (foundation)   0.34 deg (ridge, 20K)
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    curation gap            +0.20 F1 (HUGE)         +15-20% (SIGNIFICANT)  unclear (WEAK?)
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    unique trap             "other" = 0 F1          contrast-only = 70%    spontaneous subtraction
                                                    (need neural to beat)  doesn't help linear
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────
    NMA repo support        loading notebook +      loading notebook +     loading notebook +
                            pm_env spec (645 lines) tutorials (W1D3)       tutorials (W1D5)
    ─────────────────────── ─────────────────────── ─────────────────────── ───────────────────────

---
---

## ENV 1: CalMS21 — mouse behavior classification

### the idea in one paragraph

student receives 28-dimensional keypoint trajectories of two interacting mice with noisy behavior labels.
63% of frames are "other" (no behavior). the evaluation metric (macro F1) IGNORES "other" — so a model
that always predicts "other" scores exactly 0.0. to score well, the student must: (1) engineer features
that distinguish attack/mount/investigation, (2) handle extreme class imbalance via resampling or class weights,
(3) deal with annotation noise at behavior boundaries. system trains LogReg on curated features,
evaluates macro F1 on hidden test sequences from DIFFERENT mice.

### references

- Sun et al. (2021) "The Multi-Agent Behavior Dataset" NeurIPS. arxiv:2104.02710
- Segalin et al. (2021) "MARS pipeline" eLife 10:e63720
- MABe 2021 challenge: 402 participants, top-1 = 0.864 F1. https://sites.google.com/view/mabe21/challenge
- FERAL (2025): 94.5 mAP using V-JEPA2 video foundation model. bioRxiv 2025.11.16.688666

### data

    location:       env_data/train/ (70 sequences), env_data/val/ (split from train)
    format:         .npy dict per sequence
    per frame:      keypoints [2 mice, 2 coords, 7 keypoints] = 28 values
                    + confidence scores [2 mice, 7 keypoints]
                    + annotation (0=attack, 1=investigation, 2=mount, 3=other)
    total frames:   ~508K train, ~262K test
    frame rate:     30 Hz
    class balance:  attack 2.8% | investigation 29% | mount 5.6% | other 63%

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

**F1-to-score mapping** (calibrated from MABe challenge results):

    F1 <= 0.33  →  0.0   (random guessing)
    F1  = 0.50  →  0.3   (basic features, no imbalance handling)
    F1  = 0.60  →  0.5   (decent features + class_weight="balanced")
    F1  = 0.70  →  0.7   (temporal features + SMOTE + boundary removal)
    F1  = 0.79  →  0.9   (1D CNN baseline level — near ceiling for LogReg)
    F1 >= 0.86  →  1.0   (MABe challenge top-1 — requires model change, not just curation)

### why it WILL work

1. **curation gap is proven**: raw keypoints → 0.45 F1. engineered features + balanced → 0.65-0.70 F1.
   this 0.20 gap is ENTIRELY from data curation decisions, not model architecture.

2. **anti-gaming is natural**: predicting "other" scores 0. predicting random scores 0.33.
   you CANNOT game this metric without genuinely understanding the data.

3. **multiple failure points**: wrong features (just coordinates), no imbalance handling,
   include boundary frames, ignore tracking failures when mice overlap.
   each one is independently discoverable and fixable.

4. **continuous improvement**: class_weight alone → +0.10. add distances → +0.05.
   add velocities → +0.05. add temporal → +0.05. remove boundaries → +0.02.
   clear hill-climbing path.

5. **real data, real challenge**: exact same dataset and metric as MABe 2021 challenge
   (402 participants, 28 teams). well-understood difficulty.

### progressive difficulty

    v1:  full annotations, 70 train sequences                    expected median: 0.35
    v2:  50% annotations removed (semi-supervised propagation)    expected median: 0.25
    v3:  add annotation noise (5% random flips)                   expected median: 0.20
    v4:  fine-grained subtypes (sniff-face vs sniff-body)         expected median: 0.10

---
---

## ENV 2: Steinmetz — cross-mouse neural decision decoding

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
- NMA preprocessed data: https://osf.io/agvxh/
- NMA tutorial W1D3 Tutorial 2: LogReg decoding on Steinmetz data
- NEDS (2025): transformer foundation model for neural decoding. arXiv:2504.08201
- Multi-task Masking (2024): "universal translator" for neural data. NeurIPS 2024. arXiv:2407.14668

### data

    location:       env_data/sessions/ (35 visible) + env_data/test_unlabeled/ (4 hidden)
    format:         .npz per session
    per session:    spks [n_neurons, n_trials, 250] — spike counts, 10ms bins, 2.5s/trial
                    brain_area [n_neurons] — "VISp", "MOs", "ACA", "SC", ...
                    contrast_left, contrast_right [n_trials] — 0, 0.25, 0.5, 1.0
                    response [n_trials] — -1=right, 0=nogo, 1=left
                    feedback_type [n_trials] — +1=correct, -1=incorrect
                    trough_to_peak [n_neurons] — waveform width for unit QC
    total:          39 sessions, 10 mice, ~30K neurons, 42 brain areas

    time structure:
        bins 0-50:    pre-stimulus (-0.5 to 0s)
        bins 50-100:  stimulus + early decision (0 to 0.5s)
        bins 100-150: decision + response (0.5 to 1.0s)
        bins 150-250: post-response (1.0 to 2.0s)

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

**accuracy-to-score mapping** (calibrated from paper + NMA tutorials):

    bal_acc <= 0.50  →  0.0   (chance)
    bal_acc  = 0.60  →  0.2   (using contrast only — no neural data)
    bal_acc  = 0.70  →  0.4   (basic neural features, one region)
    bal_acc  = 0.75  →  0.55  (multi-region + proper time windows)
    bal_acc  = 0.80  →  0.7   (region-level aggregation + cross-session normalization)
    bal_acc  = 0.85  →  0.85  (near-optimal for LogReg)
    bal_acc >= 0.90  →  1.0   (requires model change or foundation model)

### why it WILL work

1. **cross-mouse is genuinely hard**: different mice have completely different neurons.
   the student MUST invent features that are mouse-independent (region-level aggregation).

2. **contrast-only floor**: using only contrast_left - contrast_right gives ~65-70%.
   scoring starts above this — neural features must ADD information.

3. **neuroscience knowledge required**: choice signals live in MOs/frontal/midbrain (Steinmetz 2019).
   including ALL neurons (including VISp which encodes stimulus, not choice) hurts cross-mouse decoding.
   this is a genuine data curation insight, not just ML hygiene.

4. **multiple independent curation axes**: unit QC (firing rate, waveform), region selection,
   time window selection, normalization strategy. each improves score independently.

5. **well-calibrated difficulty**: 10 mice with varying brain area coverage.
   cross-mouse generalization has been extensively studied (IBL consortium, NEDS, MtM).

### progressive difficulty

    v1:  35 train, 4 hidden test sessions                        expected median: 0.30
    v2:  restrict to 10 best sessions (student selects)           expected median: 0.25
    v3:  remove brain_area labels — discover from activity        expected median: 0.15
    v4:  add decoy neurons (20% random Poisson spikes)            expected median: 0.10

---
---

## ENV 3: Stringer orientations — large-scale neuron selection

### the idea in one paragraph

student receives ~23,600 neurons from mouse visual cortex, each with responses to ~4,600 oriented stimuli.
goal: decode the stimulus orientation (continuous 0-360 degrees) from a curated subset of neurons.
the trap: LogReg(C=1.0) with all 23K neurons → massive overfitting (p >> n). student must select
a subset or reduce dimensionality. suite2p quality metrics (compact, skew) and spontaneous activity
components are provided for curation decisions. the unique value: continuous metric (R2/angular error)
and signal-processing-as-curation (spontaneous subtraction, neuropil quality, z-scoring strategy).

### references

- Stringer et al. (2019) "Spontaneous behaviors drive multidimensional, brainwide activity"
  Science 364:eaav7893. doi:10.1126/science.aav7893
- Stringer et al. (2019) "High-dimensional geometry of population responses"
  Nature 571:361-365. doi:10.1038/s41586-019-1346-5
- Stringer et al. (2021) "High-precision coding in visual cortex"
  Cell 184:2767-2778. doi:10.1016/j.cell.2021.03.042
- Code: https://github.com/MouseLand/stringer-et-al-2019
- NMA data: https://osf.io/ny4ut/download (stringer_orientations.npy)

### CRITICAL CAVEAT

the Stringer 2021 Cell paper demonstrates that:
1. spontaneous activity subtraction has NO effect on linear decoding
   (stimulus and spontaneous signals occupy orthogonal subspaces)
2. removing orientation-untuned neurons HURTS decoding
   (untuned neurons carry population-level information)
3. with ridge regression + all neurons, curation doesn't improve performance

the curation challenge exists ONLY because we use fixed LogReg(C=1.0) as downstream,
which overfits with 23K features. this makes the task more about "ML dimensionality
reduction" than about "understanding the data." compare to CalMS21 where curation requires
understanding mouse behavior, or Steinmetz where it requires understanding brain regions.

### data

    location:       env_data/stringer_orientations.npy
    format:         numpy dict (allow_pickle=True)
    fields:
        sresp           [23589, 4598]   — deconvolved responses (neurons x trials)
        istim           [4598]          — stimulus orientation per trial (0 to 2*pi)
        stat            list of 23589   — suite2p per-neuron quality metrics
        u_spont         [23589, 128]    — spontaneous activity PC weights
        v_spont         [128, 910]      — spontaneous activity PC timecourses
        mean_spont      [23589]         — per-neuron mean spontaneous rate
        std_spont       [23589]         — per-neuron std spontaneous rate
        run             [4598]          — running speed per trial
        xyz             [3, 23589]      — 3D neuron positions (um)

    preprocessing from paper (utils.py compile_resp):
        1. normalize: (sresp - mean_spont) / std_spont
        2. subtract spontaneous PCs: sresp -= u_spont[:,:npc] @ (u_spont[:,:npc].T @ sresp)
        3. z-score across stimuli: (sresp - mean) / std

### tools

    bash            600s/call, 14400s cumulative
    submit_answers  key-value dict (step 1 exploration)

    python pkgs:    numpy, pandas, scipy, scikit-learn, matplotlib

### what student submits

    step 1:  submit_answers — 8 exploration answers (neuron count, stimulus range, etc.)
    step 2:  /workdir/neuron_selection.npz — quality/selectivity labels + selection criteria
    step 3:  /workdir/curated_features.npz — trial-level features from selected neurons
    step 4:  /workdir/predictions.npz — predicted orientation for held-out trials

### evaluation

the orientation decoding task is discretized for LogReg:
bin orientations into 12 bins (30-degree each) → 12-class classification.
LogReg(C=1.0, multinomial) on student's features. evaluate on hidden 20% held-out trials.

**step 1 — exploration:** ExactMatchJudge, fraction correct.

**step 2 — neuron selection quality:**
compare student's selection to reference (high-selectivity neurons from paper's d-prime analysis).
selection F1 + SVD rank bonus (40-100 PCs is optimal range).

**step 3 — downstream decoding:**
system trains LogReg on student's curated features.
evaluates balanced accuracy on hidden held-out trials (12-class).
efficiency bonus: fewer neurons for same accuracy = better curation.
score = 0.7 * acc_mapped(bal_acc_12class) + 0.3 * efficiency

**step 4 — student predictions:**
evaluate student's predicted orientation bins against truth.
score = acc_mapped(bal_acc)

**accuracy-to-score mapping** (12-class balanced accuracy):

    bal_acc <= 0.083  →  0.0   (chance = 1/12)
    bal_acc  = 0.30   →  0.3   (basic PCA + LogReg)
    bal_acc  = 0.50   →  0.5   (good neuron selection + preprocessing)
    bal_acc  = 0.70   →  0.7   (optimal subset + proper z-scoring)
    bal_acc  = 0.85   →  0.9   (near-optimal)
    bal_acc >= 0.90   →  1.0   (approaching ridge regression ceiling)

### why it WILL work (with caveats)

1. **p >> n forces curation**: 23K neurons, 4.6K trials. LogReg WILL overfit.
   student MUST reduce dimensionality somehow.

2. **suite2p stat fields provide curation axes**: compact, skew, npix_norm give
   genuine cell quality metrics. this IS neuroscience-relevant filtering.

3. **continuous metric is unique**: R2 / angular error differentiates from other two envs.

4. **spontaneous subtraction trap is educational**: student may think it helps (it doesn't
   for linear decoders). discovering this is a genuine insight about neural coding.

BUT:
5. **curation surface is shallow**: the main challenge is "don't use all neurons with LogReg."
   once the student does PCA or selects ~1000 neurons, there's less room for improvement.
   compare to CalMS21 where EVERY curation decision (features, balance, boundaries) independently
   improves the score.

### progressive difficulty

    v1:  single session, 20% held-out trials, 12 orientation bins   expected median: 0.25
    v2:  finer bins (36 classes, 10 degrees each)                    expected median: 0.15
    v3:  add running-state modulation (must account for it)          expected median: 0.10
    v4:  cross-session (train on one mouse, test on another)         expected median: 0.05

---
---

## recommendation

### if you want ONE environment

**CalMS21.** unambiguously the strongest:
- largest curation impact (+0.20 F1)
- most diverse curation decisions (features, labels, quality, temporal)
- proven difficulty calibration (MABe challenge, 402 participants)
- pm_env spec already 80% written in NMA repo
- smallest data (500 MB) → fastest iteration
- clear anti-gaming (predict "other" = 0)

### if you want TWO environments

**CalMS21 + Steinmetz.** complementary:
- CalMS21 = feature/label curation on behavioral data
- Steinmetz = sample/quality curation on neural data + cross-mouse generalization
- different modalities (pose keypoints vs electrophysiology)
- different metrics (macro F1 vs balanced accuracy)

### if you want THREE environments

add Stringer, but acknowledge it's the weakest for curation.
its unique value: continuous metric + signal processing angle.
but the curation depth is shallow compared to the other two.

alternative for the 3rd slot:
- **Allen 2P Visual Behavior** — cell selection from calcium imaging, 8-class decoding.
  avoids the Stringer problem because cell counts are small (8-100 per experiment),
  making selection actually critical. but overlaps with Stringer on modality.

---

*sources: all citation details in research reports from agents (CalMS21, Steinmetz, Stringer).*
*existing NMA pm_env spec: projects/behavior_and_theory/pm_env_calms21_spec.md*
