# pm_env Comprehensive Neural ML Environment Catalog

> comprehensive structured document cataloging all pm_env environments derivable from NMA resources
> created: 2026-01-30 | author: claude opus 4.5

---

## executive summary

this catalog identifies **62 unique pm_env environments** with **250+ task variants** derived from neuromatch academy resources + 35 evaluated external repos. environments are organized by customer priority alignment (★★★★★ to ★★★☆☆) and grouped by shared infrastructure to minimize duplication.

> **update 2026-01-30**: added 17 new environments (E46-E62) from systematic repo evaluation

### customer priority matrix (updated with E46-E62)

| priority | category | env count | task variants |
|----------|----------|-----------|---------------|
| ★★★★★ | messy input data science | 14 | 84 |
| ★★★★★ | implement ML papers | 10 | 58 |
| ★★★★★ | data curation | 5 | 32 |
| ★★★★☆ | hill-climbing core metrics | 12 | 68 |
| ★★★★☆ | hard scientific reasoning | 8 | 42 |
| ★★★★☆ | training/inference optimization | 6 | 24 |
| ★★★☆☆ | kernel tasks (pallas/jax) | 3 | 12 |

### hardware constraints

- **compute**: TPU v5e (2x2), CPU only, NO GPU
- **framework**: JAX/MaxText, Qwen models (0.6B-4B)
- **time**: 2-4 hour tasks, 10 min max per tool call
- **stack**: podman containers, MCP tools, uv/python

### key insight

NMA datasets are IDEAL for pm_env bc:
1. real-world messy data (multi-format, missing values, cross-session drift)
2. continuous metrics (decoding accuracy, latency, correlation)
3. multiple valid approaches (no single "correct" solution)
4. documented failure modes from student projects

---

## part 1: dataset-driven environments (tier 1 – build first)

### E01: AJILE12 multi-format ECoG wrangling

**customer alignment**: ★★★★★ messy input data science

**dataset**: AJILE12 – 55 days continuous human ECoG + pose tracking
- source: [DANDI archive](https://www.dandiarchive.org/)
- format: NWB files, parquet metadata, video annotations
- size: ~50GB per subject (12 subjects)

**infrastructure**: CPU-only, streaming data loader, ~2GB RAM footprint

**pm_env opportunity**: multi-format data wrangling is EXACTLY what customer wants
- ECoG neural data (NWB format, HDF5-based)
- pose trajectories (parquet)
- behavioral annotations (JSON/CSV)
- electrode metadata (separate file)

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E01-T1 | load + merge ECoG with pose data | % frames correctly aligned | medium |
| E01-T2 | extract high-gamma features across sessions | SNR of extracted features | medium |
| E01-T3 | handle electrode re-referencing across days | consistency score | hard |
| E01-T4 | build streaming loader for 55-day dataset | throughput (samples/sec) | hard |
| E01-T5 | detect + handle artifacts in long recordings | F1 on artifact detection | hard |
| E01-T6 | segment continuous data into movement events | IoU with ground truth | medium |

**failure modes** (from NMA student projects):
- electrode channel mismatch across sessions
- timestamp drift over multi-day recordings
- memory overflow on naive loading
- incorrect spectral feature extraction (Nyquist violations)

**cross-references**:
- NMA notebook: `projects/ECoG/exploreAJILE12.ipynb`
- website template: `website_projects/ECoG/AJILE12/`
- student project: NMA2022Chaosfan/HMM_SteinmetzDataset (similar pipeline)
- paper: [Singh et al. 2021](https://doi.org/10.1016/j.jneumeth.2021.109199)

---

### E02: IBL brainwide neuropixels cross-region analysis

**customer alignment**: ★★★★★ messy input data science + hill-climbing

**dataset**: IBL brainwide map – standardized behavioral task across 12 labs
- source: [IBL ONE API](https://one.internationalbrainlab.org/)
- format: ALF (Alyx file convention), parquet, NWB
- size: ~500GB total, streamable via ONE

**infrastructure**: CPU-only, ONE API client, ~4GB RAM for population analysis

**pm_env opportunity**: cross-lab standardization + multi-region integration
- behavioral data (trials, wheel movements, licks)
- spike-sorted units (kilosort outputs)
- probe insertion coordinates
- histology-aligned atlas regions

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E02-T1 | load sessions with units in specific region | API call efficiency | easy |
| E02-T2 | align spike times to behavioral events | spike-behavior correlation | medium |
| E02-T3 | compare stimulus encoding across brain regions | explained variance (R²) | medium |
| E02-T4 | decode choice from population activity | AUC across sessions | hard |
| E02-T5 | handle cross-lab variability in preprocessing | decoder generalization | hard |
| E02-T6 | compute response latencies across hierarchy | rank correlation with anatomy | medium |
| E02-T7 | predict block side from neural activity | temporal accuracy | hard |

**failure modes**:
- ONE API rate limiting / authentication
- inconsistent unit quality across labs
- atlas coordinate system confusion (CCF vs IBL)
- temporal alignment jitter between modalities

**cross-references**:
- NMA notebook: `projects/neurons/IBL_BWM_Neuromatch_tutorial.ipynb`
- NMA notebook: `projects/neurons/IBL_ONE_tutorial.ipynb`
- website template: `website_projects/neurons/brain-wide-map/`
- paper: [IBL et al. 2023](https://doi.org/10.1101/2023.07.04.547681)

---

### E03: algonauts visual encoding (DNN → brain)

**customer alignment**: ★★★★★ implement ML papers + hill-climbing

**dataset**: algonauts challenge – predicting fMRI from video/images
- source: [algonauts project](http://algonauts.csail.mit.edu/)
- format: fMRI beta weights, video frames, DNN features
- size: ~10GB (preprocessed)

**infrastructure**: CPU + small model inference, pre-extracted DNN features

**pm_env opportunity**: classic encoding model challenge
- map DNN layer activations → voxel responses
- compare architectures (ResNet, CLIP, video models)
- optimize encoding hyperparameters

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E03-T1 | extract features from pretrained DNN | feature quality (variance) | easy |
| E03-T2 | fit ridge regression encoding model | R² on held-out data | medium |
| E03-T3 | compare layers for ROI prediction | layer-wise R² ranking | medium |
| E03-T4 | temporal integration for video stimuli | temporal prediction accuracy | hard |
| E03-T5 | noise ceiling estimation | ceiling-corrected R² | hard |
| E03-T6 | multi-ROI joint encoding | total explained variance | hard |

**failure modes**:
- overfitting on small fMRI datasets
- incorrect hemodynamic delay modeling
- feature dimensionality explosion
- train/test leakage on repeated stimuli

**cross-references**:
- NMA notebook: `projects/fMRI/load_algonauts_videos.ipynb`
- NMA notebook: `projects/fMRI/load_algonauts2021.ipynb`
- awesome list: CYHSM/awesome-neuro-ai-papers

---

### E04: data curation pipeline (dolma-style)

**customer alignment**: ★★★★★ data curation

**dataset**: CommonCrawl/Dolma subsets for neural text
- source: user-provided corpus or simulated
- format: JSONL, parquet
- size: configurable (100MB–10GB)

**infrastructure**: CPU-only, streaming, Qwen-0.6B for quality scoring

**pm_env opportunity**: replicate data curation workflows
- quality filtering (perplexity, toxicity, repetition)
- deduplication (MinHash, exact)
- domain classification
- data mixing strategies

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E04-T1 | implement perplexity-based filtering | downstream loss improvement | medium |
| E04-T2 | deduplicate corpus efficiently | duplicate removal rate + speed | medium |
| E04-T3 | classify domain relevance | F1 on domain labels | medium |
| E04-T4 | optimize data mixture ratios | eval loss on held-out | hard |
| E04-T5 | detect + remove low-quality patterns | manual audit agreement | hard |
| E04-T6 | curate scientific text subset | citation quality proxy | hard |
| E04-T7 | use LLM to rephrase/augment data | downstream task improvement | hard |

**failure modes**:
- over-filtering removes diverse examples
- dedup too aggressive on similar-but-valid
- domain classifier overfits to surface features
- data leakage from eval into train

**cross-references**:
- customer quote: "curate data with a script / embeddings / another model"
- paper: [Dolma](https://arxiv.org/abs/2402.00159)

---

### E05: paper implementation (recent ML papers)

**customer alignment**: ★★★★★ implement ML papers

**dataset**: arxiv papers + reference implementations
- source: recent (<2 years) papers with code
- format: PDF + code snippets
- size: N/A (task-based)

**infrastructure**: CPU/TPU for implementation, reference tests

**pm_env opportunity**: reading comprehension → working code
- understand paper methodology
- implement key algorithm
- validate against reported results

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E05-T1 | port attention mechanism to JAX | test pass + correctness | medium |
| E05-T2 | implement flash attention variant | speedup vs naive + correctness | hard |
| E05-T3 | add KV cache optimization to existing code | inference speedup | hard |
| E05-T4 | implement speculative decoding | acceptance rate + speedup | hard |
| E05-T5 | port training loop to TPU (pjit) | training throughput | hard |
| E05-T6 | implement MoE routing from paper | load balance + quality | hard |
| E05-T7 | add gradient checkpointing | memory reduction + correctness | medium |

**failure modes**:
- misunderstanding paper notation
- subtle numerical precision issues
- incorrect broadcasting/shaping
- missing edge cases from paper

**cross-references**:
- customer quote: "add the technique to an existing popular repo"
- awesome list: mazabou/awesome-neurofm

---

### E06: CalMS21 social behavior feature engineering

**customer alignment**: ★★★★★ messy input data science

**dataset**: CalMS21 – mouse social behavior with pose tracking
- source: [CalMS21 challenge](https://arxiv.org/abs/2104.02710)
- format: pose keypoints, behavior annotations
- size: ~5GB (compressed)

**infrastructure**: CPU-only, sklearn/numpy

**pm_env opportunity**: classic behavior classification challenge
- 63% of frames are "other" (class imbalance)
- features must be hand-crafted or learned
- temporal structure matters

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E06-T1 | compute velocity/acceleration features | feature variance | easy |
| E06-T2 | engineer inter-animal distance features | predictive power | medium |
| E06-T3 | handle class imbalance (attack/mount/sniff vs other) | macro F1 | hard |
| E06-T4 | temporal feature extraction (bout duration) | bout detection IoU | medium |
| E06-T5 | cluster behavioral subtypes | silhouette + interpretability | hard |
| E06-T6 | build transition matrix model | likelihood on held-out | hard |
| E06-T7 | test Markov assumption at multiple timescales | KL divergence from Markov | hard |

**failure modes**:
- ignoring class imbalance → trivial "other" classifier
- temporal leakage in train/test split
- overfitting to specific animal pairs
- features that don't generalize across sessions

**cross-references**:
- NMA notebook: `projects/behavior_and_theory/Loading_CalMS21_data.ipynb`
- website template: `website_projects/behavior_and_theory/mouse_social_behavior/`
- paper: [Segalin et al. 2020](https://doi.org/10.7554/eLife.63720)

---

## part 2: analysis pipeline environments (tier 2 – build second)

### E07: allen visual behavior 2P cell-type analysis

**customer alignment**: ★★★★☆ hill-climbing + scientific reasoning

**dataset**: allen visual behavior 2-photon calcium imaging
- source: [allen brain observatory](https://portal.brain-map.org/)
- format: NWB via allen SDK
- size: ~100GB streamable

**infrastructure**: CPU + allen SDK, ~8GB RAM for session

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E07-T1 | load session and extract ΔF/F traces | trace quality (SNR) | easy |
| E07-T2 | align neural activity to stimulus changes | response latency accuracy | medium |
| E07-T3 | classify cell types by response properties | classifier accuracy | hard |
| E07-T4 | detect omission responses | sensitivity/specificity | hard |
| E07-T5 | model prediction error signaling | explained variance | hard |
| E07-T6 | handle session-to-session variability | cross-session correlation | hard |

**failure modes**:
- neuropil contamination in ΔF/F
- incorrect baseline estimation
- overfitting to session-specific artifacts
- misaligned behavioral timestamps

**cross-references**:
- NMA notebook: `projects/neurons/load_Allen_Visual_Behavior_from_SDK.ipynb`
- student project: hosseindamavandi/NMA-Allen-Visual-Behavior-2p

---

### E08: zhong 80k neuron visual learning

**customer alignment**: ★★★★☆ hill-climbing (large-scale)

**dataset**: stringer/zhong 80k neuron visual learning
- source: figshare (janelia)
- format: numpy arrays
- size: ~20GB

**infrastructure**: CPU-only, dimensionality reduction heavy

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E08-T1 | load and preprocess 80k neuron dataset | data integrity check | easy |
| E08-T2 | efficient PCA on large population | variance explained | medium |
| E08-T3 | incremental PCA for memory efficiency | throughput + accuracy | hard |
| E08-T4 | decode visual features from population | decoding R² | hard |
| E08-T5 | track learning-related changes | pre/post learning divergence | hard |

**failure modes**:
- memory overflow on naive covariance
- numerical instability in PCA
- confounding learning with drift
- incorrect trial alignment

**cross-references**:
- NMA notebook: `projects/neurons/visual_learning_80k_neurons.ipynb`
- NMA notebook: `projects/neurons/load_stringer_orientations.ipynb`

---

### E09: HMM latent state detection

**customer alignment**: ★★★★☆ hill-climbing + scientific reasoning

**dataset**: steinmetz neuropixels or simulated
- source: steinmetz 2019 or IBL
- format: spike times, behavioral events

**infrastructure**: CPU-only, hmmlearn or custom JAX

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E09-T1 | fit gaussian HMM to neural population | log-likelihood | medium |
| E09-T2 | select optimal number of states | BIC/AIC + interpretability | hard |
| E09-T3 | Poisson HMM for spike counts | goodness of fit | hard |
| E09-T4 | align HMM states to behavioral events | state-behavior correlation | hard |
| E09-T5 | cross-validate HMM across sessions | generalization score | hard |
| E09-T6 | compare HMM to switching linear dynamical system | model comparison | hard |

**failure modes**:
- overfitting with too many states
- EM stuck in local minima
- states not behaviorally meaningful
- temporal structure artifacts

**cross-references**:
- student project: NMA2022Chaosfan/HMM_SteinmetzDataset
- student project: ivanorsolic/NeuromatchAcademy
- NMA notebook: `projects/neurons/load_steinmetz_decisions.ipynb`

---

### E10: IBL behavior modeling

**customer alignment**: ★★★★☆ scientific reasoning

**dataset**: IBL behavioral data (wheel, licks, choices)
- source: IBL ONE API
- format: parquet

**infrastructure**: CPU-only, scipy optimization

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E10-T1 | fit psychometric function to choices | deviance explained | medium |
| E10-T2 | model block-wise prior effects | bias estimation accuracy | hard |
| E10-T3 | reaction time modeling | RT prediction R² | hard |
| E10-T4 | detect lapses and biases | lapse rate estimation | medium |
| E10-T5 | bayesian ideal observer model | negative log-likelihood | hard |

**failure modes**:
- psychometric fit non-convergence
- ignoring temporal structure
- overfitting session-specific biases
- incorrect lapse model

**cross-references**:
- NMA notebook: `projects/behavior_and_theory/IBL_behavior_data.ipynb`
- website template: `website_projects/behavior_and_theory/mouse_behavior_2afc/`

---

### E11: neural decoding benchmark

**customer alignment**: ★★★★☆ hill-climbing

**dataset**: multi-dataset (steinmetz, allen, IBL)
- format: standardized spike/calcium data

**infrastructure**: CPU + sklearn/JAX

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E11-T1 | logistic regression decoder | AUC | easy |
| E11-T2 | regularization tuning (L1/L2) | CV accuracy | medium |
| E11-T3 | temporal kernel optimization | time-resolved AUC | hard |
| E11-T4 | population vs single-unit comparison | decoder improvement | medium |
| E11-T5 | cross-session generalization | transfer accuracy | hard |
| E11-T6 | neural network decoder (small MLP) | accuracy + calibration | hard |

**failure modes**:
- train/test temporal leakage
- ignoring trial structure
- overfitting high-dimensional data
- incorrect cross-validation scheme

**cross-references**:
- student project: seungjaeryanlee/nma-cn-project
- course: ebatty/EncodingDecodingNotes

---

### E12: fMRI GLM + RDM analysis

**customer alignment**: ★★★★☆ scientific reasoning

**dataset**: HCP task fMRI
- source: HCP connectome DB
- format: CIFTI/NIfTI

**infrastructure**: CPU + nilearn

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E12-T1 | design matrix construction | model specification correctness | medium |
| E12-T2 | first-level GLM estimation | contrast map quality | medium |
| E12-T3 | group-level analysis | effect size + statistical power | hard |
| E12-T4 | representational similarity analysis | RDM correlation | hard |
| E12-T5 | searchlight RSA | voxel-wise significance | hard |

**failure modes**:
- incorrect HRF modeling
- motion artifact contamination
- multiple comparisons problem
- RDM distance metric choice

**cross-references**:
- NMA notebook: `projects/fMRI/load_hcp_task.ipynb`
- website template: `website_projects/fMRI/mapping-brain-representation/`

---

## part 3: theory/simulation environments (tier 3)

### E13: retinotopic mapping pRF

**customer alignment**: ★★★★☆ scientific reasoning

**dataset**: HCP retinotopy data
- source: HCP
- format: CIFTI

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E13-T1 | implement pRF model | position accuracy | medium |
| E13-T2 | nonlinear optimization for pRF params | R² improvement | hard |
| E13-T3 | compare pRF models (gaussian vs DoG) | model comparison | hard |
| E13-T4 | map visual field coverage | coverage completeness | medium |

**cross-references**:
- NMA notebook: `projects/fMRI/load_hcp_retino.ipynb`
- website template: `website_projects/fMRI/retinotopic-mapping/`

---

### E14: ECoG face perception psychometrics

**customer alignment**: ★★★★☆ scientific reasoning

**dataset**: ECoG faces/houses dataset
- source: course materials
- format: matlab/.mat files

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E14-T1 | extract face-selective electrodes | selectivity index | medium |
| E14-T2 | temporal dynamics of face processing | latency estimation | hard |
| E14-T3 | compare face vs house responses | d-prime | medium |
| E14-T4 | decode face identity from ECoG | classification accuracy | hard |

**cross-references**:
- NMA notebook: `projects/ECoG/load_ECoG_faceshouses.ipynb`
- website template: `website_projects/ECoG/NeuralBasisOfFacePerception/`

---

### E15: navigational affordances RDM searchlight

**customer alignment**: ★★★★☆ scientific reasoning

**dataset**: bonner & epstein navigational affordances
- source: course materials
- format: fMRI + stimuli

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E15-T1 | compute scene RDMs | RDM validity | medium |
| E15-T2 | searchlight RSA across cortex | peak detection | hard |
| E15-T3 | compare DNN vs behavioral RDMs | correlation difference | hard |
| E15-T4 | layer-wise DNN comparison | best layer identification | hard |

**cross-references**:
- NMA notebook: `projects/fMRI/load_bonner_navigational_affordances.ipynb`
- website template: `website_projects/fMRI/navigational-affordances/`

---

### E16: bayesian estimation error modeling

**customer alignment**: ★★★★☆ scientific reasoning

**dataset**: laquitaine human estimation data
- source: course materials
- format: behavioral responses

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E16-T1 | fit psychometric bias model | parameter recovery | medium |
| E16-T2 | bayesian ideal observer | likelihood ratio | hard |
| E16-T3 | compare models (ML vs Bayesian) | AIC/BIC | hard |
| E16-T4 | prior learning dynamics | learning rate estimation | hard |

**cross-references**:
- NMA notebook: `projects/behavior_and_theory/laquitaine_human_errors.ipynb`
- website template: `website_projects/behavior_and_theory/human_estimation_error_bayesian/`

---

### E17: circular statistics pipeline

**customer alignment**: ★★★★☆ data science

**dataset**: estimation error data (circular)
- source: laquitaine or simulated
- format: angular measurements

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E17-T1 | implement circular mean/variance | numerical accuracy | easy |
| E17-T2 | von Mises distribution fitting | KS test p-value | medium |
| E17-T3 | circular-linear regression | R² on circular data | hard |
| E17-T4 | handle wrap-around in angular data | prediction error | medium |

**failure modes** (common bug category):
- using linear mean on circular data
- incorrect angle wrapping
- Cartesian vs polar confusion

**cross-references**:
- NMA notebook: `projects/behavior_and_theory/laquitaine_motion_prior_learning.ipynb`

---

### E18: spike sorting QC (bombcell metrics)

**customer alignment**: ★★★★☆ data quality

**dataset**: neuropixels spike-sorted data
- source: IBL or steinmetz
- format: phy outputs

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E18-T1 | compute ISI violation rate | metric accuracy | easy |
| E18-T2 | amplitude cutoff detection | threshold optimization | medium |
| E18-T3 | waveform quality metrics | correlation with manual | hard |
| E18-T4 | automated unit classification | agreement with experts | hard |

**cross-references**:
- awesome list: Julie-Fabre/awesome_neuropixels
- tool: bombcell (Allen Institute)

---

### E19: motor RNN training (JAX)

**customer alignment**: ★★★☆☆ theory + training optimization

**dataset**: simulated motor trajectories
- source: generated
- format: numpy arrays

**infrastructure**: TPU/CPU, JAX + optax

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E19-T1 | implement vanilla RNN in JAX | test trajectory MSE | medium |
| E19-T2 | add target-based training | reach accuracy | medium |
| E19-T3 | regularize for low-dimensional dynamics | dimensionality | hard |
| E19-T4 | train GRU for longer sequences | sequence loss | hard |
| E19-T5 | analyze fixed points | fixed point count | hard |

**cross-references**:
- NMA notebook: `projects/behavior_and_theory/motor_RNNs.ipynb`
- website template: `website_projects/behavior_and_theory/structure_dynamics_function_motor_rnn/`
- student project: wl17443/ring-attractor (Julia reference)

---

### E20: working memory attractor models

**customer alignment**: ★★★☆☆ theory

**dataset**: simulated attractor networks
- source: generated
- format: numpy arrays

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E20-T1 | implement bump attractor | stability duration | medium |
| E20-T2 | tune connectivity for persistence | persistence time | hard |
| E20-T3 | add noise and measure drift | drift rate | hard |
| E20-T4 | compare continuous vs discrete attractors | capacity | hard |

**cross-references**:
- website template: `website_projects/behavior_and_theory/working_memory_attractor_models/`
- student project: wl17443/ring-attractor

---

### E21: ring attractor networks

**customer alignment**: ★★★☆☆ theory + parameter tuning

**dataset**: simulated
- format: parameters for leaky integrate-and-fire

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E21-T1 | implement ring attractor (JAX) | heading accuracy | hard |
| E21-T2 | parameter search for stability | stability region size | hard |
| E21-T3 | add external input integration | velocity integration error | hard |
| E21-T4 | compare to empirical head direction | correlation | hard |

**cross-references**:
- student project: wl17443/ring-attractor (Julia implementation)
- website template: `website_projects/behavior_and_theory/working_memory_attractor_models/`

---

### E22: RNN working memory capacity

**customer alignment**: ★★★☆☆ theory + capacity limits

**dataset**: simulated memory tasks
- format: numpy arrays

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E22-T1 | implement N-back task generator | task validity | easy |
| E22-T2 | train RNN on N-back | accuracy vs N | medium |
| E22-T3 | measure information capacity | bits stored | hard |
| E22-T4 | compare architectures (LSTM vs GRU) | capacity comparison | hard |

**cross-references**:
- NMA notebook: `projects/behavior_and_theory/RNN_working_memory.ipynb`
- website template: `website_projects/behavior_and_theory/working_memory_rnns/`

---

## part 4: cross-domain benchmark environments

### E23: multi-dataset decoding benchmark

**customer alignment**: ★★★★☆ hill-climbing

**dataset**: steinmetz + allen + IBL (standardized)
- format: unified spike/calcium format

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E23-T1 | implement standard decoder API | API compliance | easy |
| E23-T2 | benchmark across datasets | average AUC | hard |
| E23-T3 | transfer learning across datasets | transfer improvement | hard |
| E23-T4 | meta-learning for neural data | few-shot accuracy | hard |

---

### E24: foundation model probing (neural)

**customer alignment**: ★★★★☆ implement ML papers

**dataset**: neural data + pretrained foundation model
- source: CEBRA, NDT, or similar

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E24-T1 | extract embeddings from foundation model | embedding quality | medium |
| E24-T2 | linear probe on downstream task | probe accuracy | medium |
| E24-T3 | compare foundation vs trained-from-scratch | accuracy delta | hard |
| E24-T4 | fine-tune foundation model (small) | fine-tune improvement | hard |

**cross-references**:
- awesome list: mazabou/awesome-neurofm

---

### E25: DNN-brain alignment comparison

**customer alignment**: ★★★★☆ scientific reasoning

**dataset**: algonauts/kay + DNN features
- format: fMRI + extracted features

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E25-T1 | extract features from multiple DNNs | feature validity | easy |
| E25-T2 | compare architectures (CNN vs ViT vs CLIP) | brain alignment score | hard |
| E25-T3 | layer-wise alignment analysis | best layer per ROI | hard |
| E25-T4 | optimize feature combination | combined R² | hard |

**cross-references**:
- NMA notebook: `projects/fMRI/load_kay_images.ipynb`
- awesome list: CYHSM/awesome-neuro-ai-papers

---

### E26: cichy fMRI-MEG fusion

**customer alignment**: ★★★★☆ multi-modal integration

**dataset**: cichy 2014 fMRI+MEG
- source: course materials
- format: preprocessed beta weights + MEG

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E26-T1 | temporal RDM from MEG | RDM quality | medium |
| E26-T2 | spatial RDM from fMRI | RDM quality | medium |
| E26-T3 | fuse spatial + temporal information | fusion improvement | hard |
| E26-T4 | compare representational geometry | RSA correlation | hard |

**cross-references**:
- NMA notebook: `projects/fMRI/load_cichy_fMRI_MEG.ipynb`
- NMA notebook: `projects/fMRI/load_cichy2014_fMRI_MEG.ipynb`

---

### E27: ECoG task comparison

**customer alignment**: ★★★★☆ multi-task analysis

**dataset**: ECoG across multiple tasks (motor imagery, memory, tracking)
- source: course materials
- format: ECoG + task events

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E27-T1 | load + align multiple ECoG datasets | alignment accuracy | medium |
| E27-T2 | compare feature importance across tasks | feature ranking | hard |
| E27-T3 | transfer decoder between tasks | transfer accuracy | hard |
| E27-T4 | identify task-invariant representations | invariance score | hard |

**cross-references**:
- NMA notebooks: `projects/ECoG/load_ECoG_*.ipynb` (multiple)

---

## part 5: tool/infrastructure environments

### E28: efficient data loading pipeline

**customer alignment**: ★★★★☆ training optimization

**dataset**: any large neural dataset
- format: various

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E28-T1 | implement streaming loader | throughput (samples/sec) | medium |
| E28-T2 | add prefetching | latency reduction | hard |
| E28-T3 | multiprocess data loading | speedup factor | hard |
| E28-T4 | handle variable-length sequences | padding efficiency | medium |

**cross-references**:
- customer quote: "data loading: single-threaded → add multiprocessing, prefetch"

---

### E29: preprocessing vectorization

**customer alignment**: ★★★★☆ inference optimization

**dataset**: neural time series
- format: numpy arrays

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E29-T1 | vectorize python loop → numpy | speedup factor | medium |
| E29-T2 | convert numpy → JAX vmap | additional speedup | hard |
| E29-T3 | batch processing optimization | throughput | hard |
| E29-T4 | memory-efficient windowing | memory reduction | medium |

**cross-references**:
- customer quote: "row-by-row python loops → vectorized numpy"

---

### E30: batching optimization

**customer alignment**: ★★★★☆ training optimization

**dataset**: variable-length neural sequences
- format: spike trains or calcium traces

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E30-T1 | implement dynamic batching | padding waste reduction | hard |
| E30-T2 | length-sorted batching | training speedup | medium |
| E30-T3 | compare padding strategies | efficiency score | medium |

**cross-references**:
- customer quote: "naive uniform batching → dynamic batching by sequence length"

---

### E31: embedding caching

**customer alignment**: ★★★★☆ inference optimization

**dataset**: repeated-stimulus experiments
- format: stimuli + neural responses

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E31-T1 | implement disk cache for embeddings | cache hit rate | medium |
| E31-T2 | memory-mapped embedding storage | access latency | hard |
| E31-T3 | cache invalidation logic | correctness | hard |

**cross-references**:
- customer quote: "recomputing embeddings every epoch → cache to disk"

---

### E32: JAX parallelism (vmap/pmap)

**customer alignment**: ★★★☆☆ kernel tasks

**dataset**: any parallelizable computation
- format: JAX arrays

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E32-T1 | convert sequential → vmap | speedup factor | medium |
| E32-T2 | implement pmap for multi-device | scaling efficiency | hard |
| E32-T3 | combine vmap + pmap | total throughput | hard |
| E32-T4 | pallas kernel for custom op | TPU utilization | hard |

**cross-references**:
- customer quote: "sequential ops that could be vmap'd or pmap'd"

---

## part 6: additional specialized environments

### E33: stringer spontaneous activity

**customer alignment**: ★★★★☆ large-scale analysis

**dataset**: stringer spontaneous recordings (~10k neurons)
- source: figshare
- format: numpy

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E33-T1 | dimensionality analysis | optimal dimensions | medium |
| E33-T2 | detect behavioral correlates | correlation strength | hard |
| E33-T3 | compare stimulus-evoked vs spontaneous | variance explained | hard |

**cross-references**:
- NMA notebook: `projects/neurons/load_stringer_spontaneous.ipynb`

---

### E34: steinmetz extra analysis

**customer alignment**: ★★★★☆ comprehensive analysis

**dataset**: extended steinmetz dataset
- source: figshare
- format: numpy

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E34-T1 | region-by-region comparison | effect size per region | medium |
| E34-T2 | choice probability analysis | CP significance | hard |
| E34-T3 | sequential sampling model | model fit | hard |

**cross-references**:
- NMA notebook: `projects/neurons/load_steinmetz_extra.ipynb`

---

### E35: HCP behavior-fMRI correlation

**customer alignment**: ★★★★☆ brain-behavior

**dataset**: HCP task with behavior
- source: HCP
- format: CIFTI + behavioral

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E35-T1 | extract behavioral measures | validity | easy |
| E35-T2 | brain-behavior correlation | correlation strength | medium |
| E35-T3 | multivariate brain-behavior | CCA score | hard |

**cross-references**:
- NMA notebook: `projects/fMRI/load_hcp_task_with_behaviour.ipynb`

---

### E36: ECoG finger flex decoding

**customer alignment**: ★★★★☆ continuous decoding

**dataset**: ECoG finger flexion
- source: course materials
- format: ECoG + finger positions

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E36-T1 | continuous position decoding | R² | medium |
| E36-T2 | velocity decoding | correlation | hard |
| E36-T3 | multi-finger prediction | joint R² | hard |

**cross-references**:
- NMA notebook: `projects/ECoG/load_ECoG_fingerflex.ipynb`

---

### E37: ECoG joystick tracking

**customer alignment**: ★★★★☆ motor control

**dataset**: ECoG joystick tracking
- source: course materials
- format: ECoG + joystick

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E37-T1 | decode joystick position | tracking error | medium |
| E37-T2 | temporal dynamics of control | latency estimation | hard |
| E37-T3 | compare electrode montages | decoder performance | hard |

**cross-references**:
- NMA notebook: `projects/ECoG/load_ECoG_joystick_track.ipynb`

---

### E38: ECoG motor imagery BCI

**customer alignment**: ★★★★☆ BCI

**dataset**: ECoG motor imagery
- source: course materials
- format: ECoG + imagery labels

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E38-T1 | imagery classification | accuracy | medium |
| E38-T2 | feature selection for BCI | information transfer rate | hard |
| E38-T3 | online simulation | latency + accuracy | hard |

**cross-references**:
- NMA notebook: `projects/ECoG/load_ECoG_motor_imagery.ipynb`

---

### E39: ECoG N-back memory

**customer alignment**: ★★★★☆ working memory

**dataset**: ECoG N-back task
- source: course materials
- format: ECoG + task events

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E39-T1 | decode memory load | classification accuracy | hard |
| E39-T2 | track theta/gamma dynamics | band power correlation | hard |
| E39-T3 | compare load levels | effect size | medium |

**cross-references**:
- NMA notebook: `projects/ECoG/load_ECoG_memory_nback.ipynb`

---

### E40: FSL course fMRI basics

**customer alignment**: ★★★☆☆ educational

**dataset**: FSL course data
- source: FSL
- format: NIfTI

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E40-T1 | preprocessing pipeline | motion parameters | easy |
| E40-T2 | first-level analysis | activation maps | medium |
| E40-T3 | ROI extraction | signal quality | medium |

**cross-references**:
- NMA notebook: `projects/fMRI/load_fslcourse.ipynb`

---

### E41: modeling steps framework

**customer alignment**: ★★★☆☆ methodology

**dataset**: train illusion data
- source: course materials
- format: behavioral

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E41-T1 | hypothesis formalization | model specification | medium |
| E41-T2 | model comparison | Bayes factor | hard |
| E41-T3 | parameter recovery | recovery accuracy | hard |

**cross-references**:
- NMA notebook: `projects/modelingsteps/ModelingSteps_*.ipynb`
- NMA notebook: `projects/modelingsteps/TrainIllusionDataProject.ipynb`

---

### E42: supervised vs unsupervised learning comparison

**customer alignment**: ★★★★☆ methodology

**dataset**: steinmetz or similar
- format: labeled neural data

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E42-T1 | compare PCA vs LDA | explained variance vs separation | medium |
| E42-T2 | cluster validation | cluster quality metrics | hard |
| E42-T3 | semi-supervised approaches | accuracy vs labeled fraction | hard |

**cross-references**:
- website template: `website_projects/neurons/supervised-unsupervised-learning/`

---

### E43: stimulus-context-behavior integration

**customer alignment**: ★★★★☆ multi-factor

**dataset**: IBL or similar
- format: neural + behavioral + stimulus

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E43-T1 | disentangle stimulus vs context | variance partitioning | hard |
| E43-T2 | interaction effects | interaction strength | hard |
| E43-T3 | temporal evolution of representations | trajectory analysis | hard |

**cross-references**:
- website template: `website_projects/neurons/stimulus-context-behavior/`

---

### E44: visual information across regions

**customer alignment**: ★★★★☆ multi-region

**dataset**: fMRI visual responses
- format: fMRI + stimuli

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E44-T1 | extract visual ROIs | ROI quality | medium |
| E44-T2 | compare information content across hierarchy | mutual information | hard |
| E44-T3 | feedforward vs feedback signatures | temporal analysis | hard |

**cross-references**:
- website template: `website_projects/fMRI/visual-information-regions/`

---

### E45: motion prior learning dynamics

**customer alignment**: ★★★★☆ learning

**dataset**: laquitaine prior learning
- source: course materials
- format: behavioral

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E45-T1 | fit learning rate model | parameter recovery | hard |
| E45-T2 | compare learning models | model comparison | hard |
| E45-T3 | detect non-stationarity | change point detection | hard |

**cross-references**:
- NMA notebook: `projects/behavior_and_theory/laquitaine_motion_prior_learning.ipynb`

---

## skill coverage matrix (updated with E46-E62)

| skill | environments | primary tier |
|-------|--------------|--------------|
| multi-format data wrangling | E01, E02, E07, E27, E46, E48, E50, E52, E55, E56 | 1 |
| feature engineering | E06, E14, E27, E50, E57 | 1 |
| ML paper implementation | E03, E05, E24, E46, E47, E49, E52, E54, E56, E61 | 1 |
| data curation/filtering | E04, E18, E47, E48, E50, E53, E55 | 1 |
| decoding/classification | E11, E23, E36, E37, E38, E46, E51, E53, E60 | 2 |
| statistical modeling (GLM) | E12, E16, E41, E46 | 2 |
| dimensionality reduction | E08, E33, E42, E49 | 2 |
| HMM/state inference | E09, E10, E47 | 2 |
| RNN/LSTM training | E19, E22, E49, E52, E54 | 1-3 |
| dynamical systems | E20, E21, E46, E59 | 3 |
| representational analysis | E15, E25, E26 | 2 |
| optimization/speedup | E28, E29, E30, E31, E32 | 1-2 |
| circular statistics | E17 | 3 |
| BCI/real-time | E38, E39 | 2 |
| foundation models | E24, E49, E54 | 1 |
| broken code debugging | E58, E59 | 1 |
| spectral analysis (LFP) | E57 | 2 |
| graph neural networks | E52 | 2 |

---

## dataset access patterns

| dataset | access method | auth required | streaming |
|---------|--------------|---------------|-----------|
| AJILE12 | DANDI API | no | yes |
| IBL | ONE API | yes (free) | yes |
| Allen 2P | Allen SDK | no | partial |
| Steinmetz | figshare | no | no |
| Stringer | figshare | no | no |
| HCP | ConnectomeDB | yes | no |
| CalMS21 | figshare | no | no |
| Algonauts | project site | no | no |
| Laquitaine | course files | no | no |
| ECoG (various) | course files | no | no |

---

## infrastructure requirements summary

| requirement | environments affected | notes |
|-------------|----------------------|-------|
| CPU-only | all | baseline |
| TPU v5e (2x2) | E05, E19, E32 | JAX/pjit |
| >4GB RAM | E02, E07, E08 | population analysis |
| >8GB RAM | E08 | 80k neuron PCA |
| ONE API access | E02, E10 | free registration |
| Allen SDK | E07 | pip install |
| streaming support | E01, E02 | large datasets |
| scipy.optimize | E10, E13, E16 | parameter fitting |
| hmmlearn | E09 | HMM fitting |
| nilearn | E12, E40 | fMRI analysis |
| JAX + optax | E19, E20, E21, E32 | RNN training |

---

## implementation roadmap

### phase 1: tier 1 environments (weeks 1-4)

| week | environments | total tasks |
|------|--------------|-------------|
| 1 | E01 (AJILE12), E06 (CalMS21) | 13 |
| 2 | E02 (IBL), E04 (data curation) | 14 |
| 3 | E03 (algonauts), E05 (paper impl) | 13 |
| 4 | integration + testing | - |

### phase 2: tier 2 environments (weeks 5-8)

| week | environments | total tasks |
|------|--------------|-------------|
| 5 | E07 (Allen 2P), E09 (HMM) | 12 |
| 6 | E11 (decoding), E12 (GLM) | 11 |
| 7 | E28-31 (optimization) | 12 |
| 8 | integration + testing | - |

### phase 3: tier 3 environments (weeks 9-12)

| week | environments | total tasks |
|------|--------------|-------------|
| 9 | E19-22 (RNN/attractors) | 16 |
| 10 | E13-17 (specialized) | 15 |
| 11 | E23-27 (cross-domain) | 18 |
| 12 | final integration | - |

---

## part 7: newly discovered environments (from repo evaluation)

> 17 additional environments identified from systematic evaluation of 35+ external repos
> evaluation criteria: pm_env fit ≥3.5/5, working code, CPU/TPU compatible

---

### E46: steinmetz RT prediction + neural ODE

**source**: shir-openu/steinmetz-neural-decision-signals
**rating**: 4.5/5

**dataset**: steinmetz neuropixels (VISp, MOs, MOp, ACA, SC)

**methods**: ridge regression RT prediction, neural ODE vs poisson GLM, cross-area communication

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E46-T1 | improve RT prediction R² above 0.15 | R² cross-validated | hard |
| E46-T2 | implement neural ODE with adjoint method | spike prediction MSE | hard |
| E46-T3 | find optimal neuron subset for decoding | accuracy vs subset size | hard |
| E46-T4 | cross-validate across mice (not sessions) | generalization score | hard |

**customer alignment**: ★★★★★ IMPLEMENT PAPERS, ★★★★★ MESSY DATA, ★★★★☆ HILL-CLIMBING

---

### E47: HMM latent states + LQG control

**source**: ivanorsolic/NeuromatchAcademy
**rating**: 4.5/5

**dataset**: steinmetz (MOs, VISp coverage)

**methods**: HMM for neural state detection, LQG optimal control, state-behavior correlation

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E47-T1 | fit HMM with 2-10 states, select via BIC | log-likelihood + BIC | hard |
| E47-T2 | implement forward-backward from scratch | match hmmlearn output | medium |
| E47-T3 | extend LQG to multi-dim wheel tracking | control cost J(s,a) | hard |
| E47-T4 | cross-validate HMM across sessions | generalization | hard |

**customer alignment**: ★★★★★ IMPLEMENT PAPERS, ★★★★★ DATA CURATION, ★★★★☆ SCIENTIFIC REASONING

---

### E48: HCP boredom/attention lapse detection

**source**: seannoah/nma2020-HCP-project
**rating**: 4.5/5

**dataset**: HCP fMRI (339 subjects, 360 parcels, 7 tasks)

**methods**: multi-classifier comparison (KNN, SVM, RF, MLP), FC analysis, "boredom" = no-response trials

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E48-T1 | beat 90% task state decoding accuracy | accuracy | hard |
| E48-T2 | detect boredom trials from FC patterns | sensitivity/specificity | hard |
| E48-T3 | find minimal parcel set maintaining accuracy | parcel count vs acc | hard |
| E48-T4 | predict trial accuracy from pre-stimulus | temporal prediction | hard |

**customer alignment**: ★★★★★ MESSY DATA, ★★★★★ DATA CURATION, ★★★★☆ HILL-CLIMBING

---

### E49: neural manifold LSTM autoencoder

**source**: sazio/NMAs
**rating**: 4.5/5

**dataset**: stringer spontaneous calcium imaging (10179 neurons, 9 V1 layers)

**methods**: bidirectional LSTM autoencoder, SAM optimizer, UMAP clustering, layer-specific analysis

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E49-T1 | implement LFADS-style KL regularization | reconstruction loss | hard |
| E49-T2 | sweep latent dims 25-200, find pareto frontier | compression vs quality | hard |
| E49-T3 | cross-layer transfer (train layer 1, test layer 9) | transfer accuracy | hard |
| E49-T4 | shared encoder vs 9 separate LSTMs | accuracy comparison | hard |

**customer alignment**: ★★★★★ MESSY DATA, ★★★★★ IMPLEMENT PAPERS, ★★★★☆ HILL-CLIMBING

---

### E50: ECoG novelty effect (miller 2015)

**source**: cagrioguz/NMA-Computational-Neuroscience-2023-Group-Project
**rating**: 4.5/5

**dataset**: miller ECoG faces/houses (7 subjects, 1kHz)

**methods**: broadband power extraction, novelty modeling, SVR/polynomial regression comparison

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E50-T1 | identify face-selective electrodes via CV | electrode selection F1 | hard |
| E50-T2 | test novelty encodings (binary, exponential decay) | MSE improvement | hard |
| E50-T3 | systematic model comparison with proper CV | model ranking | hard |
| E50-T4 | replicate miller 2015 novelty enhancement | paper metrics | hard |

**customer alignment**: ★★★★★ MESSY DATA, ★★★★★ IMPLEMENT PAPERS, ★★★★★ DATA CURATION

---

### E51: allen 2P image decoding

**source**: farhadpanthaki/NMA_Allen
**rating**: 4/5

**dataset**: allen visual behavior 2P (VIP/SST/excitatory cells, change detection)

**methods**: image decoding from traces, cell-type analysis, contrast regression

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E51-T1 | decode image identity across cell types | accuracy | hard |
| E51-T2 | predict hit/miss from neural traces | AUC | hard |
| E51-T3 | build encoding model: contrast → response | R² | hard |

**customer alignment**: ★★★★★ MESSY DATA, ★★★★☆ HILL-CLIMBING, ★★★★☆ SCIENTIFIC REASONING

---

### E52: GNN-LSTM working memory demand

**source**: saamehsanaaee/WMAD-Montbretia_Cabinet-ISP
**rating**: 4/5

**dataset**: HCP WM/EMOTION/LANGUAGE tasks (100 subjects, 360 parcels)

**methods**: GNN-LSTM hybrid, functional connectivity graphs, cross-task transfer

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E52-T1 | optimize FC threshold 0.1-0.5 | graph quality vs accuracy | hard |
| E52-T2 | attention over parcels for interpretability | attention weights | hard |
| E52-T3 | cross-task transfer (WM → EMOTION) | domain shift analysis | hard |
| E52-T4 | replace LSTM with transformer encoder | accuracy comparison | hard |

**customer alignment**: ★★★★★ MESSY DATA, ★★★★★ IMPLEMENT PAPERS, ★★★★☆ HILL-CLIMBING

---

### E53: steinmetz behavior classification

**source**: maryamdaniali/NeuromatchAcademy
**rating**: 4/5

**dataset**: steinmetz 2019 (39 sessions, multi-area)

**methods**: 1D CNN, temporal windowing, cross-session analysis

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E53-T1 | optimize temporal window per brain area | decoding accuracy | hard |
| E53-T2 | find best area combination for prediction | feature selection | hard |
| E53-T3 | cross-session transfer to held-out mice | generalization | hard |
| E53-T4 | reproduce steinmetz 2019 timing claims | paper metrics | hard |

**customer alignment**: ★★★★★ MESSY DATA, ★★★★☆ HILL-CLIMBING, ★★★★☆ DATA CURATION

---

### E54: foundation model for neural data (NDT-1)

**source**: patrickmineault/fmn-tutorial
**rating**: 4.5/5 ⬆️

**dataset**: DANDI neural datasets (preprocessed spike trains)

**methods**: NDT-1 transformer (Ye & Pandarinath 2021), poisson reconstruction, positional encoding, behavior decoding

**code structure** (verified via API):
- `src/ndt_transformer.py` – full NDT implementation with TransformerEncoder, PositionalEncoding, Poisson loss
- `src/ndt_reimplementation.py` – simplified single-file reimplementation
- `tutorials/Tutorial_Student.ipynb` – guided student version
- `tutorials/Tutorial_Solution.ipynb` – complete solutions

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E54-T1 | implement NDT-1 encoder from Ye & Pandarinath 2021 | reconstruction loss | hard |
| E54-T2 | optimize positional encoding (learnable vs sinusoidal) | accuracy delta | hard |
| E54-T3 | tune transformer hyperparams (heads, layers, hidden) | poisson NLL | hard |
| E54-T4 | behavior decoder architecture search | R² | hard |
| E54-T5 | cross-session transfer learning with frozen encoder | transfer improvement | hard |

**key classes to implement**:
- `PositionalEncoding` with learnable/fixed modes
- `TransformerEncoderLayerWithHooks` for attention visualization
- Poisson likelihood output layer

**customer alignment**: ★★★★★ IMPLEMENT PAPERS, ★★★★★ TRAINING OPT, ★★★★☆ MESSY DATA

---

### E55: multi-format neural data science (CSHL 2019)

**source**: marius10p/NeuralDataScienceCSHL2019
**rating**: 4.5/5 ⬆️

**dataset**: allen observatory 2P, neuropixels, monkey ephys, suite2p calcium imaging

**methods**: GPFA (Byron Yu), visual coding analysis, motion correction, multi-modal integration

**repo structure** (verified via API):
- `AllenObservatory/` – 3 CSHL notebooks + solutions + Visual Coding 2P Cheat Sheet
- `ByronYuExercises/` – `exercises.pdf` (50KB) + `data_for_exercises.mat` + `fa/` (factor analysis)
- `Suite2p/` – motion_correction.ipynb
- `Neuropixels/` – Demos/, Utilities/, myFunctions/, SteHaaCar2018.pdf, howtoreadLFP.txt
- `MonkeyEphys/` – primate electrophysiology
- `StatisticalClinics/` – stats refresher

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E55-T1 | implement GPFA from Byron Yu exercises | latent R² | hard |
| E55-T2 | allen observatory 2P visual decoder | accuracy | hard |
| E55-T3 | suite2p motion correction optimization | registration quality | hard |
| E55-T4 | neuropixels LFP preprocessing pipeline | SNR | hard |
| E55-T5 | cross-modality transfer (2P → neuropixels) | domain alignment | hard |
| E55-T6 | factor analysis dimensionality selection | AIC/BIC | medium |

**NOTE**: this is a COMPREHENSIVE multi-format course – perfect for "messy data wrangling" tasks

**customer alignment**: ★★★★★ MESSY DATA, ★★★★★ DATA CURATION, ★★★★★ IMPLEMENT PAPERS

---

### E56: computational neuroscience course (UVA)

**source**: melizalab/comp-neurosci
**rating**: 4.5/5 ⬆️

**dataset**: starling (extracellular, pprox, spikes, stimuli) + zebra finch (intracellular)

**methods**: receptive fields, LTI systems, decoding models, dynamical systems, spiking neural networks

**repo structure** (verified via API – 8 comprehensive notebooks):
- `1-Time-Varying-Data.ipynb` – spike stats, rate estimation
- `2-Data-and-Spike-Stats.ipynb` – fano factor, CV, ISI
- `3-Linear-Time-Invariant-Systems.ipynb` – convolution, filters
- `4-Receptive-Fields.ipynb` – STA, spatiotemporal RFs (uses `filters.npz`)
- `5-Decoding-Models.ipynb` – bayesian inference, population vector
- `6-Optional-Dynamical-Systems.ipynb` – phase portraits
- `7-Reduced-Dynamical-Models.ipynb` – dimensionality reduction
- `8-Spiking-Neural-Networks.ipynb` – I&F, network dynamics
- `data/starling/` – extracellular, pprox, spikes, stimuli subdirs
- `data/zf/intracellular/` – zebra finch intracellular recordings
- `comp_neurosci_uva/` – helper module with `signal` and `data` submodules

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E56-T1 | estimate spatiotemporal RF via reverse correlation | R² | hard |
| E56-T2 | bayesian decoder vs population vector comparison | accuracy delta | hard |
| E56-T3 | compute neural information rate (bits/spike) | mutual information | hard |
| E56-T4 | LNP model with static nonlinearity | prediction R² | hard |
| E56-T5 | spiking network synchronization analysis | coherence | hard |
| E56-T6 | fit integrate-and-fire model to intracellular data | spike timing error | hard |
| E56-T7 | dynamical systems phase portrait analysis | fixed point accuracy | medium |

**NOTE**: full computational neuroscience curriculum with real avian data – excellent for scientific reasoning tasks

**customer alignment**: ★★★★★ MESSY DATA, ★★★★★ IMPLEMENT PAPERS, ★★★★★ SCIENTIFIC REASONING

---

### E57: LFP spectral + granger causality

**source**: kayteeflick/nma_cn_flow_state
**rating**: 3.5/5

**dataset**: steinmetz LFP data

**methods**: PSD, LFP coherence, granger causality, phase coupling

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E57-T1 | wavelet coherence for time-frequency | coherence map quality | hard |
| E57-T2 | optimize frequency bands for classification | accuracy | hard |
| E57-T3 | multivariate granger causality | F-statistic | hard |
| E57-T4 | LFP phase → RT prediction | correlation | hard |

**customer alignment**: ★★★★★ MESSY DATA, ★★★★☆ IMPLEMENT PAPERS, ★★★★☆ SCIENTIFIC REASONING

---

### E58: HCP empathy RT regression (broken → fix)

**source**: vijpandaturtle/empathy-hcp-neuromatch
**rating**: 3.5/5

**dataset**: HCP SOCIAL task (mental vs random)

**methods**: logistic regression (99% saturated), 1D CNN, RT regression (currently R² = -1.17!)

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E58-T1 | fix RT prediction overfit → positive R² | R² | hard |
| E58-T2 | compare linear vs CNN representations | accuracy delta | medium |

**customer alignment**: ★★★★★ HILL-CLIMBING (broken baseline!), ★★★★★ MESSY DATA

**NOTE**: this is a PERFECT "fix the bug" task – student inherits broken code with negative R²

---

### E59: hopfield attractor networks

**source**: doomgutt/NMA-CN-2022-Project--Group-A---Attractor-Networks
**rating**: 3.5/5

**dataset**: MNIST, custom alphabet, hand-drawn images

**methods**: hopfield with hebbian/storkey rules, async/sync activation, energy analysis

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E59-T1 | implement pseudo-inverse rule (TODO in code) | capacity improvement | hard |
| E59-T2 | optimize noise tolerance at 30% noise | recall accuracy | hard |
| E59-T3 | fix storkey rule bug | benchmark match | medium |

**customer alignment**: ★★★★☆ IMPLEMENT PAPERS, ★★★☆☆ HILL-CLIMBING

---

### E60: allen neuropixels V1 decoding

**source**: MartynasArba/NMA_DL_project
**rating**: 3.5/5

**dataset**: allen neuropixels visual coding (15K stimuli, 93 neurons)

**methods**: CNN with 5-dim latent, L2 reg, early stopping

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E60-T1 | optimize neuron filtering criteria | accuracy | hard |
| E60-T2 | find minimum latent space for >95% acc | compression | hard |
| E60-T3 | cross-animal generalization | transfer accuracy | hard |

**customer alignment**: ★★★★☆ MESSY DATA, ★★★★☆ HILL-CLIMBING

---

### E61: normalizing flows for neural ICA

**source**: patrickmineault/xcorr-notebooks
**rating**: 3.5/5

**dataset**: olivetti faces, synthetic ICA signals

**methods**: NICE/LICE normalizing flows, NMF via gradient descent, thompson sampling

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E61-T1 | implement localizing NMF from scratch | reconstruction MSE | hard |
| E61-T2 | LU-decomposition flow for ICA | component recovery % | hard |
| E61-T3 | thompson sampling regret optimization | regret vs optimal | hard |

**NOTE**: explicit failure mode documented – 1/3 success rate for ICA demixing

**customer alignment**: ★★★★★ IMPLEMENT PAPERS, ★★★★★ HILL-CLIMBING (multiple restarts needed)

---

### E62: bristol computational neuroscience (fly H1 motion detection)

**source**: conorhoughton/COMS30127
**rating**: 4/5 ⬆️

**dataset**: fly H1 neuron from de Ruyter van Steveninck et al.
- `rho.dat` – 1.2MB spike train (600,000 samples @ 500Hz = 20 min recording)
- `stim.dat` – 8MB stimulus (visual motion velocity)

**methods**: spike statistics, poisson processes, spike-triggered average, integrate-and-fire

**repo structure** (verified via API – comprehensive course 2013-2018):
- `2017_18/coursework/spike_trains/` – main H1 dataset
  - `load.py` / `load.jl` – simple text loaders
  - `poisson.py` / `poisson.jl` – poisson process with refractory period
  - `coursework2.pdf` / `.tex` – detailed assignment specs
- `2017_18/10_spike_trains/` – spike train theory
- `2017_18/11_integrate_and_fire/` – I&F neuron models
- `2017_18/12_synapses/` – synaptic dynamics
- `2017_18/13_HH_ionchannels/` – Hodgkin-Huxley
- `2017_18/14_phase_analysis/` – oscillation analysis
- `2017_18/9_visual_system/` – visual processing context

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E62-T1 | compute fano factor, CV, compare to poisson | theoretical match | medium |
| E62-T2 | estimate STA for motion direction | RF quality | hard |
| E62-T3 | fit I&F model to spike timing | spike timing error (ms) | hard |
| E62-T4 | implement poisson process with refractory period | rate accuracy | medium |
| E62-T5 | information-theoretic analysis (bits/spike) | mutual information | hard |
| E62-T6 | coupled I&F network synchronization | phase coherence | hard |

**data format**: simple text files (int for spikes, float for stimulus) – easy to parse but raw

**customer alignment**: ★★★★★ MESSY DATA (raw text format), ★★★★★ IMPLEMENT PAPERS, ★★★★☆ SCIENTIFIC REASONING

---

### E63: EEG preprocessing and classification (MNE-Python)

**source**: https://github.com/mne-tools/mne-python (3224★)
**rating**: 4.5/5

**dataset**: MNE sample data, EEGBCI motor imagery, various clinical EEG

**methods**: artifact rejection, ICA, source localization, time-frequency, machine learning classification

**repo structure** (verified via API):
- `tutorials/preprocessing/` – filtering, resampling, artifact handling
- `tutorials/epochs/` – event-related analysis
- `tutorials/time-freq/` – spectral analysis, wavelets
- `tutorials/machine-learning/` – decoding, cross-validation
- `tutorials/stats-sensor-space/` – cluster permutation tests
- `tutorials/inverse/` – source localization

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E63-T1 | ICA artifact rejection optimization | component accuracy | hard |
| E63-T2 | motor imagery BCI classification | accuracy | hard |
| E63-T3 | time-frequency power extraction pipeline | SNR | medium |
| E63-T4 | ERP component detection (P300, N400) | latency/amplitude error | hard |
| E63-T5 | source localization (beamformer vs dSPM) | localization error | hard |
| E63-T6 | cross-subject transfer learning | generalization accuracy | hard |

**NOTE**: MNE is the standard EEG/MEG library – tasks test preprocessing decisions and ML pipeline design

**customer alignment**: ★★★★★ MESSY DATA (multi-format EEG), ★★★★★ IMPLEMENT PAPERS, ★★★★☆ HILL-CLIMBING

---

### E64: calcium imaging preprocessing (Suite2p + CaImAn)

**source**:
- https://github.com/MouseLand/suite2p (424★) – cell detection
- https://github.com/flatironinstitute/CaImAn (711★) – full pipeline

**rating**: 4.5/5

**dataset**: two-photon calcium imaging movies (simulated + real)

**methods**: motion correction, ROI detection, neuropil subtraction, spike deconvolution, demixing

**suite2p key modules**:
- `registration/` – motion correction (rigid + nonrigid)
- `detection/` – cell detection via correlation
- `extraction/` – fluorescence trace extraction
- `classification/` – cell vs non-cell classifier

**caiman key modules**:
- `motion_correction/` – NoRMCorre algorithm
- `source_extraction/` – CNMF, CNMF-E (1-photon)
- `components_evaluation/` – quality metrics
- `utils/visualization.py` – diagnostic plots

**task variants**:

| task | description | metric | difficulty |
|------|-------------|--------|------------|
| E64-T1 | motion correction parameter tuning | registration quality | hard |
| E64-T2 | ROI detection threshold optimization | F1 (cells) | hard |
| E64-T3 | neuropil subtraction coefficient | trace correlation | hard |
| E64-T4 | spike deconvolution (OASIS vs MCMC) | timing accuracy | hard |
| E64-T5 | CNMF component number selection | AIC/BIC | hard |
| E64-T6 | 1-photon vs 2-photon pipeline comparison | cell yield | hard |

**NOTE**: these are the two dominant calcium imaging pipelines – tasks test preprocessing parameter optimization

**customer alignment**: ★★★★★ MESSY DATA (large movies), ★★★★★ DATA CURATION, ★★★★★ HILL-CLIMBING

---

## updated summary statistics

| category | original envs | new envs | total |
|----------|--------------|----------|-------|
| tier 1 (★★★★★) | 6 | 13 | 19 |
| tier 2 (★★★★☆) | 12 | 5 | 17 |
| tier 3 (★★★☆☆) | 10 | 1 | 11 |
| optimization/tools | 5 | 0 | 5 |
| **TOTAL** | 45 | 19 | **64** |

**rating upgrades after API verification:**
- E54: 4/5 → 4.5/5 (full NDT-1 implementation with student/solution notebooks)
- E55: 4/5 → 4.5/5 (comprehensive multi-format course with Byron Yu GPFA)
- E56: 4/5 → 4.5/5 (8 notebooks covering full comp neuro curriculum)
- E62: 3/5 → 4/5 (20 min fly H1 dataset with detailed coursework)
- E63: NEW – MNE-Python EEG preprocessing (3224★)
- E64: NEW – Suite2p + CaImAn calcium imaging (424★ + 711★)

---

## full repository links (for development reference)

### NMA official repos
| env | repo | url |
|-----|------|-----|
| all | NeuromatchAcademy/course-content | https://github.com/NeuromatchAcademy/course-content |
| E03, E05, E24 | NeuromatchAcademy/course-content-dl | https://github.com/NeuromatchAcademy/course-content-dl |

### math/theory foundations
| env | repo | url |
|-----|------|-----|
| E16, E17, E20 | ebatty/MathToolsforNeuroscience | https://github.com/ebatty/MathToolsforNeuroscience |
| E20, E21 | wl17443/ring-attractor | https://github.com/wl17443/ring-attractor |
| all | patrickmineault/codebook | https://github.com/patrickmineault/codebook |

### HMM/steinmetz projects
| env | repo | url |
|-----|------|-----|
| E09 | NMA2022Chaosfan/HMM_SteinmetzDataset | https://github.com/NMA2022Chaosfan/HMM_SteinmetzDataset |
| E46 | shir-openu/steinmetz-neural-decision-signals | https://github.com/shir-openu/steinmetz-neural-decision-signals |
| E47 | ivanorsolic/NeuromatchAcademy | https://github.com/ivanorsolic/NeuromatchAcademy |
| E53 | maryamdaniali/NeuromatchAcademy | https://github.com/maryamdaniali/NeuromatchAcademy |

### allen/2P projects
| env | repo | url |
|-----|------|-----|
| E07 | hosseindamavandi/NMA-Allen-Visual-Behavior-2p | https://github.com/hosseindamavandi/NMA-Allen-Visual-Behavior-2p |
| E51 | farhadpanthaki/NMA_Allen | https://github.com/farhadpanthaki/NMA_Allen |
| E60 | MartynasArba/NMA_DL_project | https://github.com/MartynasArba/NMA_DL_project |

### HCP/fMRI projects
| env | repo | url |
|-----|------|-----|
| E48 | seannoah/nma2020-HCP-project | https://github.com/seannoah/nma2020-HCP-project |
| E52 | saamehsanaaee/WMAD-Montbretia_Cabinet-ISP | https://github.com/saamehsanaaee/WMAD-Montbretia_Cabinet-ISP |
| E58 | vijpandaturtle/empathy-hcp-neuromatch | https://github.com/vijpandaturtle/empathy-hcp-neuromatch |

### ECoG projects
| env | repo | url |
|-----|------|-----|
| E50 | cagrioguz/NMA-Computational-Neuroscience-2023-Group-Project | https://github.com/cagrioguz/NMA-Computational-Neuroscience-2023-Group-Project |

### neural networks/deep learning
| env | repo | url |
|-----|------|-----|
| E49 | sazio/NMAs | https://github.com/sazio/NMAs |
| E54 | patrickmineault/fmn-tutorial | https://github.com/patrickmineault/fmn-tutorial |
| E59 | doomgutt/NMA-CN-2022-Project--Group-A---Attractor-Networks | https://github.com/doomgutt/NMA-CN-2022-Project--Group-A---Attractor-Networks |
| E61 | patrickmineault/xcorr-notebooks | https://github.com/patrickmineault/xcorr-notebooks |

### comprehensive courses
| env | repo | url |
|-----|------|-----|
| E55 | marius10p/NeuralDataScienceCSHL2019 | https://github.com/marius10p/NeuralDataScienceCSHL2019 |
| E56 | melizalab/comp-neurosci | https://github.com/melizalab/comp-neurosci |
| E62 | conorhoughton/COMS30127 | https://github.com/conorhoughton/COMS30127 |

### LFP/spectral analysis
| env | repo | url |
|-----|------|-----|
| E57 | kayteeflick/nma_cn_flow_state | https://github.com/kayteeflick/nma_cn_flow_state |

### awesome lists
| env | repo | url |
|-----|------|-----|
| E18 | Julie-Fabre/awesome_neuropixels | https://github.com/Julie-Fabre/awesome_neuropixels |
| E03, E25 | CYHSM/awesome-neuro-ai-papers | https://github.com/CYHSM/awesome-neuro-ai-papers |
| E24 | mazabou/awesome-neurofm | https://github.com/mazabou/awesome-neurofm |

### preprocessing tools (E63, E64)
| env | repo | url |
|-----|------|-----|
| E63 | mne-tools/mne-python | https://github.com/mne-tools/mne-python |
| E64 | MouseLand/suite2p | https://github.com/MouseLand/suite2p |
| E64 | flatironinstitute/CaImAn | https://github.com/flatironinstitute/CaImAn |

---

## appendix A: NMA repository cross-references

| repository | environments | relevance |
|------------|--------------|-----------|
| NeuromatchAcademy/course-content | all | base notebooks |
| NeuromatchAcademy/course-content-dl | E03, E05, E24 | DL methods |
| ebatty/MathToolsforNeuroscience | E16, E17, E20 | math foundations |
| patrickmineault/codebook | all | code quality |
| wl17443/ring-attractor | E20, E21 | attractor reference |
| NMA2022Chaosfan/HMM_SteinmetzDataset | E09 | HMM reference |
| hosseindamavandi/NMA-Allen-Visual-Behavior-2p | E07 | Allen 2P reference |
| Julie-Fabre/awesome_neuropixels | E18 | spike sorting |
| CYHSM/awesome-neuro-ai-papers | E03, E25 | neuro-AI papers |
| mazabou/awesome-neurofm | E24 | foundation models |
| **shir-openu/steinmetz-neural-decision-signals** | E46 | RT prediction, neural ODE |
| **ivanorsolic/NeuromatchAcademy** | E47 | HMM + LQG |
| **seannoah/nma2020-HCP-project** | E48 | HCP boredom detection |
| **sazio/NMAs** | E49 | LSTM autoencoder calcium |
| **cagrioguz/NMA...2023-Group-Project** | E50 | ECoG novelty |
| **farhadpanthaki/NMA_Allen** | E51 | Allen 2P decoding |
| **saamehsanaaee/WMAD-Montbretia** | E52 | GNN-LSTM HCP |
| **maryamdaniali/NeuromatchAcademy** | E53 | steinmetz behavior |
| **patrickmineault/fmn-tutorial** | E54 | NDT-1 foundation model |
| **marius10p/NeuralDataScienceCSHL2019** | E55 | multi-format data |
| **melizalab/comp-neurosci** | E56 | birdsong signal detection |
| **kayteeflick/nma_cn_flow_state** | E57 | LFP spectral |
| **vijpandaturtle/empathy-hcp-neuromatch** | E58 | broken RT (hill-climb!) |
| **doomgutt/...Attractor-Networks** | E59 | hopfield networks |
| **MartynasArba/NMA_DL_project** | E60 | allen neuropixels |
| **patrickmineault/xcorr-notebooks** | E61 | normalizing flows |
| **conorhoughton/COMS30127** | E62 | fly H1 data |
| **mne-tools/mne-python** | E63 | EEG preprocessing (3224★) |
| **MouseLand/suite2p** | E64 | calcium cell detection (424★) |
| **flatironinstitute/CaImAn** | E64 | calcium CNMF pipeline (711★) |

---

## appendix B: failure mode taxonomy (updated)

| category | description | example environments |
|----------|-------------|---------------------|
| data format | parsing/alignment errors | E01, E02, E27, **E50, E55, E62** |
| class imbalance | trivial classifiers | E06, E07 |
| temporal leakage | train/test contamination | E06, E09, E11, **E53** |
| memory overflow | large data handling | E01, E08, **E49** |
| numerical precision | float32 vs float64 | E13, E17, E20, **E61** |
| cross-session drift | non-stationarity | E01, E07, E11, **E46, E53** |
| overfitting | small N, high D | E03, E09, E16, **E58 (R²=-1.17!)** |
| model selection | hyperparameter choices | E09, E11, E19, **E47, E52** |
| API rate limiting | external services | E02, E07 |
| circular statistics | angle wrapping bugs | E16, E17 |
| EM local minima | HMM/mixture convergence | E09, **E47, E61 (1/3 success rate)** |
| graph construction | FC threshold sensitivity | **E52** |
| cross-task transfer | domain shift | **E52, E53** |
| electrode selection | feature selection in ECoG | **E50** |
| broken baseline code | inheriting buggy code | **E58, E59** |
| ICA convergence | artifact rejection | **E63** |
| motion correction | registration artifacts | **E64** |
| ROI detection | cell vs noise classification | **E64** |

---

## appendix C: customer quote mapping (updated)

| customer quote | mapped environments |
|----------------|-------------------|
| "messy input data science – tasks where the agent receives data that's spread across multiple formats" | E01, E02, E06, E27, **E46, E48, E50, E52, E55 (CSHL multi-format), E56 (starling+zf), E57, E62 (fly H1 text), E63 (EEG multi-format), E64 (calcium movies)** |
| "hill-climbing on core metrics – environments with a single continuous metric" | E03, E08, E11, E23, **E46, E48, E49, E51, E54 (NDT-1 poisson NLL), E58, E60, E61, E63 (ICA optimization), E64 (ROI detection)** |
| "implement ML papers – add the technique to an existing popular repo" | E05, E24, **E46, E47, E49, E52, E54 (Ye & Pandarinath 2021), E55 (Byron Yu GPFA), E56 (LNP models), E61, E62 (I&F models), E63 (MNE tutorials), E64 (Suite2p/CaImAn)** |
| "data curation – filtering, quality" | E04, E18, **E47, E48, E50, E53, E55, E63 (artifact rejection), E64 (cell classification)** |
| "hard scientific reasoning comprehension – 10+ pages of dense scientific writing" | E12, E16, E25, **E47, E55 (exercises.pdf), E56 (8 notebooks), E62 (coursework2.pdf)** |
| "data loading: single-threaded → add multiprocessing, prefetch" | E28 |
| "preprocessing: row-by-row python loops → vectorized numpy" | E29 |
| "batching: naive uniform → dynamic batching by sequence length" | E30 |
| "caching: recomputing embeddings every epoch → cache to disk" | E31 |
| "parallelism: sequential ops that could be vmap'd or pmap'd" | E32 |
| "fix the issue and create a NEW commit" / debugging broken code | **E58 (R²=-1.17), E59 (storkey bug)** |
| "training optimization" | **E49, E54 (transformer tuning)** |

---

## appendix D: quick-start tasks (existing notebooks → envs)

these tasks require minimal setup bc notebooks already exist:

| env | task | notebook/source | time to port |
|-----|------|-----------------|--------------|
| E02-T1 | IBL session loading | IBL_ONE_tutorial.ipynb | 1 day |
| E06-T1 | CalMS21 velocity features | Loading_CalMS21_data.ipynb | 1 day |
| E07-T1 | Allen 2P trace extraction | load_Allen_Visual_Behavior_from_SDK.ipynb | 1 day |
| E09-T1 | basic HMM fit | load_steinmetz_decisions.ipynb | 2 days |
| E12-T1 | GLM design matrix | load_hcp_task.ipynb | 1 day |
| E19-T1 | vanilla RNN | motor_RNNs.ipynb | 2 days |
| **E46-T1** | RT prediction baseline | shir-openu decision_signals repo | 1 day |
| **E47-T1** | HMM state detection | ivanorsolic HMM notebook | 1 day |
| **E48-T1** | HCP classifier comparison | seannoah HCP notebooks | 1 day |
| **E49-T1** | LSTM autoencoder | sazio/NMAs notebooks | 2 days |
| **E50-T1** | ECoG novelty | cagrioguz ecog_faceshouses_main.py | 1 day |
| **E51-T1** | Allen 2P decoding | farhadpanthaki NMA_project.ipynb | 1 day |
| **E53-T1** | steinmetz behavior | maryamdaniali ClassifyActionResponse.py | 1 day |
| **E54-T1** | NDT-1 encoder | fmn-tutorial/tutorials/Tutorial_Student.ipynb | 1 day |
| **E55-T1** | GPFA implementation | NeuralDataScienceCSHL2019/ByronYuExercises/ | 1 day |
| **E55-T2** | Allen 2P visual decoding | NeuralDataScienceCSHL2019/AllenObservatory/*.ipynb | 1 day |
| **E56-T1** | receptive field STA | comp-neurosci/4-Receptive-Fields.ipynb | 0.5 day |
| **E56-T2** | decoding models | comp-neurosci/5-Decoding-Models.ipynb | 1 day |
| **E58-T1** | fix broken RT (R²=-1.17) | vijpandaturtle notebooks | 0.5 day |
| **E61-T1** | normalizing flow ICA | patrickmineault xcorr-notebooks | 2 days |
| **E62-T1** | fly H1 spike stats | COMS30127/coursework/spike_trains/ | 0.5 day |
| **E63-T1** | ICA artifact rejection | mne-python/tutorials/preprocessing/ | 1 day |
| **E63-T2** | motor imagery BCI | mne-python/tutorials/machine-learning/ | 1 day |
| **E64-T1** | Suite2p ROI detection | MouseLand/suite2p/detection/ | 1 day |
| **E64-T2** | CaImAn motion correction | CaImAn/demos/general/ | 1 day |

---

## appendix E: verified API-accessible repos (clone failures recovered)

these repos failed to clone due to network issues but were fully explored via GitHub API:

| repo | env | key files verified | data size |
|------|-----|-------------------|-----------|
| patrickmineault/fmn-tutorial | E54 | `src/ndt_transformer.py`, `tutorials/Tutorial_Student.ipynb` | checkpoints/ |
| marius10p/NeuralDataScienceCSHL2019 | E55 | `ByronYuExercises/exercises.pdf`, `AllenObservatory/*.ipynb` | data_for_exercises.mat |
| melizalab/comp-neurosci | E56 | `4-Receptive-Fields.ipynb`, `data/starling/`, `data/zf/` | filters.npz |
| conorhoughton/COMS30127 | E62 | `coursework/spike_trains/rho.dat`, `stim.dat`, `load.py` | 9.2MB total |

---

*document end | 2026-01-30 | enhanced via GitHub API exploration*
