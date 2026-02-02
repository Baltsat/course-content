# pm_env Neural ML Environment Catalog — FINAL

> **30 environments | 160+ task variants | TPU/CPU only**  
> consolidated from 62 candidates | customer-priority aligned  
> created: 2026-01-30

---

## how to read this document

```
ENV-XX: [NAME] ★★★★★
├── what it is         → 1-2 sentence plain english
├── student does       → concrete actions the agent takes
├── we measure         → score computation (continuous [0,1])
├── hard because       → documented failure modes
├── can't game by      → reward hacking prevention
├── needs              → data, compute, libraries
└── tasks              → specific variants with metrics
```

**priority**: ★★★★★ = customer explicitly requested | ★★★★☆ = high alignment | ★★★☆☆ = useful

---

## TIER 1: CUSTOMER PRIORITY (build weeks 1-4)

---

### ENV-01: multi-format neural data wrangling (AJILE12) ★★★★★

**what it is**: 55 days of continuous human ECoG brain recordings + body pose tracking. 5+ file formats that must be merged correctly.

**student does**: load NWB neural data, parquet pose data, JSON/CSV annotations. align timestamps across modalities. handle electrode metadata drift. build streaming loader for 50GB+ dataset.

**we measure**:

- frame alignment accuracy (% correct)
- throughput (samples/sec)
- memory footprint
- artifact detection F1

**hard because**:

- electrode channel IDs change between days
- timestamps drift seconds over 55 days
- naive loading → OOM
- spectral extraction violates Nyquist if done wrong

**can't game by**:

- alignment checked vs hidden reference
- must process ≥10 days (no cherry-picking)
- throughput on standardized hardware

**needs**: DANDI archive, ~50GB/subject, CPU, streaming, pynwb

| task | student does | metric |
|------|--------------|--------|
| T1 | merge ECoG + pose across sessions | alignment % |
| T2 | extract high-gamma (70-150Hz) | SNR |
| T3 | handle electrode re-referencing | cross-day consistency |
| T4 | streaming loader for 55 days | throughput |
| T5 | artifact detection | F1 |
| T6 | segment into movement events | IoU |

Behavioral and Neural Variability of Naturalistic Arm Movements. eNeuro, 2021. <https://doi.org/10.1523/ENEURO.0007-21.2021>

Mining naturalistic human behaviors in long-term video and neural recordings. Journal of Neuroscience Methods, 2021. <https://doi.org/10.1016/j.jneumeth.2021.109199>

---

### ENV-02: cross-lab neuropixels integration (IBL) ★★★★★

**what it is**: mouse decision-making recorded across 12 labs worldwide. 42 brain regions, cross-lab variability in preprocessing.

**student does**: query ONE API, align spikes to behavior, decode choices from populations, handle cross-lab differences.

**we measure**:

- decoder AUC on held-out lab
- explained variance per region
- API efficiency

**hard because**:

- ONE API rate limits
- unit quality varies by lab
- atlas coordinate confusion (CCF vs IBL)
- temporal alignment jitter

**can't game by**:

- test lab not seen during training
- ≥5 brain regions required
- can't just use behavioral history

**needs**: IBL ONE API (free registration), ~500GB streamable, CPU

| task | student does | metric |
|------|--------------|--------|
| T1 | load sessions with VISp units | API efficiency |
| T2 | align spikes to stimulus | correlation |
| T3 | compare encoding across regions | R² per region |
| T4 | decode choice from population | cross-lab AUC |
| T5 | handle preprocessing variance | generalization |

---

### ENV-03: DNN → brain encoding models (algonauts) ★★★★★

**what it is**: predict fMRI voxel responses from DNN features. classic neuroscience benchmark.

**student does**: extract features from pretrained nets, fit ridge encoding, compare architectures (ResNet/CLIP/ViT), optimize hyperparameters.

**we measure**:

- R² on held-out stimuli (noise-ceiling corrected)
- layer-wise ranking
- cross-ROI performance

**hard because**:

- small fMRI dataset → overfitting
- hemodynamic delay modeling
- feature dimensionality explosion
- repeated stimuli → leakage risk

**can't game by**:

- test stimuli never seen during tuning
- noise ceiling computed
- ≥3 ROIs required (V1, V4, IT)

**needs**: algonauts project, ~10GB, CPU + pretrained inference

| task | student does | metric |
|------|--------------|--------|
| T1 | extract DNN features | variance |
| T2 | fit ridge encoding | held-out R² |
| T3 | compare layers per ROI | ranking |
| T4 | temporal integration (video) | temporal R² |
| T5 | noise ceiling estimation | corrected R² |
| T6 | multi-ROI joint model | total variance |

---

### ENV-04: data curation pipeline ★★★★★

**what it is**: filter large text corpus to improve downstream model. student decides what to keep/remove/rephrase.

**student does**: implement perplexity filtering, deduplication (MinHash), domain classification, data mixing. use Qwen-0.6B for quality scoring.

**we measure**:

- downstream eval loss improvement after continual pretraining
- filtering throughput
- dedup rate

**hard because**:

- over-filtering kills diversity
- dedup too aggressive on valid paraphrases
- domain classifiers overfit surface features
- eval→train leakage easy

**can't game by**:

- minimum corpus size (can't filter to tiny set)
- eval set completely held out
- downstream training is seeded/deterministic

**needs**: CommonCrawl/Dolma subset, 100MB-10GB, CPU, Qwen-0.6B

| task | student does | metric |
|------|--------------|--------|
| T1 | perplexity-based filtering | downstream Δ |
| T2 | efficient deduplication | rate + speed |
| T3 | domain classification | F1 |
| T4 | mixture ratio optimization | eval loss |
| T5 | LLM-based rephrasing | downstream Δ |
| T6 | filter for scientific text | citation proxy |

---

### ENV-05: ML paper implementation ★★★★★

**what it is**: read recent paper, understand method, implement in JAX, validate against reported results.

**student does**: port attention variants, implement flash attention, add KV cache optimization, speculative decoding, MoE routing.

**we measure**:

- test suite pass + numerical correctness
- speedup vs naive baseline
- memory reduction

**hard because**:

- paper notation ≠ code
- numerical precision (float32 vs bfloat16)
- broadcasting/shaping errors
- edge cases not in paper

**can't game by**:

- adversarial test inputs not in paper
- similarity check vs reference impl
- must pass correctness AND performance

**needs**: arxiv papers, reference tests, CPU/TPU, JAX+optax

**CONCRETE PAPERS** (customer requested specificity):

- Flash Attention 2: [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) → port to JAX/Pallas
- Speculative Decoding: [arXiv:2302.01318](https://arxiv.org/abs/2302.01318) → implement for Qwen
- GQA: [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) → add to MaxText
- RoPE: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) → implement variants
- LoRA: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) → JAX implementation

| task | student does | metric |
|------|--------------|--------|
| T1 | port attention to JAX | test pass |
| T2 | flash attention variant | speedup + correctness |
| T3 | KV cache optimization | inference speedup |
| T4 | speculative decoding | acceptance rate |
| T5 | MoE routing | load balance |
| T6 | gradient checkpointing | memory Δ |
| T7 | port to JAX (general) | correctness |

---

### ENV-06: behavior classification from pose (CalMS21) ★★★★★

**what it is**: classify mouse social behaviors (attack/mount/sniff) from pose keypoints. 63% "other" class.

**student does**: engineer features (velocity, distance, bout duration), handle class imbalance, build temporal models.

**we measure**:

- macro F1 (penalizes trivial classifier)
- per-class F1
- bout detection IoU

**hard because**:

- class imbalance → trivial "other" classifier gets 63%
- temporal leakage in naive splits
- features don't generalize across animal pairs

**can't game by**:

- macro F1 primary (not accuracy)
- test = different animal pairs
- temporal buffer in splits

**needs**: CalMS21, ~5GB, CPU, sklearn

| task | student does | metric |
|------|--------------|--------|
| T1 | velocity/acceleration features | predictive power |
| T2 | inter-animal distance | correlation |
| T3 | class imbalance handling | macro F1 |
| T4 | bout detection | IoU |
| T5 | Markov transition model | likelihood |

---

### ENV-07: inference optimization ★★★★★

**what it is**: make Qwen inference faster. student receives slow baseline, must optimize while preserving correctness.

**student does**: implement int8/fp8 quantization, KV cache compression, speculative decoding, continuous batching, pallas kernels.

**we measure**:

- tokens/second improvement
- memory reduction
- latency percentiles
- output divergence (must be < threshold)

**hard because**:

- quantization introduces errors
- KV cache compression loses info
- spec decode acceptance varies
- variable sequence lengths

**can't game by**:

- correctness on hidden test prompts
- divergence threshold enforced
- diverse input distribution

**needs**: Qwen 0.6B-4B, TPU v5e, JAX, AQT library

| task | student does | metric |
|------|--------------|--------|
| T1 | int8 quantization | speedup + divergence |
| T2 | KV cache compression | memory + quality |
| T3 | speculative decoding | acceptance + speedup |
| T4 | continuous batching | throughput |
| T5 | pallas kernel | TPU utilization |
| T6 | attention chunking | memory reduction |

---

### ENV-08: pipeline optimization (MERGED) ★★★★★

**what it is**: take slow naive ML pipeline, make it fast. data loading, preprocessing, batching, caching.

**student does**: add multiprocessing to loader, vectorize python loops, dynamic batching, embedding caching.

**we measure**:

- end-to-end speedup factor
- memory efficiency
- CPU utilization

**hard because**:

- multiprocessing overhead for small batches
- vectorization requires broadcast understanding
- cache invalidation is tricky

**can't game by**:

- must match naive output exactly
- standardized hardware
- minimum dataset size

**needs**: any neural dataset, CPU

| task | student does | metric |
|------|--------------|--------|
| T1 | streaming loader | throughput |
| T2 | prefetching | latency Δ |
| T3 | multiprocess loading | speedup |
| T4 | vectorize loops → numpy | speedup |
| T5 | numpy → JAX vmap | speedup |
| T6 | dynamic batching | padding waste Δ |
| T7 | embedding cache | hit rate |

---

### ENV-09: evaluate ML papers (NEW) ★★★★★

**what it is**: given ML paper, identify methodological flaws, statistical errors, or questionable claims. reading comprehension + critical analysis.

**student does**: read paper, identify issues (p-hacking, leakage, wrong baseline, missing ablation), explain why it matters, suggest fixes.

**we measure**:

- flaw detection recall (did they find the planted issues?)
- false positive rate
- explanation quality (rubric judge)

**hard because**:

- flaws are subtle (not obvious bugs)
- must distinguish real limitations from nitpicks
- need domain knowledge

**can't game by**:

- papers have known planted flaws + real flaws
- must explain reasoning (not just list)
- rubric judge checks quality

**needs**: modified arxiv papers, rubric judge, CPU

| task | student does | metric |
|------|--------------|--------|
| T1 | find statistical issues | recall |
| T2 | find methodology flaws | recall |
| T3 | find missing baselines | recall |
| T4 | explain significance | rubric score |
| T5 | suggest fixes | quality |

---

### ENV-10: knowledge distillation (NEW) ★★★★★

**what it is**: transfer knowledge from large teacher to small student model. part of data curation pipeline.

**student does**: generate synthetic data from teacher, filter for quality, train student, optimize distillation hyperparameters.

**we measure**:

- student performance on downstream eval
- data efficiency (samples to reach threshold)
- compute efficiency

**hard because**:

- teacher data quality varies
- student capacity limits
- distribution mismatch
- over-distillation → mode collapse

**can't game by**:

- eval on held-out tasks
- minimum diversity constraint
- can't just use teacher directly

**needs**: Qwen-4B teacher, Qwen-0.6B student, TPU, eval benchmark

| task | student does | metric |
|------|--------------|--------|
| T1 | generate distillation data | quality |
| T2 | filter synthetic data | downstream Δ |
| T3 | optimize temperature | student accuracy |
| T4 | multi-teacher ensemble | improvement |

---

## TIER 2: HIGH ALIGNMENT (build weeks 5-8)

---

### ENV-11: allen 2P calcium imaging ★★★★☆

**what it is**: 2-photon calcium imaging from mouse visual cortex. cell-type specific (excitatory vs inhibitory).

**student does**: extract ΔF/F traces, align to stimuli, classify cell types, detect prediction error signals.

**we measure**:

- cell type classification accuracy
- omission detection sensitivity
- cross-session correlation

**hard because**:

- neuropil contamination
- baseline estimation noise
- session variability

**can't game by**:

- test = held-out sessions
- transgenic labels (can't fake)
- multiple cell types required

**needs**: allen SDK, ~100GB streamable, CPU, 8GB RAM

| task | student does | metric |
|------|--------------|--------|
| T1 | extract ΔF/F | SNR |
| T2 | align to stimulus | latency |
| T3 | classify cell types | accuracy |
| T4 | detect omissions | sensitivity |
| T5 | model prediction error | variance |

---

### ENV-12: HMM latent state inference ★★★★☆

**what it is**: discover hidden neural states from population activity.

**student does**: fit HMM, select state count via BIC, align states to behavior, compare to switching linear dynamical system.

**we measure**:

- log-likelihood on held-out
- state-behavior correlation
- cross-session generalization

**hard because**:

- too many states → overfit
- EM local minima
- states may not be meaningful

**can't game by**:

- held-out time periods
- states must correlate with behavior
- minimum state duration

**needs**: steinmetz/IBL, CPU, hmmlearn or JAX

| task | student does | metric |
|------|--------------|--------|
| T1 | gaussian HMM | log-likelihood |
| T2 | state count selection | BIC |
| T3 | poisson HMM | fit |
| T4 | align to behavior | correlation |
| T5 | cross-session | transfer |

---

### ENV-13: neural decoding benchmark ★★★★☆

**what it is**: standardized decoder comparison across datasets (steinmetz, allen, IBL).

**student does**: implement decoders, tune regularization, optimize temporal kernels, test transfer.

**we measure**:

- AUC averaged across datasets
- cross-session transfer

**hard because**:

- temporal leakage
- trial structure matters
- high-D overfitting

**can't game by**:

- temporal separation
- cross-session test
- minimum trials

**needs**: multi-dataset standardized, CPU, sklearn/JAX

| task | student does | metric |
|------|--------------|--------|
| T1 | logistic regression | AUC |
| T2 | regularization tuning | CV accuracy |
| T3 | temporal kernel | time-resolved AUC |
| T4 | MLP decoder | calibration |
| T5 | cross-session transfer | accuracy |

---

### ENV-14: fMRI GLM + RSA ★★★★☆

**what it is**: fit GLM to fMRI, compute contrasts, representational similarity analysis.

**student does**: design matrix with HRF, first-level GLM, group stats, RDM computation.

**we measure**:

- contrast quality (effect size)
- RDM correlation
- multiple comparison correction

**hard because**:

- HRF modeling errors
- motion artifacts
- multiple comparisons

**can't game by**:

- effect sizes reasonable
- FDR mandatory
- replicates across subjects

**needs**: HCP task fMRI (**preprocessed 10-subject subset**, ~20GB, not full 1TB), nilearn

| task | student does | metric |
|------|--------------|--------|
| T1 | design matrix | correctness |
| T2 | first-level GLM | contrast |
| T3 | group analysis | effect size |
| T4 | RSA | RDM correlation |
| T5 | searchlight | significance |

---

### ENV-15: ECoG motor decoding (MERGED) ★★★★☆

**what it is**: decode motor variables from human ECoG. combines finger flex, joystick, imagery tasks.

**student does**: extract high-gamma, build continuous decoders, compare tasks, optimize for BCI.

**we measure**:

- R² for continuous
- accuracy for imagery
- information transfer rate

**hard because**:

- electrode coverage varies
- spectral features noisy
- continuous harder than classification

**can't game by**:

- held-out time
- multiple subjects
- latency constraints

**needs**: ECoG course materials, CPU

| task | student does | metric |
|------|--------------|--------|
| T1 | finger flexion | R² |
| T2 | joystick tracking | error |
| T3 | motor imagery | accuracy |
| T4 | BCI feature selection | ITR |
| T5 | cross-task transfer | accuracy |

---

### ENV-16: large-scale dimensionality reduction ★★★★☆

**what it is**: analyze 10k-80k neuron populations. memory-efficient algorithms.

**student does**: incremental PCA, compare methods, track learning changes.

**we measure**:

- variance explained at fixed dim
- throughput
- memory

**hard because**:

- naive covariance → OOM
- numerical instability
- learning vs drift confound

**can't game by**:

- must process full data
- memory budget enforced
- compared to full-batch

**needs**: stringer 80k neurons, ~20GB, CPU

| task | student does | metric |
|------|--------------|--------|
| T1 | efficient PCA 80k | variance |
| T2 | incremental PCA | throughput |
| T3 | compare PCA/ICA/NMF | ranking |
| T4 | decode visual | R² |

---

### ENV-17: interpretability / neuron attribution (NEW) ★★★★☆

**what it is**: understand what neural network neurons represent. attribution, probing, feature visualization.

**student does**: compute saliency maps, train probes, identify important neurons, compare attribution methods.

**we measure**:

- attribution faithfulness (ROAR/KAR)
- probe accuracy
- human interpretability rating

**hard because**:

- attribution methods disagree
- probes can memorize
- interpretability is subjective

**can't game by**:

- faithfulness via perturbation
- probes on held-out data
- human eval for sanity

**needs**: small trained model, CPU/TPU

| task | student does | metric |
|------|--------------|--------|
| T1 | saliency maps | faithfulness |
| T2 | linear probes | accuracy |
| T3 | compare attribution methods | agreement |
| T4 | identify important neurons | ablation effect |

---

### ENV-18: foundation model probing ★★★★☆

**what it is**: use pretrained neural foundation models (CEBRA, NDT) for downstream tasks.

**student does**: extract embeddings, fit linear probes, fine-tune, compare to from-scratch.

**we measure**:

- probe accuracy improvement
- fine-tune improvement
- embedding quality

**hard because**:

- foundation may not transfer
- fine-tune can overfit
- fair comparison tricky

**can't game by**:

- from-scratch baseline optimized
- test held out
- hyperparams tuned on val

**needs**: CEBRA/NDT pretrained, neural data, CPU/TPU

| task | student does | metric |
|------|--------------|--------|
| T1 | extract embeddings | quality |
| T2 | linear probe | accuracy |
| T3 | compare to scratch | delta |
| T4 | fine-tune | improvement |

---

### ENV-19: spike sorting QC ★★★★☆

**what it is**: assess quality of spike-sorted units. compute metrics, classify good/bad.

**student does**: ISI violations, amplitude cutoffs, waveform metrics, automated classifier.

**we measure**:

- agreement with expert labels
- metric accuracy

**hard because**:

- metrics noisy for low-firing
- threshold arbitrary
- experts disagree

**can't game by**:

- test from different recordings
- above-chance on multiple metrics

**needs**: IBL/steinmetz phy outputs, CPU

| task | student does | metric |
|------|--------------|--------|
| T1 | ISI violation | accuracy |
| T2 | amplitude cutoff | threshold |
| T3 | waveform metrics | correlation |
| T4 | auto classification | agreement |

---

### ENV-20: broken baseline debugging (R² = -1.17) ★★★★☆

**what it is**: student inherits code with negative R². must diagnose and fix.

**student does**: identify bug location, explain what's wrong, fix to positive R².

**we measure**:

- final R²
- diagnosis accuracy
- fix minimality

**hard because**:

- bug could be anywhere (data/model/eval)
- multiple bugs may interact
- easy to introduce new bugs

**can't game by**:

- must explain bug (not just rewrite)
- fix must be minimal
- R² on hidden test

**needs**: HCP fMRI (broken code), CPU

| task | student does | metric |
|------|--------------|--------|
| T1 | diagnose | accuracy |
| T2 | fix preprocessing | R² |
| T3 | fix model | final R² |

---

### ENV-21: DNN-brain alignment comparison ★★★★☆

**what it is**: compare how DNNs predict brain responses. CNNs vs ViTs vs CLIP.

**student does**: extract features, compute alignment, layer-wise analysis.

**we measure**:

- brain alignment score
- best layer per ROI

**hard because**:

- feature dims differ
- alignment method matters
- layer selection combinatorial

**can't game by**:

- ≥3 architectures required
- standardized extraction
- test stimuli held out

**needs**: algonauts/kay, CPU, pretrained models

| task | student does | metric |
|------|--------------|--------|
| T1 | extract CNN/ViT/CLIP | validity |
| T2 | compare alignment | score |
| T3 | layer-wise analysis | best layer |
| T4 | feature combination | R² |

---

### ENV-22: multi-modal fusion (fMRI + MEG) ★★★★☆

**what it is**: combine spatial (fMRI) + temporal (MEG) for representational analysis.

**student does**: temporal RDMs from MEG, spatial from fMRI, fuse, compare geometry.

**we measure**:

- fusion improvement
- RSA correlation

**hard because**:

- different noise characteristics
- alignment tricky
- fusion can overfit

**can't game by**:

- test stimuli held out
- fusion must beat both modalities

**needs**: cichy 2014 data, CPU

| task | student does | metric |
|------|--------------|--------|
| T1 | temporal RDM (MEG) | quality |
| T2 | spatial RDM (fMRI) | quality |
| T3 | fuse | improvement |
| T4 | geometry comparison | RSA |

---

## TIER 3: SPECIALIZED (build weeks 9-12)

---

### ENV-23: dynamical systems / attractors ★★★☆☆

**what it is**: implement attractor networks for working memory. **VALIDATED AGAINST REAL IBL HEAD DIRECTION DATA.**

**student does**: implement bump/ring attractors in JAX, tune stability, add noise, compare to empirical HD cells.

**we measure**:

- correlation with IBL head direction recordings
- stability duration
- capacity

**hard because**:

- parameter tuning sensitive
- noise destabilizes
- capacity vs stability tradeoff

**can't game by**:

- must validate vs real neural data (not just simulations)
- stability multi-timescale
- held-out patterns

**needs**: simulated + IBL HD data, JAX, CPU/TPU

| task | student does | metric |
|------|--------------|--------|
| T1 | bump attractor | stability |
| T2 | ring attractor | heading accuracy |
| T3 | compare to empirical HD | correlation |
| T4 | capacity | bits |

---

### ENV-24: RNN motor control ★★★☆☆

**what it is**: train RNNs for reaching. **COMPARED TO REAL MOTOR CORTEX DATA.**

**student does**: implement RNN in JAX, regularize for low-d dynamics, analyze fixed points, compare to monkey M1.

**we measure**:

- trajectory MSE + neural similarity
- dimensionality
- fixed point count

**hard because**:

- training unstable
- low-d regularization hurts perf
- numerical fixed point analysis

**can't game by**:

- must achieve trajectory AND neural similarity
- dynamics compared to real data

**needs**: simulated + churchland M1, JAX, TPU

| task | student does | metric |
|------|--------------|--------|
| T1 | vanilla RNN | MSE |
| T2 | regularize | dimensionality |
| T3 | fixed points | count |
| T4 | compare to M1 | similarity |

---

### ENV-25: circular statistics ★★★☆☆

**what it is**: handle angular data correctly. common bug: linear mean on circular data.

**student does**: circular mean/variance, von Mises fitting, circular-linear regression.

**we measure**:

- numerical correctness
- KS test

**hard because**:

- linear mean wrong for angles
- wrap-around at 0/360
- polar vs cartesian

**can't game by**:

- test includes boundary cases
- all quadrants
- precision checked

**needs**: laquitaine or simulated, CPU

| task | student does | metric |
|------|--------------|--------|
| T1 | circular stats | accuracy |
| T2 | von Mises | KS |
| T3 | regression | R² |

---

### ENV-26: JAX parallelism (vmap/pmap/pallas) ★★★☆☆

**what it is**: convert sequential to parallel JAX.

**student does**: identify parallelizable ops, vmap, pmap, pallas kernels.

**we measure**:

- speedup factor
- scaling efficiency
- TPU utilization

**hard because**:

- broadcast understanding
- device placement
- kernel debugging

**can't game by**:

- must match sequential exactly
- standardized hardware
- multiple device counts

**needs**: JAX, TPU v5e

| task | student does | metric |
|------|--------------|--------|
| T1 | sequential → vmap | speedup |
| T2 | pmap multi-device | scaling |
| T3 | vmap + pmap | throughput |
| T4 | pallas kernel | utilization |

---

### ENV-27: normalizing flows for ICA ★★★☆☆

**what it is**: use normalizing flows to separate mixed sources. **documented 1/3 success rate.**

**student does**: implement NICE/LICE, apply to ICA, handle failure mode.

**we measure**:

- source recovery % (across restarts)
- reconstruction MSE

**hard because**:

- ICA works 1/3 of time
- flows must be invertible
- local minima

**can't game by**:

- success rate over many restarts
- must beat random (>50%)
- sources verified by correlation

**needs**: synthetic ICA, CPU

| task | student does | metric |
|------|--------------|--------|
| T1 | NICE flow | MSE |
| T2 | LU flow | invertibility |
| T3 | ICA demixing | recovery % |

---

### ENV-28: bayesian behavior modeling ★★★☆☆

**what it is**: fit bayesian models to human behavioral data.

**student does**: ideal observer, psychometric fits, model comparison.

**we measure**:

- model comparison (AIC/BIC)
- parameter recovery

**hard because**:

- convergence issues
- temporal structure ignored
- session biases

**can't game by**:

- parameter recovery on simulated
- cross-validation

**needs**: laquitaine data, scipy

| task | student does | metric |
|------|--------------|--------|
| T1 | psychometric fit | deviance |
| T2 | bayesian observer | likelihood |
| T3 | model comparison | AIC/BIC |

---

### ENV-29: birdsong neural encoding ★★★☆☆

**what it is**: analyze neural responses to birdsong. STA, LNP, information theory.

**student does**: estimate receptive fields, fit LNP, compute info rate.

**we measure**:

- prediction R²
- bits/spike

**hard because**:

- high-D stimulus
- STA convergence slow
- info estimation biased

**can't game by**:

- test stimuli held out
- info bias-corrected

**needs**: starling/finch data, CPU

| task | student does | metric |
|------|--------------|--------|
| T1 | STA receptive field | R² |
| T2 | LNP model | R² |
| T3 | information rate | bits/spike |

---

### ENV-30: hopfield networks ★★★☆☆

**what it is**: implement associative memory. hebbian vs storkey rules.

**student does**: implement hopfield, compare rules, optimize noise tolerance, fix storkey bug.

**we measure**:

- recall at 30% noise
- capacity

**hard because**:

- storkey has common bug
- pseudo-inverse numerical
- spurious attractors

**can't game by**:

- test patterns not stored
- noise standardized
- >90% clean recall required

**needs**: MNIST/alphabet, CPU

| task | student does | metric |
|------|--------------|--------|
| T1 | hebbian | recall |
| T2 | storkey (fix bug) | benchmark |
| T3 | pseudo-inverse | capacity |
| T4 | noise tolerance | accuracy |

---

## APPENDICES

---

### appendix A: customer quote → environment

| quote | environments |
|-------|-------------|
| "messy input data science – multiple formats" | ENV-01, 02, 06, 11 |
| "hill-climbing on core metrics" | ENV-03, 04, 07, 08, 13 |
| "implement ML papers" | ENV-05, 18 |
| "evaluate ML papers" | **ENV-09** (NEW) |
| "data curation" | ENV-04, 10, 19 |
| "inference optimization" | ENV-07 |
| "training optimization" | ENV-08, 10 |
| "neuron attribution/interp" | **ENV-17** (NEW) |
| "fix the bug" | ENV-20 |
| "train a speculator" | ENV-07-T3 |
| "vmap/pmap" | ENV-26 |

---

### appendix B: infrastructure

| requirement | environments |
|-------------|-------------|
| CPU-only | all except TPU-required |
| TPU v5e | ENV-05, 07, 10, 24, 26 |
| >8GB RAM | ENV-16 |
| ONE API | ENV-02 |
| Allen SDK | ENV-11 |
| nilearn | ENV-14 |
| JAX+optax | ENV-05, 07, 10, 23, 24, 26 |

---

### appendix C: dataset access

| dataset | method | auth | size |
|---------|--------|------|------|
| AJILE12 | DANDI | no | 50GB/subj |
| IBL | ONE API | yes (free) | 500GB stream |
| Allen 2P | SDK | no | 100GB stream |
| Steinmetz | figshare | no | 10GB |
| Stringer 80k | figshare | no | 20GB |
| CalMS21 | figshare | no | 5GB |
| Algonauts | project | no | 10GB |
| HCP | ConnectomeDB | yes | **20GB subset** |
| ECoG | course | no | ~1GB |

---

### appendix D: implementation priority

**phase 1 (weeks 1-4)**: tier 1 environments

- wk1: ENV-01 (AJILE12), ENV-06 (CalMS21)
- wk2: ENV-02 (IBL), ENV-04 (curation)
- wk3: ENV-03 (encoding), ENV-05 (paper impl)
- wk4: ENV-07 (inference), ENV-08 (pipeline), ENV-09 (eval papers), ENV-10 (distillation)

**phase 2 (weeks 5-8)**: tier 2 environments

- ENV-11 through ENV-22

**phase 3 (weeks 9-12)**: tier 3 environments

- ENV-23 through ENV-30

---

### appendix E: failure mode taxonomy

| mode | environments |
|------|-------------|
| data format errors | ENV-01, 02 |
| class imbalance | ENV-06, 11 |
| temporal leakage | ENV-06, 12, 13 |
| OOM | ENV-01, 16 |
| numerical precision | ENV-25, 27 |
| cross-session drift | ENV-01, 11, 13 |
| overfitting | ENV-03, 12, 20 |
| EM local minima | ENV-12, 27 |
| broken inherited code | ENV-20, 30 |

---

### appendix F: what makes a GOOD pm_env task (summary)

per customer + pm_env docs:

1. **no memorizable pattern** – reasoning required, not recall
2. **multiple failure points** – different students fail differently  
3. **continuous reward** – not binary pass/fail where possible
4. **open-ended improvement** – can always do better (like ELO)
5. **real data** – avoid purely synthetic
6. **can't game** – reward hacking prevention built-in
7. **diverse solutions** – multiple valid approaches exist
8. **validates against reality** – theory envs need real data comparison

---

*FINAL DOCUMENT | 2026-01-30 | 30 environments | 160+ tasks*
