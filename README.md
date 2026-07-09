# Higgs Boson Detection with XGBoost

** Language / Idioma:** **[English](#english)** · [Español](#español)

---

<a name="english"></a>

# 🇬🇧 English

Solution for Kaggle's **Higgs Boson Machine Learning Challenge** (2014). The goal is to separate
**signal** events (a collision that produced a Higgs boson decaying into two tau leptons,
`H → ττ`) from **background** events that leave similar signatures in the detector but are not
Higgs.

This model reaches an **AMS ≈ 3.7** on the leaderboard, which would place it around the
**top 100** of the original competition (the winner reached 3.806). This README explains
**every part of the code, what each variable means, and why each decision was made**, so it can
serve as learning material.

## Table of contents

1. [The problem in one sentence](#en-1-the-problem-in-one-sentence)
2. [The physics, without equations](#en-2-the-physics-without-equations)
3. [The data: what each column means](#en-3-the-data-what-each-column-means)
4. [The AMS metric: why not accuracy](#en-4-the-ams-metric-why-not-accuracy)
5. [Solution architecture](#en-5-solution-architecture)
6. [Features: why there is no custom feature engineering](#en-6-featurespy--feature-engineering)
7. [`train_model.py` — training step by step](#en-7-train_modelpy--training-step-by-step)
8. [`generate_submission.py` — inference and submission](#en-8-generate_submissionpy--inference-and-submission)
9. [Why each design decision](#en-9-why-each-design-decision)
10. [How to run it](#en-10-how-to-run-it)
11. [Ideas to improve and learn more](#en-11-ideas-to-improve-and-learn-more)

---

<a name="en-1-the-problem-in-one-sentence"></a>

## 1. The problem in one sentence

We are given **250,000 simulated collisions** for which we know the answer (signal `s` or
background `b`) and **550,000 collisions** for which we do not. We must train a model on the
former and decide, for each of the remaining 550,000, whether it is signal or background. We do
not optimize classic accuracy, but a physics metric called **AMS** that measures "discovery
significance".

At its core, this is a **weighted, imbalanced binary classification problem**.

<a name="en-2-the-physics-without-equations"></a>

## 2. The physics, without equations

When two protons collide at the LHC (Large Hadron Collider), particles are created that decay
almost instantly. The detector does not see the Higgs directly: it sees the "debris" (momentum,
energy, angles of the daughter particles). The challenge is to recognize the **pattern** of
debris left by an `H → ττ` versus other processes that look similar.

Concepts that appear in the variables and are worth keeping in mind:

- **`pt` (transverse momentum):** the particle's momentum in the plane perpendicular to the beam.
  It is the most informative quantity because the momentum along the beam is poorly known.
- **`eta` (η, pseudorapidity):** a way to measure the angle relative to the beam axis. η≈0 is
  perpendicular to the beam; large η is almost parallel.
- **`phi` (φ, azimuthal angle):** the angle around the beam (0 to 2π). It is a circular angle:
  0 and 2π are the same place.
- **`MET` (Missing Transverse Energy):** the energy that is "missing". Since neutrinos escape
  without a trace, their presence is inferred from the momentum imbalance. Taus produce
  neutrinos, so MET is a strong signal hint.
- **`jet`:** a spray of particles produced by a quark or gluon. An event can have 0, 1, 2 or 3+
  jets.

<a name="en-3-the-data-what-each-column-means"></a>

## 3. The data: what each column means

Each row is **one event** (one collision). The training file has 33 columns: `EventId` +
**30 physics variables** + `Weight` + `Label`. The test file has neither `Weight` nor `Label`.

### Special columns

| Column | Meaning | Use |
|---|---|---|
| `EventId` | Unique event identifier | Submission only; **not** a feature |
| `Weight` | Event importance weight | For training and AMS; **never** a feature |
| `Label` | `s` (signal) or `b` (background) | The target variable (train only) |

> **`Weight` is not used as an input feature.** It is a statistical weight indicating "how many
> real events this row represents". The simulation generated far more signals than actually occur;
> the weights correct that imbalance so the physics counts add up. Using it as a feature would let
> the model cheat (signal and background weights are systematically different).

### `PRI_` variables (primitives): what the detector measures

Raw, direct measurements.

| Variable | Meaning |
|---|---|
| `PRI_tau_pt` | Transverse momentum of the hadronic tau |
| `PRI_tau_eta` | Pseudorapidity (η) of the tau |
| `PRI_tau_phi` | Azimuthal angle (φ) of the tau |
| `PRI_lep_pt` | Transverse momentum of the lepton (electron or muon) |
| `PRI_lep_eta` | Pseudorapidity of the lepton |
| `PRI_lep_phi` | Azimuthal angle of the lepton |
| `PRI_met` | Magnitude of the missing transverse energy (MET) |
| `PRI_met_phi` | Azimuthal angle of the MET |
| `PRI_met_sumet` | Total transverse energy deposited in the detector |
| `PRI_jet_num` | Number of jets (0, 1, 2, 3) |
| `PRI_jet_leading_pt` | `pt` of the most energetic jet |
| `PRI_jet_leading_eta` | η of the leading jet |
| `PRI_jet_leading_phi` | φ of the leading jet |
| `PRI_jet_subleading_pt` | `pt` of the second jet |
| `PRI_jet_subleading_eta` | η of the second jet |
| `PRI_jet_subleading_phi` | φ of the second jet |
| `PRI_jet_all_pt` | Scalar sum of the `pt` of all jets |

### `DER_` variables (derived): what physicists compute

Physicists already combined the primitives into meaningful quantities. These are "expert-made
features" and tend to be the most powerful.

| Variable | Meaning |
|---|---|
| `DER_mass_MMC` | Estimated Higgs mass via the *Missing Mass Calculator*. **The most discriminating one.** |
| `DER_mass_transverse_met_lep` | Transverse mass between MET and lepton |
| `DER_mass_vis` | Invariant mass of the visible system (tau + lepton) |
| `DER_pt_h` | Transverse momentum of the Higgs system (vector sum tau + lep + MET) |
| `DER_deltaeta_jet_jet` | \|Δη\| between the two jets |
| `DER_mass_jet_jet` | Invariant mass of the two jets |
| `DER_prodeta_jet_jet` | Product of the ηs of the two jets |
| `DER_deltar_tau_lep` | Angular separation ΔR between tau and lepton |
| `DER_pt_tot` | Modulus of the vector sum of all `pt` |
| `DER_sum_pt` | Scalar sum of the `pt` of tau, lepton and jets |
| `DER_pt_ratio_lep_tau` | Ratio `pt_lep / pt_tau` |
| `DER_met_phi_centrality` | Centrality of the MET relative to tau and lepton |
| `DER_lep_eta_centrality` | Centrality of the lepton η relative to the two jets |

### The `-999.0` value: missing data with physical meaning

Many variables do not always exist. If an event has **0 jets**, there is no
`PRI_jet_leading_pt`; if it has **fewer than 2 jets**, there is no `DER_mass_jet_jet`. In those
cases the value is `-999.0`.

**It is not noise: it is "not applicable".** That is why we explicitly tell XGBoost that `-999.0`
means "missing" (`missing=-999.0`), so it learns to treat it as its own category and not as the
negative number −999.

<a name="en-4-the-ams-metric-why-not-accuracy"></a>

## 4. The AMS metric: why not accuracy

The competition is scored with the **AMS** (*Approximate Median Significance*). It measures, in
units of standard deviations (σ), how significant the "discovery" made by your event selection
would be. In particle physics, 5σ is considered a discovery.

```
AMS = sqrt( 2 · ( (s + b + b_reg) · ln(1 + s/(b + b_reg)) − s ) )
```

Where, **over the events your model classifies as signal**:

- `s` = sum of **weights** of the events that **were** signal (weighted true positives).
- `b` = sum of **weights** of the events that **were** background (weighted false positives).
- `b_reg` = 10 (a fixed regularization constant that stabilizes AMS when `b` is small).

Implementation in `train_model.py`:

```python
def calculate_ams(true_labels, predictions, weights, threshold):
    y_pred = (predictions > threshold).astype(int)          # do we call it signal?
    s = np.sum(weights[(true_labels == 1) & (y_pred == 1)]) # signal caught (weighted)
    b = np.sum(weights[(true_labels == 0) & (y_pred == 1)]) # background leaked (weighted)
    br = 10.0
    radicand = 2 * ((s + b + br) * math.log(1.0 + s / (b + br)) - s)
    if radicand < 0:
        return 0.0
    return math.sqrt(radicand)
```

**Key consequences** (and why the model is designed the way it is):

1. **It counts weights, not events.** That is why `Weight` is essential.
2. **It rewards purity over volume.** Leaking background (`b`) penalizes a lot. It pays to be
   conservative and keep only the clearest signal events → **the optimal threshold is high (~0.9),
   not 0.5**.
3. **It is a *ranking* + cut metric.** All that matters is which events end up above the
   threshold. This justifies using AUC as the stopping metric and ranking by probability at the
   end.

<a name="en-5-solution-architecture"></a>

## 5. Solution architecture

```
                    data/training.csv (250k, labeled, 30 features)
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  train_model.py                              │
        │  StratifiedKFold 5 folds  ×  seed bagging    │
        │  → 5 XGBoost models + 5 optimal thresholds    │
        └─────────────────────────────────────────────┘
                              │
              higgs_model_fold_1..5.json  +  mean threshold
                              │
                              ▼
                    data/test.csv (550k, unlabeled, 30 features)
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  generate_submission.py                       │
        │  average of the 5 probabilities (ensemble)    │
        │  → ranking → threshold cut → s/b              │
        └─────────────────────────────────────────────┘
                              │
                              ▼
                   submission_ensembled.csv
```

Two central ideas:

- **K-Fold ensembling:** we train 5 models, each on a different 80% of the data. Averaging their
  predictions reduces variance and adds robustness.
- **Seed bagging:** each model uses a different random seed, which increases the ensemble's
  diversity (row and column subsampling differs in each one).

<a name="en-6-featurespy--feature-engineering"></a>

## 6. Features: why there is no custom feature engineering

The model is trained on the **30 official features as-is** (13 `DER_` + 17 `PRI_`). There is no
separate feature-engineering step, and that is a deliberate, *measured* decision — not laziness.

**The experiment.** An earlier version added an engineered feature, `Delta_R_tau_lep`: the
angular separation `ΔR = √(Δη² + Δφ²)` between the tau and the lepton (a distance in the (η, φ)
plane, where signal and background have different geometries). It looked reasonable, but the
dataset **already ships `DER_deltar_tau_lep`** — the exact same quantity computed by physicists.
So the "new" feature was really a duplicate.

We verified it instead of guessing. A controlled 5-fold cross-validation (identical splits, seeds
and params) measured the CV AMS with and without it:

| Configuration | Mean CV AMS |
|---|---|
| **With** `Delta_R_tau_lep` | 5.5538 |
| **Without** `Delta_R_tau_lep` | 5.5426 |
| Δ | −0.0112 (within fold-to-fold noise) |

The feature added **nothing** (the tiny difference is noise), so it was removed. The model is
simpler and scores at least as well — **AMS 3.72** on Kaggle, the project's best (see the threshold
note in [section 11](#en-11-ideas-to-improve-and-learn-more)).

> **Lesson:** good feature engineering means adding information the model does not already have.
> Recomputing a quantity that is already a column — even a physically meaningful one — is wasted
> effort. And the way to know is to **measure with cross-validation**, not to trust intuition.
> Genuinely new features (mass combinations, φ-rotation invariances, etc.) are the real lever —
> see [section 11](#en-11-ideas-to-improve-and-learn-more).

<a name="en-7-train_modelpy--training-step-by-step"></a>

## 7. `train_model.py` — training step by step

### 7.1 Loading and preparation

```python
df = pd.read_csv("data/training.csv")
df["Label"] = df["Label"].map({'s': 1, 'b': 0})   # to binary 1/0

y = df["Label"].values          # target
weights = df["Weight"].values   # physics weights
X = df.drop(columns=['EventId', 'Weight', 'Label'])  # features only (the 30 official ones)
```

Note that `EventId`, `Weight` and `Label` are **removed from `X`**: they are not predictors.

### 7.2 Cross-validation + ensemble

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
seeds = [23, 5, 2024, 777, 8888]
```

- **`StratifiedKFold`:** splits the data into 5 chunks while preserving the signal/background
  ratio in each. Each iteration trains on 4 chunks (80%) and validates on 1 (20%).
- **`shuffle=True, random_state=23`:** shuffles before splitting, with a fixed seed for
  reproducibility.
- **`seeds`:** a different seed per fold → diversity for the ensemble.

### 7.3 XGBoost hyperparameters

```python
params = {
    'objective': 'binary:logistic',  # binary classification, output = probability
    'eval_metric': 'auc',            # stopping metric (ranking)
    'eta': 0.01,                     # learning rate (small, careful steps)
    'max_depth': 6,                  # depth of each tree (complexity control)
    'subsample': 0.8,                # use 80% of rows per tree (regularization)
    'colsample_bytree': 0.8,         # use 80% of columns per tree (regularization)
}
```

### 7.4 Inside each fold

```python
# 1) Class balancing specific to this fold
sum_wpos = np.sum(w_tr[y_tr == 1])
sum_wneg = np.sum(w_tr[y_tr == 0])
params['scale_pos_weight'] = sum_wneg / sum_wpos
params['seed'] = seeds[fold]

# 2) Rescale validation weights to the real test scale (550k)
w_va_scaled = w_va * (550000 / len(w_va))

# 3) DMatrix (XGBoost's optimized structure) with missing-value handling
dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, missing=-999.0)
dval   = xgb.DMatrix(X_va, label=y_va, weight=w_va, missing=-999.0)

# 4) Training with early stopping
model = xgb.train(
    params, dtrain,
    num_boost_round=5000,          # high ceiling of trees...
    evals=[(dval, 'validation')],
    early_stopping_rounds=50,      # ...but stop if AUC doesn't improve for 50 rounds
    verbose_eval=False
)
```

- **`scale_pos_weight = sum_wneg / sum_wpos`:** the (weighted) signal is extremely rare compared
  to background. This parameter multiplies the positive class's gradient so the model **does not
  ignore it**. It is recalculated **inside** the fold, using only training data, to avoid leaking
  validation information (*data leakage*).
- **`weight=w_tr`:** the physics weights enter the loss function. This way the model optimizes the
  **real** problem (expected yields) and not the artificial count from the simulation.
- **`w_va * (550000 / len(w_va))`:** a validation fold has ~50k events; the test has 550k. Since
  AMS depends on **absolute** weight sums (and `b_reg=10` is a fixed constant), the validation
  weights must be rescaled to the test scale so that the estimated AMS —and the optimal
  threshold— are comparable to those on the leaderboard.
- **`missing=-999.0`:** crucial. XGBoost learns a "default direction" at each split for missing
  values, instead of treating them as the number −999.
- **`eta=0.01` + `num_boost_round=5000` + `early_stopping_rounds=50`:** low learning rate = small,
  precise steps; many trees available; but early stopping cuts off when the validation AUC stops
  improving for 50 rounds and keeps the best iteration. This is the standard recipe to squeeze
  XGBoost without overfitting.

### 7.5 Optimal threshold search

```python
val_preds = model.predict(dval)
best_fold_ams = 0.0
best_fold_threshold = 0.0
for t in np.arange(0.80, 0.99, 0.01):
    ams = calculate_ams(y_va, val_preds, w_va_scaled, threshold=t)
    if ams > best_fold_ams:
        best_fold_ams = ams
        best_fold_threshold = t
```

The model returns a probability, but the cut that maximizes AMS is **not 0.5**. Because of
`scale_pos_weight` and the penalty on false positives, the optimum is **high (~0.9)**. That is why
only the range `[0.80, 0.99)` is swept and the `t` that maximizes AMS on that fold is chosen.

Finally, each fold's model is saved (`higgs_model_fold_{i}.json`) and the 5 optimal thresholds are
averaged:

```python
final_threshold = np.mean(fold_thresholds)   # "Robust Mean Threshold"
```

Averaging the 5 thresholds is more robust than trusting a single fold's optimum (which may be
overfit to its particular split).

<a name="en-8-generate_submissionpy--inference-and-submission"></a>

## 8. `generate_submission.py` — inference and submission

```python
df_test = pd.read_csv('data/test.csv')
X_test = df_test.drop(columns=['EventId'])
dtest = xgb.DMatrix(X_test, missing=-999.0)

# Ensemble: average of the 5 probabilities
ensemble_probabilities = np.zeros(len(df_test))
for i in range(1, 6):
    model = xgb.Booster()
    model.load_model(f'higgs_model_fold_{i}.json')
    ensemble_probabilities += model.predict(dtest)
final_probabilities = ensemble_probabilities / 5
```

> **Golden rule:** the test features must be **identical** to the training ones (same columns,
> same order). Here both simply use the 30 official columns straight from the CSV, so it matches.
> If training used columns you don't reproduce at test time (e.g. engineered or PCA features),
> XGBoost would fail with a *feature mismatch* or produce invalid predictions.

### Kaggle's submission format

The submission needs 3 columns: `EventId`, `RankOrder` and `Class`.

```python
submission = submission.sort_values(by='Prob', ascending=True)
submission['RankOrder'] = range(1, len(submission) + 1)   # 1 = least signal, 550000 = most signal

PROBABILITY_THRESHOLD = 0.940     # robust threshold from training (out-of-fold)
submission['Class'] = np.where(submission['Prob'] > PROBABILITY_THRESHOLD, 's', 'b')
```

- **`RankOrder`:** orders the 550k events from least to most "signal-like". We sort by ascending
  probability and number them 1…550000. Kaggle requires it to evaluate AMS at different cuts.
- **`Class`:** `'s'` if the probability exceeds the threshold, `'b'` otherwise. The `0.940` is the
  threshold chosen on the pooled out-of-fold predictions (see [section 11](#en-11-ideas-to-improve-and-learn-more));
  it scored **AMS 3.72** on Kaggle, the project's best.

The result is sorted by `EventId` and saved to `submission_ensembled.csv`.

<a name="en-9-why-each-design-decision"></a>

## 9. Why each design decision

| Decision | Reason |
|---|---|
| **XGBoost (gradient boosting over trees)** | Dominant on tabular data; handles missing values natively; captures nonlinear interactions between physics variables. |
| **Use `Weight` in the loss** | AMS is computed with weights; training without them would optimize the wrong problem. |
| **`scale_pos_weight` per fold** | The weighted signal is a minority; without this the model would ignore it. Recomputing per fold avoids data leakage. |
| **`missing=-999.0`** | `-999` means "not applicable" (e.g. no jets); treating it as missing and not as a number is physically correct. |
| **5-Fold + seed bagging** | Reduces variance and overfitting; the ensemble generalizes better than a single model. |
| **`eval_metric='auc'`** | The real objective (AMS) is ranking-based; AUC is a good derivable proxy for early stopping. |
| **Low `eta` + early stopping** | Careful convergence without overfitting; validation decides the number of trees, not you by hand. |
| **Threshold ~0.9 and not 0.5** | AMS penalizes false positives; it pays to be pure and conservative. |
| **Rescaling weights to 550k** | AMS depends on absolute weight sums; the test scale must be matched to estimate threshold and AMS well. |
| **Averaging thresholds** | More robust than a single fold's optimum. |

<a name="en-10-how-to-run-it"></a>

## 10. How to run it

Requirements (see `requirements.txt`): Python 3.10+, `pandas`, `numpy`, `xgboost>=2.0`,
`scikit-learn`.

```bash
# 1) (optional) create an environment and install dependencies
pip install -r requirements.txt

# 2) Train: generates higgs_model_fold_1..5.json and prints the "Robust Mean Threshold"
python train_model.py

# 3) (if needed) copy that threshold into generate_submission.py -> PROBABILITY_THRESHOLD

# 4) Generate the submission CSV
python generate_submission.py
```

Expected data layout:

```
data/
├── training.csv   # 250,000 labeled events (with Weight and Label)
└── test.csv       # 550,000 events to predict
```

Output: `submission_ensembled.csv`, ready to upload to Kaggle.

<a name="en-11-ideas-to-improve-and-learn-more"></a>

## 11. Ideas to improve and learn more

### Already tried in this project (and the honest result)

- **Removing the redundant `Delta_R_tau_lep`** (done): CV-neutral (−0.011), so we kept the simpler
  model. See [section 6](#en-6-featurespy--feature-engineering).
- **Isotonic calibration of the probabilities**: AMS **dropped** (3.714 → 3.682). See the lesson
  below.
- **Per-jet-number models** (`PRI_jet_num` split into {0, 1, ≥2}) **+ per-group OOF thresholds** —
  the textbook "winners' trick". A controlled CV said a tie (5.4925 vs 5.4901), and Kaggle
  confirmed it: **3.716 vs 3.714 — +0.002, within leaderboard noise**. Not adopted: 3× the models
  and complexity for no real gain. Why so little? XGBoost's native missing-value handling already
  learns the jet-conditional structure, and splitting shrinks the data available to each model.
  (The winners who gained from this used linear models / neural nets, which handle structural
  missingness far worse.)
- **Single threshold on out-of-fold (OOF) predictions** instead of averaging per-fold thresholds
  (**adopted**): a more principled, more stable operating point. In CV it looked like a tie, but on
  the real leaderboard it clearly won — the OOF threshold `0.940` scored **3.72**, while the noisy
  per-fold average `0.934` scored only 3.70. This is the current `PROBABILITY_THRESHOLD` and the
  project's best result. It also shows how sensitive AMS is to the operating point: a 0.006 shift
  in the cut moved the score by 0.02, with no change to the model at all.

### Still worth trying

1. **Genuinely new physics features** — φ-rotation (fix `PRI_tau_phi = 0` and rotate the rest to
   remove an irrelevant rotational symmetry), log-transform of long-tailed variables (`pt`,
   masses), mass combinations. This is what actually separated the top solutions.
2. **Hyperparameter tuning** (`max_depth`, `eta`, `min_child_weight`, `gamma`, `lambda`/`alpha`)
   with Optuna, always validating with AMS.
3. **A diverse stacked ensemble** — combine genuinely different base learners (LightGBM, a neural
   net) with a simple meta-model. Stacking only helps when the base models are *diverse*; stacking
   five near-identical XGBoost models gains nothing.

### Lessons learned from this project

- **Match the technique to the metric.** AMS-by-threshold is a **ranking** metric — only the order
  of events and where you cut matter. That is why **isotonic calibration hurt**: it fixes
  probability *magnitude*, which the metric ignores, and averaging per-fold calibrators only
  reshuffles the combined ranking for the worse. Calibration is a monotonic transform, so for a
  single model it would not change the ranking (nor the AMS) at all.
- **Measure before you commit.** Both "sure-fire" upgrades (the jet split, the OOF threshold)
  turned out to be ties. A controlled cross-validation predicted that *before* spending a Kaggle
  submission, and the leaderboard confirmed it. Rigorous measurement is what stops you from
  shipping complexity in exchange for noise.

---
---

<a name="español"></a>

# 🇪🇸 Español

**🌐 Idioma / Language:** [English](#english) · **[Español](#español)**

Solución para el **Higgs Boson Machine Learning Challenge** de Kaggle (2014). El objetivo es
separar eventos de **señal** (una colisión que produjo un bosón de Higgs que decae en dos
leptones tau, `H → ττ`) de eventos de **fondo** (background) que producen firmas parecidas en
el detector pero que no son Higgs.

Este modelo alcanza un **AMS ≈ 3.7** en el leaderboard, lo que lo situaría en torno al
**top 100** de la competición original (el ganador rozó 3.806). Este README explica **cada
parte del código, qué significa cada variable y por qué se tomó cada decisión**, para que
sirva como material de aprendizaje.

## Índice

1. [El problema en una frase](#es-1-el-problema-en-una-frase)
2. [La física, sin ecuaciones](#es-2-la-física-sin-ecuaciones)
3. [Los datos: qué significa cada columna](#es-3-los-datos-qué-significa-cada-columna)
4. [La métrica AMS: por qué no usamos accuracy](#es-4-la-métrica-ams-por-qué-no-usamos-accuracy)
5. [Arquitectura de la solución](#es-5-arquitectura-de-la-solución)
6. [Características: por qué no hay ingeniería de características propia](#es-6-featurespy--ingeniería-de-características)
7. [`train_model.py` — entrenamiento paso a paso](#es-7-train_modelpy--entrenamiento-paso-a-paso)
8. [`generate_submission.py` — inferencia y envío](#es-8-generate_submissionpy--inferencia-y-envío)
9. [Por qué cada decisión de diseño](#es-9-por-qué-cada-decisión-de-diseño)
10. [Cómo ejecutarlo](#es-10-cómo-ejecutarlo)
11. [Ideas para mejorar y aprender más](#es-11-ideas-para-mejorar-y-aprender-más)

---

<a name="es-1-el-problema-en-una-frase"></a>

## 1. El problema en una frase

Nos dan **250.000 colisiones** simuladas de las que sabemos la respuesta (señal `s` o fondo `b`)
y **550.000 colisiones** de las que no. Debemos entrenar un modelo con las primeras y decidir,
para cada una de las 550.000 restantes, si es señal o fondo. No se optimiza la precisión clásica,
sino una métrica física llamada **AMS** que mide la "significancia de descubrimiento".

Es, en el fondo, un **problema de clasificación binaria desbalanceada con pesos por evento**.

<a name="es-2-la-física-sin-ecuaciones"></a>

## 2. La física, sin ecuaciones

Cuando dos protones chocan en el LHC (Gran Colisionador de Hadrones), se crean partículas que
se desintegran casi al instante. El detector no ve el Higgs directamente: ve los "restos"
(momento, energía, ángulos de las partículas hijas). El reto es reconocer el **patrón** de
restos que deja un `H → ττ` frente a otros procesos que se le parecen.

Conceptos que aparecen en las variables y conviene tener en la cabeza:

- **`pt` (momento transverso):** momento de la partícula en el plano perpendicular al haz. Es la
  cantidad más informativa porque el momento a lo largo del haz no se conoce bien.
- **`eta` (η, pseudorapidez):** una forma de medir el ángulo respecto al eje del haz. η≈0 es
  perpendicular al haz; η grande es casi paralelo.
- **`phi` (φ, ángulo azimutal):** el ángulo alrededor del haz (0 a 2π). Es un ángulo circular:
  0 y 2π son el mismo sitio.
- **`MET` (Missing Transverse Energy):** energía "que falta". Como los neutrinos escapan sin
  dejar rastro, se deduce su presencia por el desequilibrio de momento. Los tau producen
  neutrinos, así que la MET es una pista fuerte de señal.
- **`jet` (chorro):** un chorro de partículas producido por un quark o gluón. Un evento puede
  tener 0, 1, 2 o 3+ jets.

<a name="es-3-los-datos-qué-significa-cada-columna"></a>

## 3. Los datos: qué significa cada columna

Cada fila es **un evento** (una colisión). El fichero de entrenamiento tiene 33 columnas:
`EventId` + **30 variables físicas** + `Weight` + `Label`. El de test no trae `Weight` ni `Label`.

### Columnas especiales

| Columna | Significado | Uso |
|---|---|---|
| `EventId` | Identificador único del evento | Solo para el envío; **no** es una feature |
| `Weight` | Peso de importancia del evento | Para entrenar y para calcular AMS; **nunca** como feature |
| `Label` | `s` (señal) o `b` (fondo) | La variable objetivo (solo en train) |

> ⚠️ **`Weight` no se usa como variable de entrada.** Es un peso estadístico que indica
> "cuántos eventos reales representa esta fila". La simulación generó muchas más señales de las
> que ocurren en la realidad; los pesos corrigen ese desbalance para que las cuentas físicas
> cuadren. Si lo metieras como feature, el modelo haría trampa (los pesos de señal y fondo son
> sistemáticamente distintos).

### Variables `PRI_` (primitivas): lo que mide el detector

Son medidas "crudas" directas.

| Variable | Significado |
|---|---|
| `PRI_tau_pt` | Momento transverso del tau hadrónico |
| `PRI_tau_eta` | Pseudorapidez (η) del tau |
| `PRI_tau_phi` | Ángulo azimutal (φ) del tau |
| `PRI_lep_pt` | Momento transverso del leptón (electrón o muón) |
| `PRI_lep_eta` | Pseudorapidez del leptón |
| `PRI_lep_phi` | Ángulo azimutal del leptón |
| `PRI_met` | Magnitud de la energía transversa faltante (MET) |
| `PRI_met_phi` | Ángulo azimutal de la MET |
| `PRI_met_sumet` | Energía transversa total depositada en el detector |
| `PRI_jet_num` | Número de jets (0, 1, 2, 3) |
| `PRI_jet_leading_pt` | `pt` del jet más energético |
| `PRI_jet_leading_eta` | η del jet líder |
| `PRI_jet_leading_phi` | φ del jet líder |
| `PRI_jet_subleading_pt` | `pt` del segundo jet |
| `PRI_jet_subleading_eta` | η del segundo jet |
| `PRI_jet_subleading_phi` | φ del segundo jet |
| `PRI_jet_all_pt` | Suma escalar del `pt` de todos los jets |

### Variables `DER_` (derivadas): lo que calculan los físicos

Los físicos ya combinaron las primitivas en cantidades con significado. Son "features hechas por
expertos" y suelen ser las más potentes.

| Variable | Significado |
|---|---|
| `DER_mass_MMC` | Masa estimada del Higgs con el *Missing Mass Calculator*. **La más discriminante.** |
| `DER_mass_transverse_met_lep` | Masa transversa entre la MET y el leptón |
| `DER_mass_vis` | Masa invariante del sistema visible (tau + leptón) |
| `DER_pt_h` | Momento transverso del sistema Higgs (suma vectorial tau + lep + MET) |
| `DER_deltaeta_jet_jet` | \|Δη\| entre los dos jets |
| `DER_mass_jet_jet` | Masa invariante de los dos jets |
| `DER_prodeta_jet_jet` | Producto de las η de los dos jets |
| `DER_deltar_tau_lep` | Separación angular ΔR entre tau y leptón |
| `DER_pt_tot` | Módulo de la suma vectorial de todos los `pt` |
| `DER_sum_pt` | Suma escalar de los `pt` de tau, leptón y jets |
| `DER_pt_ratio_lep_tau` | Cociente `pt_lep / pt_tau` |
| `DER_met_phi_centrality` | Centralidad de la MET respecto a tau y leptón |
| `DER_lep_eta_centrality` | Centralidad de la η del leptón respecto a los dos jets |

### El valor `-999.0`: datos faltantes con sentido físico

Muchas variables no siempre existen. Si un evento tiene **0 jets**, no hay `PRI_jet_leading_pt`;
si tiene **menos de 2 jets**, no hay `DER_mass_jet_jet`. En esos casos el dato es `-999.0`.

**No es ruido: es "no aplica".** Por eso a XGBoost le decimos explícitamente que `-999.0`
significa "faltante" (`missing=-999.0`), para que aprenda a tratarlo como una categoría propia
y no como el número negativo −999.

<a name="es-4-la-métrica-ams-por-qué-no-usamos-accuracy"></a>

## 4. La métrica AMS: por qué no usamos accuracy

La competición se puntúa con el **AMS** (*Approximate Median Significance*). Mide, en unidades de
desviaciones estándar (σ), cómo de significativo sería el "descubrimiento" que hace tu selección
de eventos. En física de partículas se considera un descubrimiento a partir de 5σ.

```
AMS = sqrt( 2 · ( (s + b + b_reg) · ln(1 + s/(b + b_reg)) − s ) )
```

Donde, **sobre los eventos que tu modelo clasifica como señal**:

- `s` = suma de **pesos** de los eventos que **eran** señal (verdaderos positivos, ponderados).
- `b` = suma de **pesos** de los eventos que **eran** fondo (falsos positivos, ponderados).
- `b_reg` = 10 (constante de regularización fija que estabiliza el AMS cuando `b` es pequeño).

Implementación en `train_model.py`:

```python
def calculate_ams(true_labels, predictions, weights, threshold):
    y_pred = (predictions > threshold).astype(int)          # ¿lo llamamos señal?
    s = np.sum(weights[(true_labels == 1) & (y_pred == 1)]) # señal bien cazada (pesada)
    b = np.sum(weights[(true_labels == 0) & (y_pred == 1)]) # fondo colado (pesado)
    br = 10.0
    radicand = 2 * ((s + b + br) * math.log(1.0 + s / (b + br)) - s)
    if radicand < 0:
        return 0.0
    return math.sqrt(radicand)
```

**Consecuencias clave** (y por qué el modelo está diseñado como está):

1. **Cuenta pesos, no eventos.** Por eso los `Weight` son imprescindibles.
2. **Premia la pureza sobre el volumen.** Colar fondo (`b`) penaliza mucho. Conviene ser
   conservador y quedarse solo con los eventos más claramente de señal → **el umbral óptimo es
   alto (~0.9), no 0.5**.
3. **Es una métrica de *ranking* + corte.** Lo único que importa es qué eventos quedan por encima
   del umbral. Esto justifica usar AUC como métrica de parada y ordenar por probabilidad al final.

<a name="es-5-arquitectura-de-la-solución"></a>

## 5. Arquitectura de la solución

```
                    data/training.csv (250k, etiquetado, 30 features)
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  train_model.py                              │
        │  StratifiedKFold 5 folds  ×  seed bagging    │
        │  → 5 modelos XGBoost + 5 umbrales óptimos     │
        └─────────────────────────────────────────────┘
                              │
              higgs_model_fold_1..5.json  +  umbral medio
                              │
                              ▼
                    data/test.csv (550k, sin etiqueta, 30 features)
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  generate_submission.py                       │
        │  media de las 5 probabilidades (ensemble)     │
        │  → ranking → corte por umbral → s/b           │
        └─────────────────────────────────────────────┘
                              │
                              ▼
                   submission_ensembled.csv
```

Dos ideas centrales:

- **K-Fold ensembling:** entrenamos 5 modelos, cada uno con un 80% distinto de los datos. Al
  promediar sus predicciones reducimos la varianza y ganamos robustez.
- **Seed bagging:** cada modelo usa una semilla aleatoria distinta, lo que aumenta la diversidad
  del ensemble (el submuestreo de filas y columnas es diferente en cada uno).

<a name="es-6-featurespy--ingeniería-de-características"></a>

## 6. Características: por qué no hay ingeniería de características propia

El modelo se entrena con las **30 variables oficiales tal cual** (13 `DER_` + 17 `PRI_`). No hay
un paso de ingeniería de características, y es una decisión **deliberada y *medida***, no pereza.

**El experimento.** Una versión anterior añadía una variable creada, `Delta_R_tau_lep`: la
separación angular `ΔR = √(Δη² + Δφ²)` entre el tau y el leptón (una distancia en el plano (η, φ),
donde señal y fondo tienen geometrías distintas). Parecía razonable, pero el dataset **ya trae
`DER_deltar_tau_lep`** — exactamente la misma cantidad calculada por los físicos. Así que la
variable "nueva" era en realidad un duplicado.

Lo verificamos en vez de suponerlo. Una validación cruzada controlada de 5 folds (mismos splits,
seeds y params) midió el AMS de CV con y sin ella:

| Configuración | AMS medio de CV |
|---|---|
| **Con** `Delta_R_tau_lep` | 5.5538 |
| **Sin** `Delta_R_tau_lep` | 5.5426 |
| Δ | −0.0112 (dentro del ruido entre folds) |

La variable **no aportaba nada** (la diferencia es ruido), así que se eliminó. El modelo es más
simple y puntúa al menos igual de bien — **AMS 3.72** en Kaggle, el mejor del proyecto (ver la nota
sobre el umbral en la [sección 11](#es-11-ideas-para-mejorar-y-aprender-más)).

> **Lección:** buena ingeniería de características = añadir información que el modelo **no** tiene
> ya. Recalcular una cantidad que ya es una columna — aunque tenga sentido físico — es esfuerzo
> tirado. Y la forma de saberlo es **medir con validación cruzada**, no fiarse de la intuición.
> Las features genuinamente nuevas (combinaciones de masas, invariancias por rotación de φ, etc.)
> son la palanca de verdad — ver [sección 11](#es-11-ideas-para-mejorar-y-aprender-más).

<a name="es-7-train_modelpy--entrenamiento-paso-a-paso"></a>

## 7. `train_model.py` — entrenamiento paso a paso

### 7.1 Carga y preparación

```python
df = pd.read_csv("data/training.csv")
df["Label"] = df["Label"].map({'s': 1, 'b': 0})   # a binario 1/0

y = df["Label"].values          # objetivo
weights = df["Weight"].values   # pesos físicos
X = df.drop(columns=['EventId', 'Weight', 'Label'])  # solo features
```

Nota que `EventId`, `Weight` y `Label` **se sacan de `X`**: no son variables predictoras.

### 7.2 Validación cruzada + ensemble

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
seeds = [23, 5, 2024, 777, 8888]
```

- **`StratifiedKFold`:** parte los datos en 5 trozos manteniendo la proporción señal/fondo en cada
  uno. Cada iteración entrena con 4 trozos (80%) y valida con 1 (20%).
- **`shuffle=True, random_state=23`:** baraja antes de partir, con semilla fija para
  reproducibilidad.
- **`seeds`:** una semilla distinta por fold → diversidad para el ensemble.

### 7.3 Hiperparámetros de XGBoost

```python
params = {
    'objective': 'binary:logistic',  # clasificación binaria, salida = probabilidad
    'eval_metric': 'auc',            # métrica de parada (ranking)
    'eta': 0.01,                     # learning rate (pasos pequeños y cuidadosos)
    'max_depth': 6,                  # profundidad de cada árbol (control de complejidad)
    'subsample': 0.8,                # usa 80% de filas por árbol (regularización)
    'colsample_bytree': 0.8,         # usa 80% de columnas por árbol (regularización)
}
```

### 7.4 Dentro de cada fold

```python
# 1) Balanceo de clases específico de este fold
sum_wpos = np.sum(w_tr[y_tr == 1])
sum_wneg = np.sum(w_tr[y_tr == 0])
params['scale_pos_weight'] = sum_wneg / sum_wpos
params['seed'] = seeds[fold]

# 2) Reescalar pesos de validación a la escala del test real (550k)
w_va_scaled = w_va * (550000 / len(w_va))

# 3) DMatrix (estructura optimizada de XGBoost) con manejo de faltantes
dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, missing=-999.0)
dval   = xgb.DMatrix(X_va, label=y_va, weight=w_va, missing=-999.0)

# 4) Entrenamiento con early stopping
model = xgb.train(
    params, dtrain,
    num_boost_round=5000,          # techo alto de árboles...
    evals=[(dval, 'validation')],
    early_stopping_rounds=50,      # ...pero para si el AUC no mejora en 50 rondas
    verbose_eval=False
)
```

- **`scale_pos_weight = sum_wneg / sum_wpos`:** la señal (ponderada) es rarísima frente al fondo.
  Este parámetro multiplica el gradiente de la clase positiva para que el modelo **no la ignore**.
  Se recalcula **dentro** del fold, usando solo datos de entrenamiento, para no filtrar
  información de validación (evitar *data leakage*).
- **`weight=w_tr`:** los pesos físicos entran en la función de pérdida. Así el modelo optimiza el
  problema **real** (yields esperados) y no el conteo artificial de la simulación.
- **`w_va * (550000 / len(w_va))`:** un fold de validación tiene ~50k eventos; el test tiene 550k.
  Como el AMS depende de sumas **absolutas** de pesos (y `b_reg=10` es una constante fija), hay
  que reescalar los pesos de validación a la escala del test para que el AMS estimado —y el umbral
  óptimo— sean comparables a los del leaderboard.
- **`missing=-999.0`:** clave. XGBoost aprende una "dirección por defecto" en cada split para los
  valores faltantes, en vez de tratarlos como el número −999.
- **`eta=0.01` + `num_boost_round=5000` + `early_stopping_rounds=50`:** learning rate bajo = pasos
  pequeños y precisos; muchos árboles disponibles; pero early stopping corta cuando el AUC de
  validación deja de mejorar durante 50 rondas y se queda con la mejor iteración. Es la receta
  estándar para exprimir XGBoost sin sobreajustar.

### 7.5 Búsqueda del umbral óptimo

```python
val_preds = model.predict(dval)
best_fold_ams = 0.0
best_fold_threshold = 0.0
for t in np.arange(0.80, 0.99, 0.01):
    ams = calculate_ams(y_va, val_preds, w_va_scaled, threshold=t)
    if ams > best_fold_ams:
        best_fold_ams = ams
        best_fold_threshold = t
```

El modelo devuelve una probabilidad, pero el corte que maximiza el AMS **no es 0.5**. Por el
`scale_pos_weight` y la penalización a los falsos positivos, el óptimo está **alto (~0.9)**. Por
eso se barre solo el rango `[0.80, 0.99)` y se elige el `t` que maximiza el AMS en ese fold.

Al final se guarda el modelo de cada fold (`higgs_model_fold_{i}.json`) y se promedian los 5
umbrales óptimos:

```python
final_threshold = np.mean(fold_thresholds)   # "Robust Mean Threshold"
```

Promediar los 5 umbrales es más robusto que fiarse del óptimo de un solo fold (que puede estar
sobreajustado a su partición concreta).

<a name="es-8-generate_submissionpy--inferencia-y-envío"></a>

## 8. `generate_submission.py` — inferencia y envío

```python
df_test = pd.read_csv('data/test.csv')
X_test = df_test.drop(columns=['EventId'])
dtest = xgb.DMatrix(X_test, missing=-999.0)

# Ensemble: media de las 5 probabilidades
ensemble_probabilities = np.zeros(len(df_test))
for i in range(1, 6):
    model = xgb.Booster()
    model.load_model(f'higgs_model_fold_{i}.json')
    ensemble_probabilities += model.predict(dtest)
final_probabilities = ensemble_probabilities / 5
```

> **Regla de oro:** las features del test deben ser **idénticas** a las del train (mismas
> columnas, mismo orden). Aquí ambos usan simplemente las 30 columnas oficiales directas del CSV,
> así que encaja. Si en train usaras columnas que no reproduces en test (p. ej. features creadas o
> de PCA), XGBoost fallaría por *feature mismatch* o daría predicciones inválidas.

### El formato de envío de Kaggle

El envío necesita 3 columnas: `EventId`, `RankOrder` y `Class`.

```python
submission = submission.sort_values(by='Prob', ascending=True)
submission['RankOrder'] = range(1, len(submission) + 1)   # 1 = menos señal, 550000 = más señal

PROBABILITY_THRESHOLD = 0.940     # umbral robusto del entrenamiento (out-of-fold)
submission['Class'] = np.where(submission['Prob'] > PROBABILITY_THRESHOLD, 's', 'b')
```

- **`RankOrder`:** ordena los 550k eventos de menos a más "señal-like". Ordenamos por
  probabilidad ascendente y numeramos 1…550000. Kaggle lo pide para poder evaluar el AMS a
  distintos cortes.
- **`Class`:** `'s'` si la probabilidad supera el umbral, `'b'` si no. El `0.940` es el umbral
  elegido sobre las predicciones out-of-fold agrupadas (ver [sección 11](#es-11-ideas-para-mejorar-y-aprender-más));
  puntuó **AMS 3.72** en Kaggle, el mejor del proyecto.

El resultado se ordena por `EventId` y se guarda en `submission_ensembled.csv`.

<a name="es-9-por-qué-cada-decisión-de-diseño"></a>

## 9. Por qué cada decisión de diseño

| Decisión | Motivo |
|---|---|
| **XGBoost (gradient boosting sobre árboles)** | Domina en datos tabulares; maneja faltantes de forma nativa; captura interacciones no lineales entre variables físicas. |
| **Usar `Weight` en la pérdida** | El AMS se calcula con pesos; entrenar sin ellos optimizaría el problema equivocado. |
| **`scale_pos_weight` por fold** | La señal ponderada es minoritaria; sin esto el modelo la ignoraría. Recalcularlo por fold evita fugas de datos. |
| **`missing=-999.0`** | `-999` significa "no aplica" (p. ej. sin jets); tratarlo como faltante y no como número es físicamente correcto. |
| **5-Fold + seed bagging** | Reduce varianza y sobreajuste; el ensemble generaliza mejor que un único modelo. |
| **`eval_metric='auc'`** | El objetivo real (AMS) es de ranking; el AUC es un buen sustituto derivable para el early stopping. |
| **`eta` bajo + early stopping** | Convergencia cuidadosa sin sobreajustar; el número de árboles lo decide la validación, no tú a mano. |
| **Umbral ~0.9 y no 0.5** | El AMS penaliza los falsos positivos; conviene ser puro y conservador. |
| **Reescalado de pesos a 550k** | El AMS depende de sumas absolutas de pesos; hay que igualar la escala del test para estimar bien umbral y AMS. |
| **Promedio de umbrales** | Más robusto que el óptimo de un solo fold. |

<a name="es-10-cómo-ejecutarlo"></a>

## 10. Cómo ejecutarlo

Requisitos (ver `requirements.txt`): Python 3.10+, `pandas`, `numpy`, `xgboost>=2.0`,
`scikit-learn`.

```bash
# 1) (opcional) crear entorno e instalar dependencias
pip install -r requirements.txt

# 2) Entrenar: genera higgs_model_fold_1..5.json y muestra el "Robust Mean Threshold"
python train_model.py

# 3) (si hiciera falta) copiar ese umbral en generate_submission.py -> PROBABILITY_THRESHOLD

# 4) Generar el CSV de envío
python generate_submission.py
```

Estructura de datos esperada:

```
data/
├── training.csv   # 250.000 eventos etiquetados (con Weight y Label)
└── test.csv       # 550.000 eventos a predecir
```

Salida: `submission_ensembled.csv`, listo para subir a Kaggle.

<a name="es-11-ideas-para-mejorar-y-aprender-más"></a>

## 11. Ideas para mejorar y aprender más

### Ya probado en este proyecto (y el resultado honesto)

- **Quitar la redundante `Delta_R_tau_lep`** (hecho): neutro en CV (−0.011), así que nos quedamos
  con el modelo más simple. Ver [sección 6](#es-6-featurespy--ingeniería-de-características).
- **Calibración isotónica de las probabilidades**: el AMS **bajó** (3.714 → 3.682). Ver la
  lección de abajo.
- **Modelos por número de jets** (`PRI_jet_num` en {0, 1, ≥2}) **+ umbrales OOF por grupo** — el
  "truco de los ganadores" de manual. La CV controlada dio empate (5.4925 vs 5.4901), y Kaggle lo
  confirmó: **3.716 vs 3.714 — +0.002, dentro del ruido del leaderboard**. No adoptado: 3× los
  modelos y complejidad a cambio de nada real. ¿Por qué tan poco? El manejo nativo de faltantes de
  XGBoost ya aprende la estructura condicional al número de jets, y el split reduce los datos
  disponibles para cada modelo. (Los ganadores que sí ganaron con esto usaban modelos lineales /
  redes, que manejan mucho peor la missingness estructural.)
- **Umbral único sobre predicciones out-of-fold (OOF)** en vez de promediar umbrales por fold
  (**adoptado**): un punto de operación más principled y estable. En CV parecía empate, pero en el
  leaderboard real ganó con claridad — el umbral OOF `0.940` puntuó **3.72**, mientras que la media
  ruidosa por fold `0.934` se quedó en 3.70. Es el `PROBABILITY_THRESHOLD` actual y el mejor
  resultado del proyecto. Además muestra lo sensible que es el AMS al punto de corte: mover el
  umbral 0.006 cambió el score 0.02, sin tocar el modelo.

### Aún vale la pena probar

1. **Features físicas genuinamente nuevas** — rotación de φ (fijar `PRI_tau_phi = 0` y rotar el
   resto, eliminando una simetría rotacional irrelevante), log-transformar variables de cola larga
   (`pt`, masas), combinaciones de masas. Esto es lo que de verdad separaba a los mejores.
2. **Tuning de hiperparámetros** (`max_depth`, `eta`, `min_child_weight`, `gamma`,
   `lambda`/`alpha`) con Optuna, validando siempre con AMS.
3. **Un ensemble diverso con stacking** — combinar modelos base genuinamente distintos (LightGBM,
   una red neuronal) con un meta-modelo simple. El stacking solo ayuda cuando las bases son
   *diversas*; apilar cinco XGBoost casi idénticos no aporta nada.

### Lecciones aprendidas de este proyecto

- **Ajusta la técnica a la métrica.** El AMS-por-umbral es una métrica de **ranking**: solo
  importa el orden de los eventos y dónde cortas. Por eso **la calibración isotónica hizo daño**:
  arregla la *magnitud* de la probabilidad, que la métrica ignora, y promediar calibradores por
  fold solo reordena el ranking combinado a peor. La calibración es una transformación monótona,
  así que para un modelo único no cambiaría el ranking (ni el AMS) en absoluto.
- **Mide antes de comprometerte.** Las dos mejoras "seguras" (el split por jets, el umbral OOF)
  resultaron ser empates. Una validación cruzada controlada lo predijo *antes* de gastar una
  submission a Kaggle, y el leaderboard lo confirmó. Medir con rigor es lo que te evita enviar
  complejidad a cambio de ruido.

---

*Higgs Boson Machine Learning Challenge — Kaggle (2014). XGBoost model with 5-Fold ensembling +
seed bagging, AMS ≈ 3.7 (~top 100).*
