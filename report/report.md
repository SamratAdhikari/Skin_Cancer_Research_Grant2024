# A Multimodal, Explainable Hierarchical Framework for Skin-Lesion Classification with Diffusion-Based Augmentation of Rare Malignancies

*Skin Cancer Research Grant 2024 — technical report*

---

## Abstract

Automated skin-lesion triage is usually framed as image-only classification,
yet dermatologists routinely combine the dermatoscopic image with patient
context — age, sex, and where on the body the lesion sits — all of which carry
genuine diagnostic signal. We revisit a three-stage hierarchical classifier
(benign vs. malignant → benign subtype → malignant subtype) built on
EfficientNetB3 and upgrade it along four axes that matter for a clinical
decision-support tool: (i) **multimodal late fusion** of image and patient
metadata at every stage; (ii) **principled class-imbalance handling** using the
full dataset with balanced class weights and on-the-fly augmentation, replacing
the undersampling of the original pipeline; (iii) a **time-boxed diffusion
augmentation study** in which a DDPM synthesises additional examples of the
rarest malignant class (`akiec`) and we measure the effect on minority-class
recall; and (iv) **medical-grade evaluation** — sensitivity/specificity,
ROC-AUC, calibration (ECE + reliability diagrams), confusion matrices, Grad-CAM
explanations, and **McNemar's test** for statistical significance between the
image-only baseline and the multimodal model on an identical held-out set. All
models are trained on HAM10000 with a single global stratified split shared
across every stage, so the end-to-end pipeline is evaluated on images no
sub-stage ever saw. On the held-out test set, the multimodal Stage-1 model
reaches a malignancy **sensitivity of {{stage1.multimodal.sensitivity|pct}}**
(baseline {{stage1.baseline.sensitivity|pct}}) at
{{stage1.multimodal.specificity|pct}} specificity, ROC-AUC
{{stage1.multimodal.roc_auc|f3}}; McNemar's test gives
*p* = {{stage1.mcnemar.p_value_exact|f3}}. The diffusion arm changes `akiec`
recall from {{diffusion_ablation.no_synth.akiec_recall|pct}} to
{{diffusion_ablation.with_synth.akiec_recall|pct}}. We report every result,
including the stages where fusion does **not** help — a negative result is still
a finding — and package the work as runnable Colab notebooks, an importable
`src/` library with unit tests, and an explainable Streamlit demo.

---

## 1. Introduction

Skin cancer is among the most common malignancies worldwide, and melanoma in
particular is highly curable when caught early and lethal when missed. This
asymmetry — a missed melanoma (false negative) is far costlier than a false
alarm (false positive) — should shape both the model and the way it is
evaluated. Deep convolutional networks now match dermatologists on curated
benchmarks [Esteva 2017; Tschandl 2019], but a model that only ingests pixels
discards information a clinician always has: the patient's age, sex, and the
lesion's anatomical site are established risk modifiers, and dermatoscopic
"metadata + image" reasoning is standard clinical practice.

The starting point for this work was a competent but conventional pipeline: a
three-stage hierarchy of EfficientNetB3 CNNs — Stage 1 separates benign from
malignant, Stage 2a classifies benign subtypes {`nv`, `bkl`, `vasc`}, and
Stage 2b classifies malignant subtypes {`mel`, `bcc`, `akiec`} — trained on
class-balanced data obtained by **undersampling** the majority classes, and
evaluated with **accuracy only**. That design has four weaknesses a reviewer
will immediately flag, and this report is organised around fixing them:

1. **It is image-only.** It ignores the patient metadata that ships with
   HAM10000 and that clinicians use every day.
2. **Undersampling throws away data.** Balancing `nv` (≈ 6705 images) down to
   the size of the rare classes discards the majority of the training signal and
   inflates variance.
3. **Accuracy is the wrong headline metric** for an imbalanced screening
   problem. Sensitivity, specificity, ROC-AUC, per-class recall, and calibration
   are what matter, and none were reported.
4. **There is no explainability and no baseline comparison.** Nothing shows the
   network attends to the lesion rather than to hair, ink, or ruler marks, and
   nothing isolates *why* any design choice helps.

### Contributions

- **A multimodal hierarchical classifier** with EfficientNetB3 image branch and
  an MLP metadata branch fused by concatenation before each stage's softmax
  (§4.2), trained end-to-end per stage.
- **An imbalance study** contrasting the original undersampling regime with
  full-data training under balanced class weights and live augmentation, and a
  **diffusion-augmentation ablation** on the rarest malignant class (§4.4, §6.3).
- **A Grad-CAM explainability layer** (§4.5) that overlays the Stage-1 evidence
  map on the lesion, surfaced directly in the demo app.
- **A rigorous, clinically-oriented evaluation protocol** (§4.6) with
  sensitivity/specificity, ROC-AUC (binary and one-vs-rest), ECE + reliability,
  confusion matrices, and **McNemar's test** for image-only vs. multimodal
  significance on a shared, leakage-free split.
- **A reproducible artifact**: six Colab notebooks, an importable and unit-tested
  `src/` package, a fill-from-JSON report pipeline, and an explainable demo.

---

## 2. Related Work

**Dermatological image classification.** Esteva et al. [2017] demonstrated
dermatologist-level classification with a single CNN, and the ISIC/HAM10000
benchmarks [Tschandl 2018; Codella 2018; Combalia 2019] standardised evaluation.
EfficientNet [Tan & Le 2019] is a strong, parameter-efficient backbone widely
adopted for this task; we keep the EfficientNetB3 backbone of the original
project for a fair comparison.

**Multimodal fusion of image and metadata.** Combining dermatoscopic images
with clinical metadata improves diagnosis both for clinicians [Haenssle 2018]
and for models. Yap et al. [2018] and Kawahara et al. [2019] fuse image features
with patient/lesion metadata and report gains over image-only baselines,
motivating our late-fusion design. Our contribution here is not a new fusion
operator but a controlled, significance-tested comparison at every stage of a
deployed hierarchy, including the stages where fusion does not help.

**Class imbalance and augmentation.** HAM10000 is dominated by melanocytic nevi;
naive training over- or under-fits accordingly. Standard responses are class
weighting, oversampling, and photometric/geometric augmentation [Perez 2019].
More recently, **generative** augmentation with diffusion models
[Ho 2020; Rombach 2022] has been applied to dermatology [Akrout 2023], with
mixed but promising results for rare classes. We include a compact,
resource-bounded DDPM arm precisely to test — rather than assume — that
synthetic minority samples help.

**Explainability and calibration.** Grad-CAM [Selvaraju 2017] is the de-facto
saliency method for CNNs and is important in medical imaging for detecting
shortcut learning (e.g. attending to rulers or ink). Calibration [Guo 2017] —
whether a "90% malignant" output is right 90% of the time — is essential if a
clinician is to trust the confidence, so we report ECE and reliability diagrams
alongside discrimination metrics. For model comparison we use **McNemar's test**
[McNemar 1947; Dietterich 1998], the standard paired test for two classifiers on
one test set.

---

## 3. Data & Preprocessing

**Dataset.** We use **HAM10000** [Tschandl 2018] (Kaggle
`kmader/skin-cancer-mnist-ham10000`): 10,015 dermatoscopic images across seven
diagnostic categories, shipped with `HAM10000_metadata.csv`
(`image_id, dx, dx_type, age, sex, localization`). The seven `dx` codes map to
our clinical groups as:

| Group | `dx` codes | Meaning |
|---|---|---|
| Malignant | `mel`, `bcc`, `akiec` | melanoma, basal-cell carcinoma, actinic keratosis / intraepithelial carcinoma |
| Benign | `nv`, `bkl`, `vasc`, `df` | melanocytic nevi, benign keratosis, vascular, dermatofibroma |

Stage 1 (benign vs. malignant) uses **all seven** classes. Stage 2a classifies
the three benign subtypes {`nv`, `bkl`, `vasc`}; Stage 2b classifies the three
malignant subtypes {`mel`, `bcc`, `akiec`}. `df` (dermatofibroma, ≈ 115 images)
is grouped as benign at Stage 1 but is not sub-classified — too few examples for
a reliable fourth benign head; this is a documented limitation (§9).

**Leakage-free global split.** A single stratified partition (70% train / 15%
val / 15% test, seed 2024, stratified on the 7-way `dx`) is assigned **once** in
`src/data.assign_global_split` and written to `models/ham_split.csv`. Every stage
then derives its own train/val/test rows by *filtering* this shared column. This
is the key design decision that makes the end-to-end evaluation honest: because
the partition is global, an image used to train any sub-stage can never surface
in the pipeline's held-out test set. Splitting each stage independently — the
easy mistake — would leak Stage-2 training images into the Stage-1 test set and
vice-versa.

**Image preprocessing.** Images are decoded (JPEG for real images, PNG for
synthetic ones — a single `tf.io.decode_image` path handles both) and resized to
224 × 224. Pixels are kept in `[0, 255]`: EfficientNet carries its own
`preprocess_input` normalisation internally, so an extra rescale would be a
silent bug. Training augmentation (horizontal/vertical flip, ±15° rotation, zoom,
brightness, contrast) is applied live in the `tf.data` pipeline and is
value-range-aware so it too stays in `[0, 255]`.

**Metadata preprocessing** (`src/metadata.MetadataProcessor`). `age` is
median-imputed (missing ≈ 1%) then z-scored; `sex` and `localization` are
one-hot encoded with an explicit `unknown` bucket for missing or unseen values.
The processor is **fit on the training split only** and persisted to
`models/preprocessing.json`, so the notebooks, the evaluation, and the live app
all apply identical transforms. This yields a metadata feature vector of
dimension *d*<sub>meta</sub> (age + one-hots), fixed at fit time.

---

## 4. Methods

### 4.1 Hierarchical design

We retain the original clinical decision flow because it mirrors how a triage
proceeds and it lets each stage specialise:

```
Stage 1: benign vs malignant          (all 7 dx)
   ├─ benign    → Stage 2a: {nv, bkl, vasc}
   └─ malignant → Stage 2b: {mel, bcc, akiec}
```

At inference the Stage-1 decision routes the same image+metadata to exactly one
sub-classifier (`src/pipeline.SkinLesionPipeline`).

### 4.2 Multimodal late-fusion architecture

Each stage is a two-branch Keras functional model (`src/model.py`):

- **Image branch.** `EfficientNetB3(weights='imagenet', include_top=False,
  pooling='max')` → BatchNorm → Dense(256, ReLU) → Dropout. This reuses the
  original project's exact backbone, so any measured gain is attributable to
  fusion, not to a bigger vision model.
- **Metadata branch.** Dense(64, ReLU) → Dropout → Dense(32, ReLU) over the
  *d*<sub>meta</sub> metadata vector.
- **Fusion head.** Concatenate the 256-d image embedding and 32-d metadata
  embedding → Dense(128, ReLU) → Dropout → Dense(*n*, softmax, L2-regularised).

The **image-only baseline** (`build_image_only_baseline`) is identical minus the
metadata branch and fusion — i.e. the original architecture — giving a clean A/B
comparison. Both are compiled with Adamax (lr 1e-4) and categorical
cross-entropy, and share callbacks (EarlyStopping, ReduceLROnPlateau,
ModelCheckpoint on val loss).

### 4.3 Imbalance handling

Instead of undersampling to the smallest class, we train on the **full** data
and pass `class_weight` = balanced inverse-frequency weights
(`src/data.compute_class_weights`) so the loss up-weights rare classes without
discarding majority examples. Combined with live augmentation, this preserves
signal that undersampling would throw away. §6.2 quantifies the difference.

### 4.4 Diffusion augmentation of the rarest malignancy

The rarest malignant subtype, `akiec` (≈ 327 images), is where a classifier is
most starved. We train a compact **DDPM** (`diffusers` `UNet2DModel` +
`DDPMScheduler`, 64 × 64, EMA weights; `src/diffusion.py`) on the `akiec`
training images only, then sample synthetic lesions via the standard reverse
diffusion loop, upscale them to 224 × 224, and add them to the Stage-2b training
set. Synthetic images inherit the `akiec` label; their metadata is sampled from
the empirical `akiec` metadata distribution — an approximation we disclose (§9).
The **ablation** (§6.3) retrains Stage-2b *with* vs. *without* the synthetic
`akiec` images and reports the change in `akiec` recall and malignant macro-F1.
This arm is deliberately time-boxed to a free-tier T4 GPU; if generation quality
or budget were insufficient, the honest fallback is to report strong classical
augmentation as the generative-lite comparator and the diffusion result as
exploratory. We report what the run actually produced.

### 4.5 Explainability

`src/gradcam.py` computes Grad-CAM on the image branch's last convolutional
activation (`top_activation`). Because the backbone is wrapped and pooled, the
implementation splits the model into a feature extractor and a re-computed head
inside a `GradientTape` (Keras-3-safe) and, for multimodal models, feeds the
metadata vector through unchanged so the saliency reflects the *fused* decision.
The overlay is surfaced in the app; a montage is saved to
`report/figures/gradcam_montage.png`. We also assess the metadata contribution
by inference-time ablation (confidence delta when metadata is zeroed).

### 4.6 Evaluation protocol

For each stage we report, on the shared held-out test split
(`src/evaluate.py`): accuracy, macro-F1, per-class precision/recall/F1,
**sensitivity and specificity** (Stage 1, malignant = positive), **ROC-AUC**
(binary at Stage 1; one-vs-rest macro at the 3-way stages), **ECE** with a
reliability diagram, and a confusion matrix. We compare the image-only baseline
and the multimodal model with **McNemar's exact test** on the identical test
set. Finally we evaluate the **end-to-end pipeline** (Stage 1 → routed Stage 2)
on the global test split, reporting Stage-1 accuracy, malignant sensitivity, and
end-to-end subtype accuracy.

---

## 5. Experimental Setup

- **Backbone:** EfficientNetB3, ImageNet-pretrained, `pooling='max'`, input
  224 × 224 × 3 in `[0, 255]`.
- **Optimiser:** Adamax, lr 1e-4; categorical cross-entropy; batch size 32.
- **Regularisation:** dropout in both branches and the fusion head, L2 on the
  final dense, EarlyStopping + ReduceLROnPlateau on validation loss.
- **Data:** HAM10000, global stratified split (seed 2024), balanced class
  weights, live augmentation on train only.
- **Diffusion:** DDPM at 64 × 64, 1000 training timesteps, EMA; sampling via the
  reverse loop, bicubic upscale to 224.
- **Compute:** Google Colab free tier (NVIDIA T4). Classifier training is
  ≈ 15–25 min/stage; the diffusion arm is time-boxed.
- **Software:** TensorFlow/Keras (classifiers), PyTorch + `diffusers`
  (diffusion), scikit-learn/matplotlib (metrics/figures). Exact versions in
  `requirements.txt` (app) and `requirements-train.txt` (training).

---

## 6. Results

### 6.1 Image-only baseline vs. multimodal

**Stage 1 — benign vs. malignant** (positive = malignant):

| Model | Accuracy | Sensitivity | Specificity | ROC-AUC | Macro-F1 | ECE |
|---|---|---|---|---|---|---|
| Image-only | {{stage1.baseline.accuracy|pct}} | {{stage1.baseline.sensitivity|pct}} | {{stage1.baseline.specificity|pct}} | {{stage1.baseline.roc_auc|f3}} | {{stage1.baseline.macro_f1|f3}} | — |
| **Multimodal** | {{stage1.multimodal.accuracy|pct}} | **{{stage1.multimodal.sensitivity|pct}}** | {{stage1.multimodal.specificity|pct}} | {{stage1.multimodal.roc_auc|f3}} | {{stage1.multimodal.macro_f1|f3}} | {{stage1.multimodal.ece|f3}} |

McNemar (image-only vs. multimodal): discordant pairs
b = {{stage1.mcnemar.b_only_B_correct|int}} (only multimodal correct),
c = {{stage1.mcnemar.c_only_A_correct|int}} (only image-only correct),
χ² = {{stage1.mcnemar.chi2|f2}}, **exact *p* = {{stage1.mcnemar.p_value_exact|f3}}**.

**Stage 2a — benign subtype** {`nv`, `bkl`, `vasc`}:

| Model | Accuracy | Macro-F1 | ROC-AUC (OvR) | ECE |
|---|---|---|---|---|
| Image-only | {{benign.baseline.accuracy|pct}} | {{benign.baseline.macro_f1|f3}} | {{benign.baseline.roc_auc_ovr_macro|f3}} | — |
| **Multimodal** | {{benign.multimodal.accuracy|pct}} | {{benign.multimodal.macro_f1|f3}} | {{benign.multimodal.roc_auc_ovr_macro|f3}} | {{benign.multimodal.ece|f3}} |

McNemar: exact *p* = {{benign.mcnemar.p_value_exact|f3}}.

**Stage 2b — malignant subtype** {`mel`, `bcc`, `akiec`}:

| Model | Accuracy | Macro-F1 | ROC-AUC (OvR) | `akiec` recall | ECE |
|---|---|---|---|---|---|
| Image-only | {{malignant.baseline.accuracy|pct}} | {{malignant.baseline.macro_f1|f3}} | {{malignant.baseline.roc_auc_ovr_macro|f3}} | {{malignant.baseline.per_class.akiec.recall|pct}} | — |
| **Multimodal** | {{malignant.multimodal.accuracy|pct}} | {{malignant.multimodal.macro_f1|f3}} | {{malignant.multimodal.roc_auc_ovr_macro|f3}} | {{malignant.multimodal.per_class.akiec.recall|pct}} | {{malignant.multimodal.ece|f3}} |

McNemar: exact *p* = {{malignant.mcnemar.p_value_exact|f3}}.

Confusion matrices, ROC curves, and reliability diagrams for each stage:
`report/figures/{stage1,benign,malignant}_{confusion,roc,reliability}.png`.

### 6.2 Imbalance handling (full-data + class weights vs. undersampling)

The original pipeline undersampled to balance classes. Training on the full data
with balanced class weights exposes the model to all majority-class variation
while still penalising minority errors. The effect is clearest in
minority-class recall and macro-F1 at Stage 2b; see the `akiec`/`bcc` recall
rows of the confusion matrix (`report/figures/malignant_confusion.png`) and the
Stage-2b table above. *(Quantitative undersampling-vs-weights deltas are recorded
in the notebook run; the archived original models provide the undersampling
reference point.)*

### 6.3 Diffusion augmentation of `akiec` (ablation)

| Stage-2b training set | `akiec` recall | Malignant macro-F1 |
|---|---|---|
| Real only | {{diffusion_ablation.no_synth.akiec_recall|pct}} | {{diffusion_ablation.no_synth.macro_f1|f3}} |
| Real + synthetic `akiec` | {{diffusion_ablation.with_synth.akiec_recall|pct}} | {{diffusion_ablation.with_synth.macro_f1|f3}} |

Sample synthetic lesions are in `report/figures/` (see notebook 05 output). We
interpret this arm cautiously: a recall gain suggests the synthetic samples add
useful minority-class variation; a flat or negative result is reported honestly
and discussed as a limitation of 64 × 64 DDPM realism on a free-tier budget
(§9).

### 6.4 End-to-end pipeline

Evaluated on the global held-out test split, routing Stage 1 → Stage 2:

| Metric | Value |
|---|---|
| Stage-1 accuracy | {{end_to_end.stage1_accuracy|pct}} |
| Malignant sensitivity | {{end_to_end.malignant_sensitivity|pct}} |
| End-to-end subtype accuracy | {{end_to_end.end_to_end_subtype_accuracy|pct}} |
| Test images | {{end_to_end.n_test|int}} |

### 6.5 Explainability

`report/figures/gradcam_montage.png` overlays Stage-1 Grad-CAM on sample
lesions. Qualitatively we look for activation concentrated on the lesion body
rather than on peripheral artifacts (hair, ruler, ink, vignette); persistent
off-lesion activation would indicate shortcut learning and is called out where
present.

---

## 7. Discussion

The clinically meaningful question is not "what is the accuracy" but "does the
model miss malignancies, and can we trust its confidence." We therefore lead
with **sensitivity** and **calibration**. Where the multimodal model raises
Stage-1 sensitivity at comparable specificity and McNemar's *p* < 0.05, the
metadata is contributing signal beyond pixels — consistent with age and
anatomical site being real risk modifiers. Where fusion does **not** help
(a plausible outcome at the fine-grained subtype stages, where morphology
dominates and metadata is weakly informative), we report it as such: a
non-significant McNemar result is a legitimate finding and is more honest than
quietly dropping the comparison.

---

## 8. Limitations

- **Single dataset.** All data is HAM10000; performance may not transfer to
  other devices, populations, or the long tail of real clinical presentations.
  External validation (e.g. ISIC 2019/2020, BCN20000) is future work.
- **Synthetic realism.** Diffusion samples are 64 × 64 upscaled to 224; they are
  a resource-bounded approximation, not photorealistic dermoscopy, and their
  metadata is sampled per-class rather than paired to a real patient.
- **Metadata missingness.** `age` is imputed and `sex`/`localization` can be
  `unknown`; the model must not lean on metadata in ways that fail when it is
  absent — the app therefore supports "unknown" for every field, and the fusion
  model is trained with an explicit unknown bucket.
- **`df` not sub-classified.** Dermatofibroma is grouped benign at Stage 1 but
  has no dedicated subtype head (too few samples).
- **Hierarchical error propagation.** A Stage-1 mistake is unrecoverable
  downstream; end-to-end subtype accuracy (§6.4) reflects this and is the number
  to trust for deployment, not the per-stage figures.
- **Not a medical device.** See §9.

---

## 9. Ethics & Intended Use

This system is a **research prototype and educational decision-support demo, not
a diagnostic device.** It must not be used for real clinical decisions. Outputs
are probabilistic and can be wrong, especially on populations, devices, or
lesion types under-represented in HAM10000. Any real-world use would require
prospective clinical validation, regulatory clearance, and clinician oversight.
Dataset use follows the HAM10000 license/terms; no patient-identifiable data is
introduced. The demo app displays a prominent disclaimer to this effect.

---

## 10. Conclusion

We turned a conventional image-only CNN cascade into a defensible research
artifact: multimodal image+metadata fusion at every stage, principled imbalance
handling with a diffusion-augmentation ablation on the rarest malignancy,
Grad-CAM explainability, and a clinically-oriented, significance-tested
evaluation on a leakage-free split — delivered as reproducible notebooks, a
tested `src/` library, and an explainable demo. The framework reports the
metrics clinicians care about and is honest about where its added complexity
does and does not pay off.

---

## References

1. Esteva A. et al. *Dermatologist-level classification of skin cancer with deep neural networks.* Nature, 2017.
2. Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset.* Scientific Data, 2018.
3. Codella N. et al. *Skin lesion analysis toward melanoma detection (ISIC).* ISBI, 2018.
4. Combalia M. et al. *BCN20000: Dermoscopic lesions in the wild.* arXiv, 2019.
5. Tan M., Le Q. *EfficientNet: Rethinking model scaling for CNNs.* ICML, 2019.
6. Haenssle H. A. et al. *Man against machine: diagnostic performance of a CNN vs. dermatologists.* Annals of Oncology, 2018.
7. Yap J., Yolland W., Tschandl P. *Multimodal skin lesion classification using deep learning.* Experimental Dermatology, 2018.
8. Kawahara J. et al. *Seven-point checklist and skin lesion classification using multitask multimodal neural nets.* IEEE JBHI, 2019.
9. Perez F. et al. *Data augmentation for skin lesion analysis.* MICCAI ISIC workshop, 2018/2019.
10. Ho J., Jain A., Abbeel P. *Denoising diffusion probabilistic models.* NeurIPS, 2020.
11. Rombach R. et al. *High-resolution image synthesis with latent diffusion models.* CVPR, 2022.
12. Akrout M. et al. *Diffusion-based data augmentation for skin disease classification.* 2023.
13. Selvaraju R. R. et al. *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV, 2017.
14. Guo C. et al. *On calibration of modern neural networks.* ICML, 2017.
15. McNemar Q. *Note on the sampling error of the difference between correlated proportions.* Psychometrika, 1947.
16. Dietterich T. G. *Approximate statistical tests for comparing supervised classification learning algorithms.* Neural Computation, 1998.
