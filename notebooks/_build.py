"""
_build.py — generate the Colab-ready notebooks from source-of-truth code.

Run:  python notebooks/_build.py
Regenerates notebooks/01..06 .ipynb. Kept in-repo so the notebooks can be
rebuilt if the src/ API changes. The notebooks import the real modules in
../src, so training/eval logic lives in one tested place.
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def write_nb(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    }
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Shared setup cells
# ---------------------------------------------------------------------------
SETUP_PIP = code(
    r"""
# --- Install dependencies (Colab already has TF/torch; this pins the extras) ---
# If a package is already present Colab will skip it quickly.
!pip -q install scikit-learn matplotlib kaggle >/dev/null 2>&1
print("deps ready")
"""
)

SETUP_PATHS = code(
    r"""
# --- Point this at your uploaded project folder -------------------------------
# Easiest: upload the whole project to Google Drive, mount it, and set PROJECT_DIR.
import os, sys

try:
    from google.colab import drive
    drive.mount('/content/drive')
    # EDIT this to where you put the project inside your Drive:
    PROJECT_DIR = '/content/drive/MyDrive/Skin_Cancer_Research_Grant2024'
except Exception:
    # Running locally / not on Colab
    PROJECT_DIR = os.path.abspath('..')

SRC       = os.path.join(PROJECT_DIR, 'src')
MODELS_DIR= os.path.join(PROJECT_DIR, 'models')
DATA_DIR  = os.path.join(PROJECT_DIR, 'data')
FIG_DIR   = os.path.join(PROJECT_DIR, 'report', 'figures')
SPLIT_CSV = os.path.join(MODELS_DIR, 'ham_split.csv')
PREP_JSON = os.path.join(MODELS_DIR, 'preprocessing.json')
for d in (MODELS_DIR, DATA_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

assert os.path.isdir(SRC), f"src/ not found at {SRC} — set PROJECT_DIR correctly"
sys.path.insert(0, SRC)
print("PROJECT_DIR =", PROJECT_DIR)
"""
)

TITLE_TAIL = """
---
*Part of: **A Multimodal, Explainable Hierarchical Framework for Skin-Lesion
Classification with Diffusion-Based Augmentation of Rare Malignancies.***
All training/eval logic lives in the tested modules under `src/`; these
notebooks orchestrate them on a Colab T4.
"""

# ===========================================================================
# 01 — DATA PREP
# ===========================================================================
nb01 = [
    md(
        r"""
# 01 · Data preparation — HAM10000

**Goal.** Download HAM10000, attach each image to its patient metadata row,
derive the binary (benign/malignant) label, and create **one global
stratified split** that every downstream stage reuses (so the end-to-end test
set in notebook 06 is never seen during any training).

We also **fit the metadata preprocessor on the training split only** and persist
it to `models/preprocessing.json` — the exact same object the Streamlit app
loads, guaranteeing train/serve consistency.
"""
        + TITLE_TAIL
    ),
    SETUP_PIP,
    SETUP_PATHS,
    md(
        r"""
## Download HAM10000 from Kaggle
Upload your `kaggle.json` API token (Kaggle → Account → *Create New API Token*).
"""
    ),
    code(
        r"""
from google.colab import files
print("Upload kaggle.json:")
files.upload()  # choose your kaggle.json
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# ~5k+5k images across two folders + HAM10000_metadata.csv
!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p "$DATA_DIR" --unzip
print("downloaded to", DATA_DIR)
"""
    ),
    code(
        r"""
import glob, os
# Locate the metadata CSV and the two image folders wherever Kaggle unzipped them.
meta_csv = glob.glob(os.path.join(DATA_DIR, '**', 'HAM10000_metadata*.csv'), recursive=True)[0]
image_dirs = sorted({os.path.dirname(p) for p in
                     glob.glob(os.path.join(DATA_DIR, '**', '*.jpg'), recursive=True)})
print("metadata:", meta_csv)
print("image dirs:", image_dirs)
"""
    ),
    md("## Build the dataframe + global split (uses the tested `src/data.py`)"),
    code(
        r"""
import pandas as pd
import data as D
import metadata as M

df = D.load_ham_dataframe(meta_csv, image_dirs)
print("rows with images:", len(df))
print("\ndx distribution:\n", df['dx'].value_counts())
print("\nbinary distribution:\n", df['binary_label'].value_counts())

# One shared train/val/test partition (stratified on the 7-way dx).
df = D.assign_global_split(df, test_size=0.15, val_size=0.15)
print("\nsplit counts:\n", df['split'].value_counts())
df.to_csv(SPLIT_CSV, index=False)
print("\nsaved split ->", SPLIT_CSV)
"""
    ),
    md(
        r"""
## Fit + save the metadata preprocessor (**train split only**)
Fitting the age-standardiser and one-hot vocabularies on the training rows only
avoids leaking test statistics into preprocessing.
"""
    ),
    code(
        r"""
train_df = df[df['split'] == 'train']
proc = M.MetadataProcessor().fit(train_df)
proc.save(PREP_JSON)
print("metadata feature_dim =", proc.feature_dim)
print("feature names:", proc.feature_names)
print("saved ->", PREP_JSON)
"""
    ),
    md(
        r"""
### ✅ Outputs of this notebook
* `models/ham_split.csv` — every image + metadata + `split` column
* `models/preprocessing.json` — the fitted `MetadataProcessor`

Proceed to **02** (Stage 1). Notebooks 02–04 all read these two files.
"""
    ),
]

# ===========================================================================
# Reusable training-cell builder for stages
# ===========================================================================
def stage_training_cells(stage, title, subtitle, epochs=30):
    intro = md(
        rf"""
# {title}

{subtitle}

**Design.** We train **two** models on the *same* EfficientNetB3 backbone so any
performance gap is attributable to the metadata, not a different vision model:

1. **Image-only baseline** — the original project's architecture (the control).
2. **Multimodal late-fusion** — image embedding ⊕ MLP over age/sex/site.

Both are trained on the **full** stage data with **balanced class weights** and
on-the-fly augmentation — no undersampling (the key methodological fix). Models
are saved to `models/` with names the inference `pipeline.py` expects.
"""
        + TITLE_TAIL
    )
    setup = code(
        rf"""
import os, json
import pandas as pd
import data as D, metadata as M, model as Model

STAGE = "{stage}"
EPOCHS = {epochs}
BATCH = 32

proc = M.MetadataProcessor.load(PREP_JSON)
df = pd.read_csv(SPLIT_CSV)
sdf = D.filter_for_stage(df, STAGE)
tr = sdf[sdf.split == 'train']; va = sdf[sdf.split == 'val']; te = sdf[sdf.split == 'test']
n_classes = len(D.STAGE_CLASSES[STAGE])
class_weight = D.compute_class_weights(tr, STAGE)
print("classes:", D.STAGE_CLASSES[STAGE], "| n_classes:", n_classes)
print("train/val/test:", len(tr), len(va), len(te))
print("class weights:", class_weight)
"""
    )
    baseline = code(
        rf"""
# ---------------- Image-only baseline (control) ----------------
tf_tr = D.make_dataset(tr, STAGE, None, multimodal=False, batch_size=BATCH, training=True, augment=True)
tf_va = D.make_dataset(va, STAGE, None, multimodal=False, batch_size=BATCH, training=False)

baseline = Model.compile_model(Model.build_image_only_baseline(n_classes))
base_path = os.path.join(MODELS_DIR, "{stage}_baseline.keras")
hist_b = baseline.fit(tf_tr, validation_data=tf_va, epochs=EPOCHS,
                      class_weight=class_weight,
                      callbacks=Model.make_callbacks(base_path))
baseline.save(base_path)  # persist best (EarlyStopping restored best weights)
json.dump(hist_b.history, open(base_path.replace('.keras', '_history.json'), 'w'))
print("saved baseline ->", base_path)
"""
    )
    multimodal = code(
        rf"""
# ---------------- Multimodal late-fusion ----------------
tf_tr_m = D.make_dataset(tr, STAGE, proc, multimodal=True, batch_size=BATCH, training=True, augment=True)
tf_va_m = D.make_dataset(va, STAGE, proc, multimodal=True, batch_size=BATCH, training=False)

mm = Model.compile_model(Model.build_multimodal_model(n_classes, proc.feature_dim))
mm_path = os.path.join(MODELS_DIR, "{stage}_multimodal.keras")
hist_m = mm.fit(tf_tr_m, validation_data=tf_va_m, epochs=EPOCHS,
                class_weight=class_weight,
                callbacks=Model.make_callbacks(mm_path))
mm.save(mm_path)
json.dump(hist_m.history, open(mm_path.replace('.keras', '_history.json'), 'w'))
print("saved multimodal ->", mm_path)
"""
    )
    quicklook = code(
        r"""
# Quick val-accuracy comparison (full metrics come in notebook 06).
print("baseline   best val_acc: %.4f" % max(hist_b.history['val_accuracy']))
print("multimodal best val_acc: %.4f" % max(hist_m.history['val_accuracy']))
"""
    )
    return [intro, SETUP_PIP, SETUP_PATHS, setup, baseline, multimodal, quicklook]


nb02 = stage_training_cells(
    "stage1",
    "02 · Stage 1 — benign vs malignant (multimodal)",
    "The most clinically important stage: **sensitivity to malignancy** is the "
    "number that matters. Trained over **all 7 diagnoses** mapped to "
    "benign/malignant, so it uses every image and stays multimodal.",
)

nb03 = stage_training_cells(
    "benign",
    "03 · Stage 2a — benign subtype {nv, bkl, vasc} (multimodal)",
    "Given a benign decision at Stage 1, classify the benign subtype. `df` "
    "(dermatofibroma, ~115 imgs) is grouped benign at Stage 1 but not "
    "sub-classified here — a documented limitation.",
)

nb04 = stage_training_cells(
    "malignant",
    "04 · Stage 2b — malignant subtype {mel, bcc, akiec} (multimodal)",
    "Given a malignant decision at Stage 1, classify the malignant subtype. "
    "`akiec` is the rare class the diffusion ablation (notebook 05) targets.",
)

# ===========================================================================
# 05 — DIFFUSION AUGMENTATION OF THE RARE CLASS (akiec)
# ===========================================================================
nb05 = [
    md(
        r"""
# 05 · Diffusion augmentation of a rare malignant class (`akiec`)

**Hypothesis.** The malignant sub-classifier struggles most on `akiec`
(actinic keratosis / intraepithelial carcinoma, ~327 images). If we train a
small **DDPM** on the `akiec` training images and add synthetic samples, the
sub-classifier's **`akiec` recall / macro-F1** should improve.

**Ablation.** We retrain the malignant multimodal model on
`real malignant train + synthetic akiec` and save it as
`malignant_multimodal_aug.keras`. Notebook 06 compares it head-to-head against
the no-synthetic model on the identical held-out test slice.

**Honesty note.** Generation is at 64×64 (T4 time budget) and bicubically
upscaled to 224; metadata for synthetic images is sampled from the `akiec`
training distribution. Both are disclosed limitations in the report.
"""
        + TITLE_TAIL
    ),
    code(
        r"""
# diffusers pulls a compatible huggingface_hub itself — do NOT upgrade hub
# separately or its imports break (see src/diffusion.py, which avoids DDPMPipeline).
!pip -q install "diffusers>=0.27,<0.41" accelerate >/dev/null 2>&1
print("diffusers ready")
"""
    ),
    SETUP_PATHS,
    md("## Train a DDPM on the `akiec` training images (time-boxed)"),
    code(
        r"""
import os, pandas as pd
import diffusion as dfn

df = pd.read_csv(SPLIT_CSV)
akiec_train = df[(df.dx == 'akiec') & (df.split == 'train')]
paths = akiec_train['image_path'].tolist()
print("akiec training images:", len(paths))

DIFF_DIR = os.path.join(MODELS_DIR, 'diffusion', 'akiec')
dfn.train_ddpm(
    paths, out_dir=DIFF_DIR,
    image_size=64, epochs=300, batch_size=16, lr=1e-4,
    base_channels=128, use_ema=True,
    max_minutes=120,     # hard wall-clock cap for a free T4 session
    log_every=25,
)
"""
    ),
    md("## Inspect a few samples (sanity check before bulk generation)"),
    code(
        r"""
import matplotlib.pyplot as plt
from PIL import Image

preview = dfn.generate_images(DIFF_DIR, n=8, out_dir='/tmp/preview',
                              image_size=64, upscale_to=None,
                              num_inference_steps=1000, seed=7)
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for ax, p in zip(axes.ravel(), preview):
    ax.imshow(Image.open(p)); ax.axis('off')
fig.suptitle("Synthetic akiec samples (64x64)")
plt.tight_layout(); plt.show()
"""
    ),
    md("## Generate the synthetic set + sample matching metadata"),
    code(
        r"""
import numpy as np
N_SYNTH = 400
SYN_DIR = os.path.join(PROJECT_DIR, 'synthetic', 'akiec')
syn_paths = dfn.generate_images(DIFF_DIR, n=N_SYNTH, out_dir=SYN_DIR,
                                image_size=64, upscale_to=224,
                                num_inference_steps=1000, seed=2024)
print("generated", len(syn_paths), "synthetic akiec images ->", SYN_DIR)

# Sample metadata from the akiec TRAIN distribution (disclosed limitation).
r = np.random.RandomState(2024)
ages = akiec_train['age'].dropna().to_numpy()
sexes = akiec_train['sex'].astype(str).to_numpy()
locs = akiec_train['localization'].astype(str).to_numpy()
syn = pd.DataFrame({'image_path': syn_paths})
syn['age'] = r.choice(ages, len(syn)) if len(ages) else 50.0
syn['sex'] = r.choice(sexes, len(syn)) if len(sexes) else 'unknown'
syn['localization'] = r.choice(locs, len(syn)) if len(locs) else 'unknown'
syn['dx'] = 'akiec'
syn['is_synthetic'] = True
syn.head()
"""
    ),
    md("## Retrain the malignant multimodal model **with** synthetic akiec"),
    code(
        r"""
import json
import data as D, metadata as M, model as Model

STAGE = 'malignant'
proc = M.MetadataProcessor.load(PREP_JSON)

mal = D.filter_for_stage(df, STAGE)
mal_train = mal[mal.split == 'train'].copy()
mal_val   = mal[mal.split == 'val'].copy()

# Label the synthetic rows for this stage and append to the training rows only.
syn_rows = syn.copy()
syn_rows['label'] = 'akiec'
syn_rows['label_idx'] = D.STAGE_CLASSES[STAGE].index('akiec')
syn_rows['split'] = 'train'
aug_train = pd.concat([mal_train, syn_rows], ignore_index=True)
print("real malignant train:", len(mal_train), "-> augmented:", len(aug_train))

class_weight = D.compute_class_weights(aug_train, STAGE)
tf_tr = D.make_dataset(aug_train, STAGE, proc, multimodal=True, batch_size=32, training=True, augment=True)
tf_va = D.make_dataset(mal_val,  STAGE, proc, multimodal=True, batch_size=32, training=False)

mm_aug = Model.compile_model(Model.build_multimodal_model(len(D.STAGE_CLASSES[STAGE]), proc.feature_dim))
aug_path = os.path.join(MODELS_DIR, 'malignant_multimodal_aug.keras')
hist = mm_aug.fit(tf_tr, validation_data=tf_va, epochs=30,
                  class_weight=class_weight,
                  callbacks=Model.make_callbacks(aug_path))
mm_aug.save(aug_path)
json.dump(hist.history, open(aug_path.replace('.keras', '_history.json'), 'w'))
print("saved diffusion-augmented model ->", aug_path)
"""
    ),
    md(
        r"""
### ✅ Outputs
* `models/diffusion/akiec/` — trained DDPM
* `synthetic/akiec/*.png` — synthetic samples
* `models/malignant_multimodal_aug.keras` — the augmented model for the ablation

Notebook 06 runs the head-to-head comparison and reports Δ akiec-recall / Δ macro-F1.
"""
    ),
]

# ===========================================================================
# 06 — EVALUATION, FIGURES, GRAD-CAM
# ===========================================================================
nb06 = [
    md(
        r"""
# 06 · Evaluation — medical metrics, significance, ablations, Grad-CAM

Everything a reviewer expects, computed on the **held-out test split** with the
tested `src/evaluate.py`:

* per-class precision/recall/F1, macro-F1, **sensitivity & specificity**
* **ROC-AUC** (binary at Stage 1, one-vs-rest for subtypes)
* **calibration** (ECE + reliability diagram)
* confusion matrices
* **McNemar's test**: image-only baseline vs. multimodal on the *same* test set
* **diffusion ablation**: malignant model with vs. without synthetic akiec
* **Grad-CAM** montage (does the model look at the lesion?)
* end-to-end hierarchical pipeline accuracy

Figures are written to `report/figures/` and numbers to `report/results.json`
+ `report/figures/summary.md` for pasting into the report.
"""
        + TITLE_TAIL
    ),
    SETUP_PIP,
    SETUP_PATHS,
    code(
        r"""
import os, json
import numpy as np, pandas as pd
from tensorflow.keras.models import load_model
import data as D, metadata as M, evaluate as ev

proc = M.MetadataProcessor.load(PREP_JSON)
df = pd.read_csv(SPLIT_CSV)
results = {}

def predict_probs(model, ds):
    return model.predict(ds, verbose=0)

def eval_stage(stage, pretty):
    # Evaluate baseline + multimodal for one stage on its TEST slice.
    classes = D.STAGE_CLASSES[stage]
    sdf = D.filter_for_stage(df, stage)
    te = sdf[sdf.split == 'test']
    y_true = te['label_idx'].to_numpy()

    base = load_model(os.path.join(MODELS_DIR, f'{stage}_baseline.keras'))
    mm   = load_model(os.path.join(MODELS_DIR, f'{stage}_multimodal.keras'))

    ds_b = D.make_dataset(te, stage, None, multimodal=False, batch_size=64, training=False)
    ds_m = D.make_dataset(te, stage, proc, multimodal=True, batch_size=64, training=False)
    pb = predict_probs(base, ds_b)
    pm = predict_probs(mm, ds_m)

    m_base = ev.summarize_predictions(y_true, pb, classes)
    m_mm   = ev.summarize_predictions(y_true, pm, classes)
    m_mm['ece'] = ev.expected_calibration_error(y_true, pm)
    mcnemar = ev.mcnemar_test(y_true, pb.argmax(1), pm.argmax(1))

    # Figures (multimodal is the headline model).
    ev.plot_confusion_matrix(y_true, pm.argmax(1), classes,
                             os.path.join(FIG_DIR, f'{stage}_confusion.png'),
                             title=f'{pretty} — confusion (multimodal)')
    ev.plot_roc_curves(y_true, pm, classes, os.path.join(FIG_DIR, f'{stage}_roc.png'))
    ev.plot_reliability(y_true, pm, os.path.join(FIG_DIR, f'{stage}_reliability.png'))

    print(f"\n=== {pretty} ===")
    print(f"  baseline   acc={m_base['accuracy']:.3f}  macroF1={m_base['macro_f1']:.3f}")
    print(f"  multimodal acc={m_mm['accuracy']:.3f}  macroF1={m_mm['macro_f1']:.3f}")
    if 'sensitivity' in m_mm:
        print(f"  multimodal sensitivity={m_mm['sensitivity']:.3f}  specificity={m_mm['specificity']:.3f}")
    print(f"  McNemar p (exact)={mcnemar.get('p_value_exact')}")
    return {'baseline': m_base, 'multimodal': m_mm, 'mcnemar': mcnemar}
"""
    ),
    code(
        r"""
results['stage1']    = eval_stage('stage1',    'Stage 1 · benign vs malignant')
results['benign']    = eval_stage('benign',    'Stage 2a · benign subtype')
results['malignant'] = eval_stage('malignant', 'Stage 2b · malignant subtype')
"""
    ),
    md("## Diffusion ablation — malignant model **with vs. without** synthetic akiec"),
    code(
        r"""
classes = D.STAGE_CLASSES['malignant']
mal = D.filter_for_stage(df, 'malignant')
te = mal[mal.split == 'test']
y_true = te['label_idx'].to_numpy()
ds_m = D.make_dataset(te, 'malignant', proc, multimodal=True, batch_size=64, training=False)

akiec_i = classes.index('akiec')
def akiec_recall(probs):
    m = ev.summarize_predictions(y_true, probs, classes)
    return m['per_class']['akiec']['recall'], m['macro_f1']

mm_plain = load_model(os.path.join(MODELS_DIR, 'malignant_multimodal.keras'))
r_plain, f_plain = akiec_recall(predict_probs(mm_plain, ds_m))

ablation = {'no_synth': {'akiec_recall': r_plain, 'macro_f1': f_plain}}
aug_path = os.path.join(MODELS_DIR, 'malignant_multimodal_aug.keras')
if os.path.exists(aug_path):
    mm_aug = load_model(aug_path)
    r_aug, f_aug = akiec_recall(predict_probs(mm_aug, ds_m))
    ablation['with_synth'] = {'akiec_recall': r_aug, 'macro_f1': f_aug}
    print(f"akiec recall : {r_plain:.3f} -> {r_aug:.3f}  (Δ {r_aug-r_plain:+.3f})")
    print(f"macro-F1     : {f_plain:.3f} -> {f_aug:.3f}  (Δ {f_aug-f_plain:+.3f})")
else:
    print("Run notebook 05 first to produce malignant_multimodal_aug.keras")
results['diffusion_ablation'] = ablation
"""
    ),
    md("## End-to-end hierarchical pipeline (on the global test split)"),
    code(
        r"""
from PIL import Image
from pipeline import SkinLesionPipeline, MALIGNANT_CLASSES

pipe = SkinLesionPipeline.from_dir(MODELS_DIR)
test = df[df.split == 'test']
SUBTYPES = ['nv','bkl','vasc','mel','bcc','akiec']   # 'df' can't be sub-classified

def _clean(v):
    return None if (pd.isna(v) or str(v) == 'unknown') else v

n_stage1_correct = 0; n = 0
n_sub_correct = 0; n_sub = 0
mal_tp = 0; mal_fn = 0
for _, row in test.iterrows():
    try:
        img = Image.open(row['image_path']).convert('RGB')
    except Exception:
        continue
    res = pipe.predict(img, age=_clean(row.get('age')),
                       sex=_clean(row.get('sex')),
                       localization=_clean(row.get('localization')),
                       gradcam=False)
    n += 1
    true_mal = row['dx'] in MALIGNANT_CLASSES
    n_stage1_correct += int(res.is_malignant == true_mal)
    if true_mal:
        mal_tp += int(res.is_malignant); mal_fn += int(not res.is_malignant)
    if row['dx'] in SUBTYPES:
        n_sub += 1
        n_sub_correct += int(res.subtype_label == row['dx'])

e2e = {
    'stage1_accuracy': n_stage1_correct / max(n, 1),
    'malignant_sensitivity': mal_tp / max(mal_tp + mal_fn, 1),
    'end_to_end_subtype_accuracy': n_sub_correct / max(n_sub, 1),
    'n_test': int(n),
}
print(json.dumps(e2e, indent=2))
results['end_to_end'] = e2e
"""
    ),
    md("## Grad-CAM montage (Stage-1 multimodal)"),
    code(
        r"""
import matplotlib.pyplot as plt
from gradcam import make_gradcam_heatmap, overlay_heatmap

stage1 = load_model(os.path.join(MODELS_DIR, 'stage1_multimodal.keras'))
sample = df[df.split == 'test'].sample(min(6, (df.split=='test').sum()), random_state=1)

fig, axes = plt.subplots(2, len(sample), figsize=(3*len(sample), 6))
for j, (_, row) in enumerate(sample.iterrows()):
    img = np.asarray(Image.open(row['image_path']).convert('RGB').resize((224,224)), dtype='float32')
    meta = proc.transform_one(age=row.get('age'), sex=str(row.get('sex')),
                              localization=str(row.get('localization'))).astype('float32')
    try:
        hm, idx, conf = make_gradcam_heatmap(stage1, img, meta=meta)
        ov = overlay_heatmap(img, hm)
    except Exception as e:
        ov = img.astype('uint8'); conf = float('nan')
    axes[0, j].imshow(img.astype('uint8')); axes[0, j].set_title(row['dx']); axes[0, j].axis('off')
    axes[1, j].imshow(ov); axes[1, j].set_title(f'CAM {conf:.2f}'); axes[1, j].axis('off')
fig.suptitle('Grad-CAM: top = input, bottom = Stage-1 attention')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'gradcam_montage.png'), dpi=150, bbox_inches='tight')
plt.show()
"""
    ),
    md("## Persist results (JSON + a markdown snippet for the report)"),
    code(
        r"""
ev.save_metrics_json(results, os.path.join(PROJECT_DIR, 'report', 'results.json'))

def _fmt(x): return '—' if x is None else f'{x:.3f}'
lines = ['# Auto-generated results summary', '']
lines.append('## Baseline vs. multimodal (test set)')
lines.append('| Stage | Model | Acc | Macro-F1 | Sensitivity | Spec | ROC-AUC | McNemar p |')
lines.append('|---|---|---|---|---|---|---|---|')
for stage, pretty in [('stage1','Stage 1'),('benign','Stage 2a'),('malignant','Stage 2b')]:
    R = results[stage]; p = R['mcnemar'].get('p_value_exact')
    for mdl in ['baseline','multimodal']:
        m = R[mdl]
        auc = m.get('roc_auc', m.get('roc_auc_ovr_macro'))
        lines.append(f"| {pretty} | {mdl} | {_fmt(m['accuracy'])} | {_fmt(m['macro_f1'])} | "
                     f"{_fmt(m.get('sensitivity'))} | {_fmt(m.get('specificity'))} | {_fmt(auc)} | "
                     f"{'' if mdl=='baseline' else _fmt(p)} |")
lines += ['', '## Diffusion ablation (malignant subtype)']
ab = results['diffusion_ablation']
lines.append('| Variant | akiec recall | macro-F1 |')
lines.append('|---|---|---|')
for k in ['no_synth','with_synth']:
    if k in ab: lines.append(f"| {k} | {_fmt(ab[k]['akiec_recall'])} | {_fmt(ab[k]['macro_f1'])} |")
lines += ['', '## End-to-end pipeline', '']
for k, v in results['end_to_end'].items():
    lines.append(f"- **{k}**: {v:.3f}" if isinstance(v, float) else f"- **{k}**: {v}")

open(os.path.join(FIG_DIR, 'summary.md'), 'w').write('\n'.join(lines))
print('\n'.join(lines))
"""
    ),
]


def main():
    write_nb("01_data_prep_ham10000.ipynb", nb01)
    write_nb("02_stage1_benign_malignant_multimodal.ipynb", nb02)
    write_nb("03_stage2a_benign_type_multimodal.ipynb", nb03)
    write_nb("04_stage2b_malignant_type_multimodal.ipynb", nb04)
    write_nb("05_diffusion_rare_class.ipynb", nb05)
    write_nb("06_evaluation_figures_gradcam.ipynb", nb06)
    print("built 01-06")


if __name__ == "__main__":
    main()



