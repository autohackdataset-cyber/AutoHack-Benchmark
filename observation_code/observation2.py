"""
Observation 2 — Non-periodic UDS false-positive analysis

Outputs (saved under Result/Result_ob2/):
  - f1_auc_result_without_uds.txt        : Precision / Recall / F1 / AUC table
  - confusion_matrix_without_uds.png/jpg/pdf : confusion matrix figure
  - cm_without_uds.csv            : raw confusion matrix

Train filter:
  - ID <  0x700 : keep everything (all labels)
  - ID >= 0x700 : keep only Fuzzing (Label == 4)

Test filter:
  - remove Label == 5 (UDS_Spoofing)
  - keep Arbitration_ID >= 0x700 Normal traffic for FP measurement
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

_SCRIPT_START = time.time()


# =============================================================================
# Configuration
# =============================================================================
PROGRAM_PATH = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(PROGRAM_PATH, ".." , "preprocess", "source", "AutoHack")
TRAIN_FILE = os.path.join(SOURCE_PATH, "train_proc.csv")
TEST_FILE  = os.path.join(SOURCE_PATH, "test_proc.csv")
OUTDIR     = os.path.join(PROGRAM_PATH, "..", "Result", "Result_ob2")

os.makedirs(OUTDIR, exist_ok=True)

RF_N_ESTIMATORS = 130
RF_MAX_DEPTH    = 30
N_JOBS          = 18

# UDS filter boundary
UDS_ID_MIN = 0x700   # 1792

# Label maps
LABEL_MAP = {
    0: 'Normal',
    1: 'DoS',
    2: 'Spoofing',
    3: 'Replay',
    4: 'Fuzzing',
    5: 'UDS_Spoofing',
}
CLS_ORDER_WITH_UDS    = ['Normal', 'DoS', 'Spoofing', 'Replay', 'Fuzzing', 'UDS_Spoofing']
CLS_ORDER_WITHOUT_UDS = ['Normal', 'DoS', 'Spoofing', 'Replay', 'Fuzzing']


def process_data(data):
    data.drop(['Timestamp'], axis=1, errors='ignore', inplace=True)
    return data


def save_fig(fig, basename):
    """Save figure as png, jpg, pdf."""
    for ext in ['png', 'jpg', 'pdf']:
        path = os.path.join(OUTDIR, f"{basename}.{ext}")
        fig.savefig(
            path, dpi=150, bbox_inches='tight',
            format='jpeg' if ext == 'jpg' else ext,
        )
        print(f"  saved -> {basename}.{ext}")


# =============================================================================
# 1. Load data
# =============================================================================
print("=" * 70)
print("Observation 2 — Non-periodic UDS false-positive analysis")
print("=" * 70)

print("\n1. Load data")
print("-" * 70)
print(f"  train: {TRAIN_FILE}")
print(f"  test : {TEST_FILE}")

df = process_data(pd.read_csv(TRAIN_FILE))
tf = process_data(pd.read_csv(TEST_FILE))

print(f"  train shape = {df.shape}")
print(f"  test  shape = {tf.shape}")

feature_columns = list(df.columns.difference(['Class', 'Label', 'Bus']))
print(f"  features ({len(feature_columns)}): {feature_columns}")


# =============================================================================
# 2. Train/Test filtering — "without UDS Messages"
# =============================================================================
print("\n" + "=" * 70)
print("2. Filter data — without UDS Messages")
print("=" * 70)
print(f"  Train filter: concat(Label==4, Arbitration_ID < 0x{UDS_ID_MIN:X}) then drop_duplicates")
print(f"                → keeps all Fuzzing + all non-UDS-range messages")
print(f"  Test  filter: remove Label==5 (UDS_Spoofing)")

df_label4    = df[df['Label'] == 4]
df_under_uds = df[df['Arbitration_ID'] < UDS_ID_MIN]
df_B = pd.concat([df_label4, df_under_uds]).drop_duplicates()
df_B = df_B[df_B['Label'] != 5].reset_index(drop=True)

# Test filter
tf_B = tf[tf['Label'] != 5].reset_index(drop=True)

print(f"\n  train: {len(df):,} -> {len(df_B):,} "
      f"(removed {len(df) - len(df_B):,})")
print(f"  test : {len(tf):,} -> {len(tf_B):,} "
      f"(removed {len(tf) - len(tf_B):,})")

train_x_B = df_B[feature_columns]
train_y_B = df_B['Label']
test_x_B  = tf_B[feature_columns]
test_y_B  = tf_B['Label']

print(f"\n  train Label value_counts (filtered):")
print(train_y_B.value_counts().sort_index().to_string())
print(f"\n  test Label value_counts (filtered):")
print(test_y_B.value_counts().sort_index().to_string())

print("\n  Training RandomForest (without UDS) ...")
print(f"    n_estimators = {RF_N_ESTIMATORS}")
print(f"    max_depth    = {RF_MAX_DEPTH}")
print(f"    n_jobs       = {N_JOBS}")

clf_B = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    n_jobs=N_JOBS,
)
print("  Start training model S (without UDS)")
clf_B.fit(train_x_B, train_y_B)
print("  Done.")

pred_B = clf_B.predict(test_x_B)
pred_B_prob = clf_B.predict_proba(test_x_B)

cm_B = confusion_matrix(
    test_y_B, pred_B,
    labels=[0, 1, 2, 3, 4, 5],
)
print(f"\n  confusion matrix (without UDS):")
print(pd.DataFrame(cm_B,
                   index=CLS_ORDER_WITH_UDS,
                   columns=CLS_ORDER_WITH_UDS).to_string())


# =============================================================================
# 3. Per-attack performance (without UDS)
# =============================================================================
print("\n" + "=" * 70)
print("3. Per-attack performance (Excluding UDS)")
print("=" * 70)

S_label_B = test_y_B.map(LABEL_MAP)
predict_S_labels_B = pd.Series(pred_B).map(LABEL_MAP)

report = classification_report(
    S_label_B, predict_S_labels_B,
    zero_division=0, digits=4, output_dict=True,
)

classes_in_model = list(clf_B.classes_)
y_true_int = test_y_B.values.astype(int)
y_true_bin = label_binarize(y_true_int, classes=classes_in_model)

aucs = {}
for i, cls_int in enumerate(classes_in_model):
    cls_name = LABEL_MAP.get(int(cls_int), str(cls_int))
    if cls_name == 'UDS_Spoofing':
        continue  # excluded (UDS_Spoofing)
    try:
        if y_true_bin.shape[1] == 1:
            auc = roc_auc_score(y_true_bin, pred_B_prob[:, i])
        else:
            auc = roc_auc_score(y_true_bin[:, i], pred_B_prob[:, i])
    except Exception:
        auc = float('nan')
    aucs[cls_name] = auc

macro_auc = float(np.nanmean(list(aucs.values())))

header = f"{'Attack Type':15s} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'AUC':>10}"
sep    = "-" * 60
lines  = [
    "Per-attack performance of baseline IDS (Excluding UDS).",
    "",
    header,
    sep,
]

for cls in CLS_ORDER_WITHOUT_UDS:
    if cls in report:
        r = report[cls]
        lines.append(
            f"{cls:15s} {r['precision']:10.4f} {r['recall']:10.4f} "
            f"{r['f1-score']:10.4f} {aucs.get(cls, float('nan')):10.4f}"
        )

lines.append(sep)
for avg_key, avg_label in [('macro avg', 'Macro Avg'), ('weighted avg', 'Weighted Avg')]:
    if avg_key in report:
        r = report[avg_key]
        lines.append(
            f"{avg_label:15s} {r['precision']:10.4f} {r['recall']:10.4f} "
            f"{r['f1-score']:10.4f} {macro_auc:10.4f}"
        )

result = "\n".join(lines)
print("\n" + result)

table9_path = os.path.join(OUTDIR, "f1_auc_result_without_uds.txt")
with open(table9_path, "w", encoding="utf-8") as f:
    f.write(result)
print(f"\n  saved -> f1_auc_result_without_uds.txt")


# =============================================================================
# 4. Confusion matrix (without UDS)
# =============================================================================
print("\n" + "=" * 70)
print("4. Confusion matrix (without UDS)")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    cm_B, annot=True, fmt=",d", cmap="Blues",
    xticklabels=CLS_ORDER_WITH_UDS, yticklabels=CLS_ORDER_WITH_UDS,
    annot_kws={'size': 13, 'weight': 'bold'},
    linewidths=0.5, linecolor='gray', ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=19, labelpad=12)
ax.set_ylabel("True Label",      fontsize=19, labelpad=12)
ax.set_title("Confusion Matrix without UDS Messages",
             fontsize=21, fontweight='bold', pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right',
                   fontweight='bold', fontsize=19)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                   fontweight='bold', fontsize=19)
plt.tight_layout()
save_fig(fig, "confusion_matrix_without_uds")
plt.close(fig)


# =============================================================================
# 5. Save raw confusion matrix as CSV
# =============================================================================
cm_B_out = pd.DataFrame(cm_B, index=CLS_ORDER_WITH_UDS, columns=CLS_ORDER_WITH_UDS)
cm_B_out.to_csv(os.path.join(OUTDIR, "cm_without_uds.csv"))
print(f"\n  saved -> cm_without_uds.csv")


print("\n" + "=" * 70)
print(f"All outputs saved -> {OUTDIR}")
elapsed = time.time() - _SCRIPT_START
print(f"Total execution time: {elapsed:.2f} s ({elapsed/60:.1f} min)")
print("=" * 70)