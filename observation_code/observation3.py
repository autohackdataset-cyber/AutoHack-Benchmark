"""
Observation 3 — Multi-bus characteristics analysis for CAN IDS (RF + XGBoost)

Outputs (saved under Result/Result_ob3/):
  - summary_macro_f1_{idx:03d}.csv
  - top5_feature_importance_{idx:03d}.csv
  - prediction_compare_RF_{idx:03d}.csv
  - overlap_ratio_by_attack_RF_{idx:03d}.csv
  - execution_time_{idx:03d}.txt
  - observation3_report_{idx:03d}.txt
"""

from __future__ import annotations

import glob
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC_DIR  = os.path.join(BASE_PATH, "preprocess", "source", "AutoHack_38f")

TRAIN_PKL_FULL = os.path.join(PROC_DIR, "train_proc_38f.pkl")
TRAIN_PKL_B    = os.path.join(PROC_DIR, "train_proc_b_38f.pkl")
TRAIN_PKL_C    = os.path.join(PROC_DIR, "train_proc_c_38f.pkl")
TRAIN_PKL_P    = os.path.join(PROC_DIR, "train_proc_p_38f.pkl")
TEST_PKL_FULL  = os.path.join(PROC_DIR, "test_proc_38f.pkl")
TEST_PKL_B     = os.path.join(PROC_DIR, "test_proc_b_38f.pkl")
TEST_PKL_C     = os.path.join(PROC_DIR, "test_proc_c_38f.pkl")
TEST_PKL_P     = os.path.join(PROC_DIR, "test_proc_p_38f.pkl")
FEAT_TXT       = os.path.join(PROC_DIR, "feature_columns.txt")

RESULT_DIR = os.path.join(BASE_PATH, "Result", "Result_ob3")

# Interface ordering adopted in train_38f.py.
INTERFACES: Tuple[str, ...] = ("B-CAN", "C-CAN", "P-CAN")

# Columns that are NOT model inputs. feature_columns.txt overrides this
# when available.
NON_FEATURE_COLUMNS: Tuple[str, ...] = ("CAN_ID", "Interface", "Label")

# Random Forest hyperparameters matched to train_38f.py.
RF_PARAMS: Dict = dict(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)

# XGBoost hyperparameters matched to train_38f.py.
XGB_PARAMS: Dict = dict(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss",
)

# Label values that identify Fuzzing messages when computing overlap IDs.
FUZZING_LABELS = {"Fuzzing", "fuzzing", 4, "4"}

# Model families evaluated in parallel.
MODEL_TYPES: Tuple[str, ...] = ("RF", "XGB")

OVERLAP_MODEL: str = "RF"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BusSplit:
    """Preprocessed train/test split for a single bus."""

    bus: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    meta_test: pd.DataFrame   # CAN_ID, Interface, Label


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_pickle(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_feature_columns(fallback_df: pd.DataFrame) -> List[str]:
    if os.path.exists(FEAT_TXT):
        with open(FEAT_TXT, "r", encoding="utf-8") as f:
            cols = [line.strip() for line in f if line.strip()]
        return cols
    return [c for c in fallback_df.columns if c not in NON_FEATURE_COLUMNS]


def _build_split(df_train: pd.DataFrame, df_test: pd.DataFrame,
                 feature_cols: List[str], bus: str) -> BusSplit:
    meta_cols = [c for c in ("CAN_ID", "Interface", "Label") if c in df_test.columns]
    return BusSplit(
        bus=bus,
        X_train=df_train[feature_cols].copy(),
        y_train=df_train["Label"].astype(str).reset_index(drop=True),
        X_test=df_test[feature_cols].copy(),
        y_test=df_test["Label"].astype(str).reset_index(drop=True),
        meta_test=df_test[meta_cols].reset_index(drop=True),
    )


def load_all() -> Tuple[Dict[str, BusSplit], BusSplit, List[str]]:
    """Load the full multi-bus split and per-bus splits from pickle files."""
    if not os.path.exists(TRAIN_PKL_FULL) or not os.path.exists(TEST_PKL_FULL):
        raise FileNotFoundError(
            "Preprocessed 38-feature pickles not found under "
            f"{PROC_DIR}. Run preprocess_38f.py first."
        )

    print("=" * 80)
    print("LOADING PROC FILES")
    print("=" * 80)

    train_full = _load_pickle(TRAIN_PKL_FULL)
    test_full  = _load_pickle(TEST_PKL_FULL)
    print(f"  train (full): {train_full.shape}")
    print(f"  test  (full): {test_full.shape}")

    per_bus_train: Dict[str, pd.DataFrame] = {
        "B-CAN": _load_pickle(TRAIN_PKL_B),
        "C-CAN": _load_pickle(TRAIN_PKL_C),
        "P-CAN": _load_pickle(TRAIN_PKL_P),
    }
    per_bus_test: Dict[str, pd.DataFrame] = {
        "B-CAN": _load_pickle(TEST_PKL_B),
        "C-CAN": _load_pickle(TEST_PKL_C),
        "P-CAN": _load_pickle(TEST_PKL_P),
    }
    for bus in INTERFACES:
        print(f"  {bus}: train={per_bus_train[bus].shape}, test={per_bus_test[bus].shape}")

    feature_cols = _load_feature_columns(train_full)
    print(f"  features ({len(feature_cols)}): {feature_cols[:5]} ...")

    per_bus_split: Dict[str, BusSplit] = {
        bus: _build_split(per_bus_train[bus], per_bus_test[bus], feature_cols, bus)
        for bus in INTERFACES
    }
    multi_split = _build_split(train_full, test_full, feature_cols, "ALL")
    return per_bus_split, multi_split, feature_cols


# ---------------------------------------------------------------------------
# Overlap-ID computation
# ---------------------------------------------------------------------------

def compute_overlap_ids(per_bus_train: Dict[str, pd.DataFrame]) -> set:
    """IDs appearing on 2+ buses in the training data with Fuzzing excluded."""
    fuzzing_strings = {str(v) for v in FUZZING_LABELS}
    id_to_buses: Dict[int, set] = {}
    for bus, df in per_bus_train.items():
        mask = ~df["Label"].astype(str).isin(fuzzing_strings)
        ids = df.loc[mask, "CAN_ID"].astype(int).unique()
        for arb_id in ids:
            id_to_buses.setdefault(int(arb_id), set()).add(bus)
    return {arb_id for arb_id, buses in id_to_buses.items() if len(buses) >= 2}


# ---------------------------------------------------------------------------
# Model factory and wrappers
# ---------------------------------------------------------------------------

def _make_model(model_type: str):
    if model_type == "RF":
        return RandomForestClassifier(**RF_PARAMS)
    if model_type == "XGB":
        return xgb.XGBClassifier(**XGB_PARAMS)
    raise ValueError(f"Unsupported model type: {model_type}")


def _fit(model_type: str, X: pd.DataFrame, y: pd.Series,
         label_encoder: Optional[LabelEncoder] = None):
    """Fit a classifier. XGBoost requires integer-encoded labels."""
    model = _make_model(model_type)
    if model_type == "XGB":
        assert label_encoder is not None, "XGBoost requires a LabelEncoder."
        y_enc = label_encoder.transform(y)
        model.fit(X, y_enc)
    else:
        model.fit(X, y)
    return model


def _predict(model_type: str, model, X: pd.DataFrame,
             label_encoder: Optional[LabelEncoder] = None) -> np.ndarray:
    """Return string-valued predictions to keep downstream logic uniform."""
    y_pred = model.predict(X)
    if model_type == "XGB":
        assert label_encoder is not None, "XGBoost requires a LabelEncoder."
        y_pred = label_encoder.inverse_transform(y_pred)
    return np.asarray(y_pred).astype(str)


def _evaluate_macro_f1(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def top_k_feature_importance(model, feature_names: List[str], k: int = 5) -> pd.DataFrame:
    importances = np.asarray(model.feature_importances_, dtype=float)
    s = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(k)
    return pd.DataFrame({
        "Rank": np.arange(1, len(s) + 1),
        "Feature": s.index,
        "Importance": s.values,
    })


# ---------------------------------------------------------------------------

def get_next_index(save_dir: str) -> int:
    patterns = ["summary_macro_f1_*.csv", "observation3_report_*.txt"]
    indices: List[int] = []
    for pat in patterns:
        for path in glob.glob(os.path.join(save_dir, pat)):
            base = os.path.basename(path)
            stem = os.path.splitext(base)[0]
            tail = stem.rsplit("_", 1)[-1]
            try:
                indices.append(int(tail))
            except ValueError:
                continue
    return (max(indices) + 1) if indices else 1


def _suffix(idx: int) -> str:
    return f"_{idx:03d}"


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------

def _run_single_model(
    model_type: str,
    per_bus: Dict[str, BusSplit],
    multi: BusSplit,
    feature_cols: List[str],
    overlap_ids: set,
    label_encoder: Optional[LabelEncoder],
    timings: Dict[str, float],
) -> Dict[str, object]:
    """Train both regimes for a model family and build the reported tables.

    Always produced (both RF and XGBoost):
      - Macro-F1 summary (single-bus vs. multi-bus, per bus)
      - Top-5 feature importance

    Produced only when ``model_type == OVERLAP_MODEL`` (Random Forest):
      - Per-message prediction_compare table
      - Overlap-ID ratio by attack class
    """
    print("\n" + "=" * 80)
    print(f"TRAINING PIPELINE - {model_type}")
    print("=" * 80)

    want_overlap = (model_type == OVERLAP_MODEL)

    # Single-bus models.
    t0 = time.perf_counter()
    single_models: Dict[str, object] = {}
    for bus in INTERFACES:
        print(f"  [{model_type}] fitting single-bus model for {bus} ...")
        single_models[bus] = _fit(
            model_type,
            per_bus[bus].X_train,
            per_bus[bus].y_train,
            label_encoder=label_encoder,
        )
    timings[f"single_bus_training_{model_type}_sec"] = time.perf_counter() - t0

    # Multi-bus model.
    t0 = time.perf_counter()
    print(f"  [{model_type}] fitting multi-bus model ...")
    multi_model = _fit(
        model_type, multi.X_train, multi.y_train, label_encoder=label_encoder
    )
    timings[f"multi_bus_training_{model_type}_sec"] = time.perf_counter() - t0

    # Evaluation.
    #
    # Single-bus: train on one bus, evaluate on the same bus's test split.
    # Multi-bus : train on the union of all buses, predict the ENTIRE
    #             multi-bus test set in a single call, then slice the
    #             predictions by Interface to report per-bus macro-F1.
    t0 = time.perf_counter()
    summary_rows: List[Dict] = []
    per_bus_single_pred: Dict[str, np.ndarray] = {}
    per_bus_multi_pred: Dict[str, np.ndarray] = {}

    # Single-bus predictions (per-bus models on per-bus test splits).
    per_bus_single_f1: Dict[str, float] = {}
    for bus in INTERFACES:
        sp = _predict(model_type, single_models[bus], per_bus[bus].X_test,
                      label_encoder=label_encoder)
        per_bus_single_pred[bus] = sp
        per_bus_single_f1[bus] = _evaluate_macro_f1(per_bus[bus].y_test, sp)

    # Multi-bus prediction: a single call on the full multi-bus test set.
    overall_multi_pred = _predict(model_type, multi_model, multi.X_test,
                                  label_encoder=label_encoder)

    # Slice by Interface metadata to obtain per-bus evaluation.
    multi_interface = multi.meta_test["Interface"].astype(str).reset_index(drop=True)
    multi_y_test = multi.y_test.astype(str).reset_index(drop=True)
    overall_multi_pred_s = pd.Series(overall_multi_pred, name="pred").astype(str)

    per_bus_multi_f1: Dict[str, float] = {}
    for bus in INTERFACES:
        bus_mask = (multi_interface == bus).values
        y_true_bus = multi_y_test[bus_mask]
        y_pred_bus = overall_multi_pred_s[bus_mask].values
        per_bus_multi_pred[bus] = y_pred_bus
        per_bus_multi_f1[bus] = _evaluate_macro_f1(y_true_bus, y_pred_bus)

    for bus in INTERFACES:
        single_f1 = per_bus_single_f1[bus]
        multi_f1  = per_bus_multi_f1[bus]
        summary_rows.append({
            "Model": model_type,
            "Bus": bus,
            "Single_Bus_MacroF1": round(single_f1, 4),
            "Multi_Bus_MacroF1":  round(multi_f1, 4),
            "Delta":              round(multi_f1 - single_f1, 4),
        })
        print(f"  [{model_type}] {bus}: single={single_f1:.4f}, "
              f"multi={multi_f1:.4f}, delta={multi_f1 - single_f1:+.4f}")

    summary_df = pd.DataFrame(summary_rows)
    timings[f"evaluation_{model_type}_sec"] = time.perf_counter() - t0

    # Top-5 feature importance.
    importance_rows: List[pd.DataFrame] = []
    for bus in INTERFACES:
        df = top_k_feature_importance(single_models[bus], feature_cols, k=5)
        df.insert(0, "Experiment", f"Single-{bus}")
        df.insert(0, "Model", model_type)
        importance_rows.append(df)
    df_multi = top_k_feature_importance(multi_model, feature_cols, k=5)
    df_multi.insert(0, "Experiment", "Multi-bus")
    df_multi.insert(0, "Model", model_type)
    importance_rows.append(df_multi)
    importance_df = pd.concat(importance_rows, axis=0, ignore_index=True)

    # If this model family is not the overlap analysis target, we stop here.
    if not want_overlap:
        return {
            "summary_df":         summary_df,
            "importance_df":      importance_df,
            "prediction_compare": None,
            "overlap_df":         None,
        }

    # ----------------------------------------------------------------------
    # Overlap-ID analysis 
    # ----------------------------------------------------------------------

    # Prediction comparison. We build the comparison directly from the
    # multi-bus test set and align each row with the corresponding
    # single-bus prediction using the Interface column. This guarantees
    # that the multi-bus predictions used here are exactly those used to
    # compute the per-bus macro-F1 above.
    t0 = time.perf_counter()

    multi_meta = multi.meta_test.reset_index(drop=True)
    multi_true = multi.y_test.astype(str).reset_index(drop=True)
    multi_interface_str = multi_meta["Interface"].astype(str)

    single_pred_full = np.empty(len(multi_meta), dtype=object)
    for bus in INTERFACES:
        bus_rows = (multi_interface_str == bus).values
        n_expected = int(bus_rows.sum())
        n_have = len(per_bus_single_pred[bus])
        if n_expected != n_have:
            raise ValueError(
                f"Size mismatch for {bus}: multi-bus test has "
                f"{n_expected} rows with Interface=={bus}, but per-bus "
                f"single-bus predictions have {n_have} rows. Check that "
                f"the combined and per-bus pickle files are consistent."
            )
        single_pred_full[bus_rows] = per_bus_single_pred[bus]

    frame = pd.DataFrame({
        "CAN_ID":         multi_meta["CAN_ID"].astype(int),
        "Interface":      multi_interface_str,
        "True_Label":     multi_true,
        "Single_Result":  pd.Series(single_pred_full).astype(str),
        "Combined_Result": overall_multi_pred_s,
    })
    frame["Match_Type"] = _vectorized_match(frame)
    frame["ID_Overlap"] = frame["CAN_ID"].isin(overlap_ids)
    prediction_compare = frame
    timings[f"prediction_compare_{model_type}_sec"] = time.perf_counter() - t0

    # Overlap ratio by attack class.
    class_order = _infer_class_order(prediction_compare["True_Label"])
    overlap_rows: List[Dict] = []
    for cls in class_order:
        cls_mask = prediction_compare["True_Label"] == cls
        total_cls = int(cls_mask.sum())
        overlap_all = int((cls_mask & prediction_compare["ID_Overlap"]).sum())
        multi_wrong_mask = cls_mask & (
            prediction_compare["Combined_Result"] != prediction_compare["True_Label"]
        )
        total_multi_wrong = int(multi_wrong_mask.sum())
        overlap_multi_wrong = int((multi_wrong_mask & prediction_compare["ID_Overlap"]).sum())
        overlap_rows.append({
            "Model": model_type,
            "Class": cls,
            "Total_Messages": total_cls,
            "Overlap_Ratio_All_%": _safe_pct(overlap_all, total_cls),
            "Multi_Misclassified": total_multi_wrong,
            "Overlap_Ratio_MultiMisclassified_%": _safe_pct(overlap_multi_wrong, total_multi_wrong),
        })
    overlap_df = pd.DataFrame(overlap_rows)

    return {
        "summary_df":         summary_df,
        "importance_df":      importance_df,
        "prediction_compare": prediction_compare,
        "overlap_df":         overlap_df,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vectorized_match(df: pd.DataFrame) -> pd.Series:
    s_ok = df["Single_Result"] == df["True_Label"]
    m_ok = df["Combined_Result"] == df["True_Label"]
    out = np.where(
        s_ok & m_ok, "Both_Correct",
        np.where(
            s_ok & ~m_ok, "Single_Only_Correct",
            np.where(~s_ok & m_ok, "Multi_Only_Correct", "Both_Wrong"),
        ),
    )
    return pd.Series(out, index=df.index, name="Match_Type")


def _infer_class_order(labels: pd.Series) -> List[str]:
    canonical = ["Normal", "Flooding", "DoS", "Spoofing", "Replay",
                 "Fuzzing", "UDS", "UDS_Spoofing"]
    present = [str(v) for v in sorted(labels.astype(str).unique())]
    ordered = [c for c in canonical if c in present]
    tail = [c for c in present if c not in ordered]
    return ordered + tail


def _safe_pct(num: int, denom: int) -> float:
    if denom <= 0:
        return float("nan")
    return round(100.0 * num / denom, 2)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _write_timing_file(path: str, timings: Dict[str, float]) -> None:
    lines = ["Execution Time Breakdown (seconds)", "=" * 40]
    for key, value in timings.items():
        lines.append(f"{key:45s} {value:10.2f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_report(
    path: str,
    *,
    combined_summary_df: pd.DataFrame,
    combined_importance_df: pd.DataFrame,
    overlap_df: Optional[pd.DataFrame],
    n_overlap_ids: int,
    timings: Dict[str, float],
) -> None:
    sections: List[str] = []
    sections.append("Observation 3: Multi-bus Characteristics of CAN IDS")
    sections.append("=" * 72)
    sections.append("")
    sections.append(
        "This report summarises the results included in the paper for the "
        "Observation 3 section, using the AutoHack2025 dataset (38-feature "
        "configuration). Results are reported on the held-out test "
        "partition. Macro-F1 and feature importance are provided for "
        "both Random Forest and XGBoost; the overlap-ID analysis is "
        "provided for Random Forest, consistent with the paper."
    )
    sections.append("")

    sections.append("1. Macro-F1 Summary (Single-bus vs. Multi-bus)")
    sections.append("-" * 72)
    sections.append(combined_summary_df.to_string(index=False))
    sections.append("")

    sections.append("2. Top-5 Feature Importance per Experiment")
    sections.append("-" * 72)
    sections.append(combined_importance_df.to_string(index=False))
    sections.append("")

    if overlap_df is not None:
        sections.append(f"3. Overlap-ID Analysis ({OVERLAP_MODEL})")
        sections.append("-" * 72)
        sections.append(
            f"Overlap IDs were computed on the training data with Fuzzing "
            f"messages excluded; a CAN_ID is considered an overlap ID if it "
            f"appears on two or more buses under this filter. A total of "
            f"{n_overlap_ids} overlap IDs were identified."
        )
        sections.append("")
        sections.append("3.1 Overlap Ratio by Attack Class")
        sections.append(overlap_df.to_string(index=False))
        sections.append("")

    sections.append("4. Execution Time")
    sections.append("-" * 72)
    for key, value in timings.items():
        sections.append(f"{key:45s} {value:10.2f} s")
    sections.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_observation3() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    idx = get_next_index(RESULT_DIR)
    sfx = _suffix(idx)

    timings: Dict[str, float] = {}
    t_total_start = time.perf_counter()

    # Step 1. Load data.
    t0 = time.perf_counter()
    per_bus, multi, feature_cols = load_all()
    timings["data_loading_sec"] = time.perf_counter() - t0

    # Step 2. Compute overlap IDs.
    t0 = time.perf_counter()
    per_bus_train_for_overlap: Dict[str, pd.DataFrame] = {
        bus: _load_pickle(path)[["CAN_ID", "Label"]]
        for bus, path in (
            ("B-CAN", TRAIN_PKL_B),
            ("C-CAN", TRAIN_PKL_C),
            ("P-CAN", TRAIN_PKL_P),
        )
    }
    overlap_ids = compute_overlap_ids(per_bus_train_for_overlap)
    timings["overlap_id_computation_sec"] = time.perf_counter() - t0
    print(f"\n  overlap IDs identified: {len(overlap_ids)}")

    # Step 3. Shared label encoder (required for XGBoost).
    all_labels = pd.concat([multi.y_train, multi.y_test], axis=0).astype(str).unique()
    le = LabelEncoder().fit(sorted(all_labels))

    # Step 4. Run each model family end-to-end.
    per_model: Dict[str, Dict] = {}
    for mt in MODEL_TYPES:
        per_model[mt] = _run_single_model(
            model_type=mt,
            per_bus=per_bus,
            multi=multi,
            feature_cols=feature_cols,
            overlap_ids=overlap_ids,
            label_encoder=le if mt == "XGB" else None,
            timings=timings,
        )

    # Step 5. Persist artifacts.
    combined_summary_df = pd.concat(
        [per_model[mt]["summary_df"] for mt in MODEL_TYPES], ignore_index=True
    )
    combined_importance_df = pd.concat(
        [per_model[mt]["importance_df"] for mt in MODEL_TYPES], ignore_index=True
    )

    summary_path = os.path.join(RESULT_DIR, f"summary_macro_f1{sfx}.csv")
    combined_summary_df.to_csv(summary_path, index=False)
    print(f"\n  saved -> {summary_path}")

    importance_path = os.path.join(RESULT_DIR, f"top5_feature_importance{sfx}.csv")
    combined_importance_df.to_csv(importance_path, index=False)
    print(f"  saved -> {importance_path}")

    overlap_info = per_model.get(OVERLAP_MODEL, {})
    prediction_compare = overlap_info.get("prediction_compare")
    overlap_df = overlap_info.get("overlap_df")

    if prediction_compare is not None:
        pc_path = os.path.join(RESULT_DIR,
                               f"prediction_compare_{OVERLAP_MODEL}{sfx}.csv")
        prediction_compare.to_csv(pc_path, index=False)
        print(f"  saved -> {pc_path}")

    if overlap_df is not None:
        ov_path = os.path.join(RESULT_DIR,
                               f"overlap_ratio_by_attack_{OVERLAP_MODEL}{sfx}.csv")
        overlap_df.to_csv(ov_path, index=False)
        print(f"  saved -> {ov_path}")

    timings["total_sec"] = time.perf_counter() - t_total_start
    timing_path = os.path.join(RESULT_DIR, f"execution_time{sfx}.txt")
    _write_timing_file(timing_path, timings)
    report_path = os.path.join(RESULT_DIR, f"observation3_report{sfx}.txt")
    _write_report(
        report_path,
        combined_summary_df=combined_summary_df,
        combined_importance_df=combined_importance_df,
        overlap_df=overlap_df,
        n_overlap_ids=len(overlap_ids),
        timings=timings,
    )
    print(f"  saved -> {timing_path}")
    print(f"  saved -> {report_path}")

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(combined_summary_df.to_string(index=False))
    print(f"\nTotal time: {timings['total_sec']:.2f} s")


def main() -> None:
    _start = time.time()
    run_observation3()
    elapsed = time.time() - _start
    print(f"\nTotal execution time: {elapsed:.2f} s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()