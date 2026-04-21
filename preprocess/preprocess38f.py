"""
preprocess_38f.py
==================
Reads raw data, extracts 38 features, and saves the proc files.
See README_preprocessing38f.md for detailed descriptions.
"""

import os
import pickle
import time
import warnings
from datetime import datetime

from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_PATH, "preprocess", "source", "AutoHack_38f")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Feature Extractor (38 features)
# =============================================================================
class CANIDSFeatureExtractor38:
    def __init__(self, window_size='10s'):
        self.window_size = window_size

    @staticmethod
    def calculate_entropy(values):
        if len(values) == 0:
            return 0.0
        vc = pd.Series(values).value_counts()
        p = vc / len(values)
        return float(-np.sum(p * np.log2(p + 1e-10)))

    @staticmethod
    def hex_to_decimal(hex_value):
        try:
            if isinstance(hex_value, str):
                return int(hex_value, 16)
            elif isinstance(hex_value, (int, np.integer)):
                return int(hex_value)
            return 0
        except Exception:
            return 0

    @staticmethod
    def _data_to_int(x):
        # Compatible with preprocessing.py: space-delimited hex to single int
        if pd.isna(x):
            return 0
        try:
            s = str(x).replace(' ', '')
            return int(s, 16) if s else 0
        except Exception:
            return 0

    def extract(self, df):
        print(f"  Processing {len(df):,} messages ...")
        df = df.copy().reset_index(drop=True)

        df['Arbitration_ID_decimal'] = df['Arbitration_ID'].apply(self.hex_to_decimal)
        df['DATA_BYTES'] = df['Data'].apply(
            lambda x: [int(b, 16) for b in str(x).strip().split()]
            if pd.notna(x) and str(x).strip() else [0] * 8
        )
        df['DATA_BYTES'] = df['DATA_BYTES'].apply(lambda x: (x + [0] * 8)[:8])
        for i in range(8):
            df[f'DATA_{i}'] = df['DATA_BYTES'].apply(lambda x: x[i])

        df['Data_int'] = df['Data'].apply(self._data_to_int)

        features = {}

        features['CAN_ID'] = df['Arbitration_ID_decimal'].values
        features['DLC']    = df['DLC'].values
        for i in range(8):
            features[f'DATA_{i}'] = df[f'DATA_{i}'].values

        data_array = np.array(df['DATA_BYTES'].tolist())

        features['MEAN']          = np.mean(data_array, axis=1)
        features['STD']           = np.std(data_array, axis=1)
        features['MIN']           = np.min(data_array, axis=1)
        features['MAX']           = np.max(data_array, axis=1)
        features['MEDIAN']        = np.median(data_array, axis=1)
        features['SKEWNESS']      = skew(data_array, axis=1)
        features['KURTOSIS']      = kurtosis(data_array, axis=1)
        features['PERCENTILE_25'] = np.percentile(data_array, 25, axis=1)
        features['PERCENTILE_75'] = np.percentile(data_array, 75, axis=1)
        features['PERCENTILE_90'] = np.percentile(data_array, 90, axis=1)
        features['MAD']           = np.mean(
            np.abs(data_array - features['MEAN'][:, np.newaxis]), axis=1
        )
        features['RMS']           = np.sqrt(np.mean(np.square(data_array), axis=1))
        features['ZERO_COUNT']    = np.sum(data_array == 0, axis=1)
        features['SUM']           = np.sum(data_array, axis=1)
        features['PRODUCT']       = np.prod(data_array + 1, axis=1)

        df['Prev_Interval'] = df['Timestamp'].diff().fillna(11).astype(float)
        df['ID_Prev_Interval'] = (
            df.groupby('Arbitration_ID_decimal')['Timestamp']
              .diff().fillna(11).astype(float)
        )
        df['Data_Prev_Interval'] = (
            df.groupby(['Arbitration_ID_decimal', 'Data_int'])['Timestamp']
              .diff().fillna(11).astype(float)
        )

        df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='s')
        df_idx = df.set_index('DateTime')

        df_idx['ID_Frequency'] = (
            df_idx.groupby('Arbitration_ID_decimal')['Arbitration_ID_decimal']
                  .transform(lambda x: x.rolling(window=self.window_size,
                                                 min_periods=1).count())
        )
        df_idx['Data_Frequency'] = (
            df_idx.groupby(['Arbitration_ID_decimal', 'Data_int'])['Data_int']
                  .transform(lambda x: x.rolling(window=self.window_size,
                                                 min_periods=1).count())
        )
        df_idx['Frequency_diff'] = df_idx['ID_Frequency'] - df_idx['Data_Frequency']

        features['Prev_Interval']      = df['Prev_Interval'].values
        features['ID_Prev_Interval']   = df['ID_Prev_Interval'].values
        features['Data_Prev_Interval'] = df['Data_Prev_Interval'].values
        features['ID_Frequency']       = df_idx['ID_Frequency'].values
        features['Data_Frequency']     = df_idx['Data_Frequency'].values
        features['Frequency_diff']     = df_idx['Frequency_diff'].values

        df_idx['ID_Prev_Interval_for_roll'] = df['ID_Prev_Interval'].values
        df_idx['IAT_MEAN'] = (
            df_idx.groupby('Arbitration_ID_decimal')['ID_Prev_Interval_for_roll']
                  .transform(lambda x: x.rolling(window=self.window_size,
                                                 min_periods=1).mean())
        )
        df_idx['IAT_STD'] = (
            df_idx.groupby('Arbitration_ID_decimal')['ID_Prev_Interval_for_roll']
                  .transform(lambda x: x.rolling(window=self.window_size,
                                                 min_periods=1).std().fillna(0))
        )

        df_idx['MEAN_for_roll'] = features['MEAN']
        df_idx['WINDOW_MEAN'] = (
            df_idx.groupby('Arbitration_ID_decimal')['MEAN_for_roll']
                  .transform(lambda x: x.rolling(window=self.window_size,
                                                 min_periods=1).mean())
        )
        df_idx['WINDOW_STD'] = (
            df_idx.groupby('Arbitration_ID_decimal')['MEAN_for_roll']
                  .transform(lambda x: x.rolling(window=self.window_size,
                                                 min_periods=1).std().fillna(0))
        )
        df_idx['WINDOW_MIN'] = (
            df_idx.groupby('Arbitration_ID_decimal')['MEAN_for_roll']
                  .transform(lambda x: x.rolling(window=self.window_size,
                                                 min_periods=1).min())
        )
        df_idx['WINDOW_MAX'] = (
            df_idx.groupby('Arbitration_ID_decimal')['MEAN_for_roll']
                  .transform(lambda x: x.rolling(window=self.window_size,
                                                 min_periods=1).max())
        )

        features['IAT_MEAN']    = df_idx['IAT_MEAN'].values
        features['IAT_STD']     = df_idx['IAT_STD'].values
        features['WINDOW_MEAN'] = df_idx['WINDOW_MEAN'].values
        features['WINDOW_STD']  = df_idx['WINDOW_STD'].values
        features['WINDOW_MIN']  = df_idx['WINDOW_MIN'].values
        features['WINDOW_MAX']  = df_idx['WINDOW_MAX'].values

        features['PAYLOAD_ENTROPY'] = np.array(
            [self.calculate_entropy(row) for row in data_array]
        )

        out = pd.DataFrame(features)
        out = out.fillna(0).astype(float)

        print(f"  ✓ Extracted {out.shape[1]} features for {len(out):,} rows")
        return out


# =============================================================================
# Main
# =============================================================================
def main():
    _start = time.time()
    SOURCE_PATH = os.path.join(BASE_PATH, "Autohack2025_Dataset", "Interface")
    train_data_path  = os.path.join(SOURCE_PATH, "train", "autohack_train_data_interface.csv")
    train_label_path = os.path.join(SOURCE_PATH, "train", "autohack_train_label_interface.csv")
    test_data_path   = os.path.join(SOURCE_PATH, "test",  "autohack_test_data_interface.csv")
    test_label_path  = os.path.join(SOURCE_PATH, "test",  "autohack_test_label_interface.csv")

    # ── Load CSVs ────────────────────────────────────────────────────────────
    load_files = [
        ("📄 train data",  train_data_path),
        ("📄 train label", train_label_path),
        ("📄 test data",   test_data_path),
        ("📄 test label",  test_label_path),
    ]
    print(f"\n📂 Loading raw CSVs — {SOURCE_PATH}")
    with tqdm(total=len(load_files), desc="  ▶ Loading", unit="file") as pbar:
        pbar.set_description(load_files[0][0])
        train_data_df = pd.read_csv(train_data_path, dtype={'Arbitration_ID': str})
        pbar.update(1)
        pbar.set_description(load_files[1][0])
        train_label_df = pd.read_csv(train_label_path)
        pbar.update(1)
        pbar.set_description(load_files[2][0])
        test_data_df = pd.read_csv(test_data_path, dtype={'Arbitration_ID': str})
        pbar.update(1)
        pbar.set_description(load_files[3][0])
        test_label_df = pd.read_csv(test_label_path)
        pbar.update(1)

    train_df = pd.concat([train_data_df, train_label_df], axis=1)
    test_df  = pd.concat([test_data_df,  test_label_df],  axis=1)
    train_label_col = train_df.columns[-1]
    test_label_col  = test_df.columns[-1]
    print(f"  train: {len(train_df):,} rows  |  test: {len(test_df):,} rows")

    train_df[train_label_col] = train_df[train_label_col].apply(
        lambda x: 'UDS' if 'UDS' in str(x) else x
    )
    test_df[test_label_col] = test_df[test_label_col].apply(
        lambda x: 'UDS' if 'UDS' in str(x) else x
    )

    # ── Extract features ─────────────────────────────────────────────────────
    extractor = CANIDSFeatureExtractor38(window_size='10s')
    subsets = [
        ("train (all)",   train_df[train_df['Interface'].notna()],        "train",   train_df,   train_label_col),
        ("train (B-CAN)", train_df[train_df['Interface'] == 'B-CAN'],     "train_b", train_df[train_df['Interface'] == 'B-CAN'], train_label_col),
        ("train (C-CAN)", train_df[train_df['Interface'] == 'C-CAN'],     "train_c", train_df[train_df['Interface'] == 'C-CAN'], train_label_col),
        ("train (P-CAN)", train_df[train_df['Interface'] == 'P-CAN'],     "train_p", train_df[train_df['Interface'] == 'P-CAN'], train_label_col),
        ("test  (all)",   test_df[test_df['Interface'].notna()],          "test",    test_df,    test_label_col),
        ("test  (B-CAN)", test_df[test_df['Interface'] == 'B-CAN'],       "test_b",  test_df[test_df['Interface'] == 'B-CAN'],  test_label_col),
        ("test  (C-CAN)", test_df[test_df['Interface'] == 'C-CAN'],       "test_c",  test_df[test_df['Interface'] == 'C-CAN'],  test_label_col),
        ("test  (P-CAN)", test_df[test_df['Interface'] == 'P-CAN'],       "test_p",  test_df[test_df['Interface'] == 'P-CAN'],  test_label_col),
    ]

    print(f"\n📂 Extracting 38 features — {len(subsets)} subsets")
    proc = {}
    with tqdm(total=len(subsets), desc="  ▶ Extracting", unit="subset") as pbar:
        for label, df_sub, key, df_orig, lbl_col in subsets:
            pbar.set_description(f"  ▶ {label}")
            feat = extractor.extract(df_sub)
            feat['Interface'] = df_orig['Interface'].values
            feat['Label']     = df_orig[lbl_col].values
            proc[key] = feat
            pbar.update(1)

    feature_columns = [c for c in proc['train'].columns if c not in ('Interface', 'Label')]

    # ── Save outputs ─────────────────────────────────────────────────────────
    save_pairs = [
        ("train_proc_38f",   proc['train']),
        ("train_proc_b_38f", proc['train_b']),
        ("train_proc_c_38f", proc['train_c']),
        ("train_proc_p_38f", proc['train_p']),
        ("test_proc_38f",    proc['test']),
        ("test_proc_b_38f",  proc['test_b']),
        ("test_proc_c_38f",  proc['test_c']),
        ("test_proc_p_38f",  proc['test_p']),
    ]

    print(f"\n📂 Saving outputs — {OUTPUT_DIR}")
    with tqdm(total=len(save_pairs) * 2 + 1, desc="  ▶ Saving", unit="file") as pbar:
        for name, df_out in save_pairs:
            csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
            pbar.set_description(f"  ▶ {name}.csv")
            df_out.to_csv(csv_path, index=False)
            pbar.update(1)
            pkl_path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
            pbar.set_description(f"  ▶ {name}.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(df_out, f)
            pbar.update(1)

        feat_txt = os.path.join(OUTPUT_DIR, "feature_columns.txt")
        pbar.set_description("  ▶ feature_columns.txt")
        with open(feat_txt, 'w', encoding='utf-8') as f:
            f.write("\n".join(feature_columns))
        pbar.update(1)

    print(f"✅ Saved {len(save_pairs) * 2 + 1} files to: {OUTPUT_DIR}")
    elapsed = time.time() - _start
    print(f"\nTotal execution time: {elapsed:.2f} s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()