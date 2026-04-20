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
    print("=" * 80)
    print("38-FEATURE PREPROCESSING")
    print("=" * 80)
    # ── Load raw CSVs ────────────────────────────────────────────────────────
    SOURCE_PATH = os.path.join(BASE_PATH, "Autohack2025_Dataset", "Interface")
    train_data_path  = os.path.join(SOURCE_PATH, "train", "autohack_train_data_interface.csv")
    train_label_path = os.path.join(SOURCE_PATH, "train", "autohack_train_label_interface.csv")
    test_data_path   = os.path.join(SOURCE_PATH, "test",  "autohack_test_data_interface.csv")
    test_label_path  = os.path.join(SOURCE_PATH, "test",  "autohack_test_label_interface.csv")

    print("\n[1] Loading raw CSVs ...")
    t0 = datetime.now()
    train_data_df  = pd.read_csv(train_data_path,  dtype={'Arbitration_ID': str})
    train_label_df = pd.read_csv(train_label_path)
    train_df = pd.concat([train_data_df, train_label_df], axis=1)
    train_label_col = train_df.columns[-1]
    print(f"  train: {len(train_df):,} rows  (label col = '{train_label_col}')")

    test_data_df  = pd.read_csv(test_data_path,  dtype={'Arbitration_ID': str})
    test_label_df = pd.read_csv(test_label_path)
    test_df = pd.concat([test_data_df, test_label_df], axis=1)
    test_label_col = test_df.columns[-1]
    print(f"  test:  {len(test_df):,} rows  (label col = '{test_label_col}')")

    # Merge UDS_Spoofing variants into 'UDS' (observation3.py convention)
    train_df[train_label_col] = train_df[train_label_col].apply(
        lambda x: 'UDS' if 'UDS' in str(x) else x
    )
    test_df[test_label_col] = test_df[test_label_col].apply(
        lambda x: 'UDS' if 'UDS' in str(x) else x
    )

    train_df_b = train_df[train_df['Interface'] == 'B-CAN']
    train_df_c = train_df[train_df['Interface'] == 'C-CAN']
    train_df_p = train_df[train_df['Interface'] == 'P-CAN']

    test_df_b = test_df[test_df['Interface'] == 'B-CAN']
    test_df_c = test_df[test_df['Interface'] == 'C-CAN']
    test_df_p = test_df[test_df['Interface'] == 'P-CAN']

    print(f"  train labels: {sorted(train_df[train_label_col].unique())}")
    print(f"  test  labels: {sorted(test_df[test_label_col].unique())}")
    print(f"  CSV load time: {datetime.now() - t0}")

    # ── Extract features ─────────────────────────────────────────────────────
    extractor = CANIDSFeatureExtractor38(window_size='10s')

    print("\n[2] Extracting features from TRAINING data ...")
    t0 = datetime.now()
    train_features = extractor.extract(train_df)
    print(f"  train extract time: {datetime.now() - t0}")

    t0 = datetime.now()
    train_features_b = extractor.extract(train_df_b)
    print(f"  train extract time: {datetime.now() - t0}")

    t0 = datetime.now()
    train_features_c = extractor.extract(train_df_c)
    print(f"  train extract time: {datetime.now() - t0}")

    t0 = datetime.now()
    train_features_p = extractor.extract(train_df_p)
    print(f"  train extract time: {datetime.now() - t0}")

    print("\n[3] Extracting features from TEST data ...")
    t0 = datetime.now()
    test_features = extractor.extract(test_df)
    print(f"  test extract time: {datetime.now() - t0}")

    t0 = datetime.now()
    test_features_b = extractor.extract(test_df_b)
    print(f"  train extract time: {datetime.now() - t0}")

    t0 = datetime.now()
    test_features_c = extractor.extract(test_df_c)
    print(f"  train extract time: {datetime.now() - t0}")

    t0 = datetime.now()
    test_features_p = extractor.extract(test_df_p)
    print(f"  train extract time: {datetime.now() - t0}")

    # ── Build proc dataframes (features + Interface + Label) ─────────────────
    feature_columns = list(train_features.columns)

    train_proc = train_features.copy()
    train_proc['Interface'] = train_df['Interface'].values
    train_proc['Label']     = train_df[train_label_col].values

    train_proc_b = train_features_b.copy()
    train_proc_b['Interface'] = train_df_b['Interface'].values
    train_proc_b['Label']     = train_df_b[train_label_col].values

    train_proc_c = train_features_c.copy()
    train_proc_c['Interface'] = train_df_c['Interface'].values
    train_proc_c['Label']     = train_df_c[train_label_col].values

    train_proc_p = train_features_p.copy()
    train_proc_p['Interface'] = train_df_p['Interface'].values
    train_proc_p['Label']     = train_df_p[train_label_col].values

    test_proc = test_features.copy()
    test_proc['Interface'] = test_df['Interface'].values
    test_proc['Label']     = test_df[test_label_col].values

    test_proc_b = test_features_b.copy()
    test_proc_b['Interface'] = test_df_b['Interface'].values
    test_proc_b['Label']     = test_df_b[test_label_col].values

    test_proc_c = test_features_c.copy()
    test_proc_c['Interface'] = test_df_c['Interface'].values
    test_proc_c['Label']     = test_df_c[test_label_col].values

    test_proc_p = test_features_p.copy()
    test_proc_p['Interface'] = test_df_p['Interface'].values
    test_proc_p['Label']     = test_df_p[test_label_col].values

    print(f"\n[4] proc shapes:")
    print(f"  train_proc: {train_proc.shape}  ({len(feature_columns)} features + Interface + Label)")
    print(f"  test_proc:  {test_proc.shape}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    print(f"\n[5] Saving outputs to {OUTPUT_DIR} ...")

    train_csv = os.path.join(OUTPUT_DIR, "train_proc_38f.csv")
    train_csv_b = os.path.join(OUTPUT_DIR, "train_proc_b_38f.csv")
    train_csv_c = os.path.join(OUTPUT_DIR, "train_proc_c_38f.csv")
    train_csv_p = os.path.join(OUTPUT_DIR, "train_proc_p_38f.csv")
    test_csv  = os.path.join(OUTPUT_DIR, "test_proc_38f.csv")
    test_csv_b  = os.path.join(OUTPUT_DIR, "test_proc_b_38f.csv")
    test_csv_c  = os.path.join(OUTPUT_DIR, "test_proc_c_38f.csv")
    test_csv_p  = os.path.join(OUTPUT_DIR, "test_proc_p_38f.csv")
    train_pkl = os.path.join(OUTPUT_DIR, "train_proc_38f.pkl")
    train_pkl_c = os.path.join(OUTPUT_DIR, "train_proc_c_38f.pkl")
    train_pkl_b = os.path.join(OUTPUT_DIR, "train_proc_b_38f.pkl")
    train_pkl_p = os.path.join(OUTPUT_DIR, "train_proc_p_38f.pkl")
    test_pkl  = os.path.join(OUTPUT_DIR, "test_proc_38f.pkl")
    test_pkl_c  = os.path.join(OUTPUT_DIR, "test_proc_c_38f.pkl")
    test_pkl_b  = os.path.join(OUTPUT_DIR, "test_proc_b_38f.pkl")
    test_pkl_p  = os.path.join(OUTPUT_DIR, "test_proc_p_38f.pkl")
    feat_txt  = os.path.join(OUTPUT_DIR, "feature_columns.txt")

    print("  saving train CSV ...")
    train_proc.to_csv(train_csv, index=False)
    print(f"    -> {train_csv}  ({os.path.getsize(train_csv) / 1024 / 1024:.1f} MB)")
    train_proc_b.to_csv(train_csv_b, index=False)
    print(f"    -> {train_csv_b}  ({os.path.getsize(train_csv_b) / 1024 / 1024:.1f} MB)")
    train_proc_c.to_csv(train_csv_c, index=False)
    print(f"    -> {train_csv_c}  ({os.path.getsize(train_csv_c) / 1024 / 1024:.1f} MB)")
    train_proc_p.to_csv(train_csv_p, index=False)
    print(f"    -> {train_csv_p}  ({os.path.getsize(train_csv_p) / 1024 / 1024:.1f} MB)")

    print("  saving test CSV ...")
    test_proc.to_csv(test_csv, index=False)
    print(f"    -> {test_csv}  ({os.path.getsize(test_csv) / 1024 / 1024:.1f} MB)")
    test_proc_b.to_csv(test_csv_b, index=False)
    print(f"    -> {test_csv_b}  ({os.path.getsize(test_csv_b) / 1024 / 1024:.1f} MB)")
    test_proc_c.to_csv(test_csv_c, index=False)
    print(f"    -> {test_csv_c}  ({os.path.getsize(test_csv_c) / 1024 / 1024:.1f} MB)")
    test_proc_p.to_csv(test_csv_p, index=False)
    print(f"    -> {test_csv_p}  ({os.path.getsize(test_csv_p) / 1024 / 1024:.1f} MB)")

    print("  saving train pickle ...")
    with open(train_pkl, 'wb') as f:
        pickle.dump(train_proc, f)
    print(f"    -> {train_pkl}  ({os.path.getsize(train_pkl) / 1024 / 1024:.1f} MB)")
    with open(train_pkl_b, 'wb') as f:
        pickle.dump(train_proc_b, f)
    print(f"    -> {train_pkl_b}  ({os.path.getsize(train_pkl_b) / 1024 / 1024:.1f} MB)")
    with open(train_pkl_c, 'wb') as f:
        pickle.dump(train_proc_c, f)
    print(f"    -> {train_pkl_c}  ({os.path.getsize(train_pkl_c) / 1024 / 1024:.1f} MB)")
    with open(train_pkl_p, 'wb') as f:
        pickle.dump(train_proc_p, f)
    print(f"    -> {train_pkl_p}  ({os.path.getsize(train_pkl_p) / 1024 / 1024:.1f} MB)")


    print("  saving test pickle ...")
    with open(test_pkl, 'wb') as f:
        pickle.dump(test_proc, f)
    print(f"    -> {test_pkl}  ({os.path.getsize(test_pkl) / 1024 / 1024:.1f} MB)")
    with open(test_pkl_b, 'wb') as f:
        pickle.dump(test_proc_b, f)
    print(f"    -> {test_pkl_b}  ({os.path.getsize(test_pkl_b) / 1024 / 1024:.1f} MB)")
    with open(test_pkl_c, 'wb') as f:
        pickle.dump(test_proc_c, f)
    print(f"    -> {test_pkl_c}  ({os.path.getsize(test_pkl_c) / 1024 / 1024:.1f} MB)")
    with open(test_pkl_p, 'wb') as f:
        pickle.dump(test_proc_p, f)
    print(f"    -> {test_pkl_p}  ({os.path.getsize(test_pkl_p) / 1024 / 1024:.1f} MB)")

    print("  saving feature column list ...")
    with open(feat_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(feature_columns))
    print(f"    -> {feat_txt}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Feature count: {len(feature_columns)}")
    print(f"  Train rows: {len(train_proc):,}")
    print(f"  Test  rows: {len(test_proc):,}")
    print()
    print("  Next step: run train_38f.py to train/evaluate.")
    elapsed = time.time() - _start
    print(f"Total execution time: {elapsed:.2f} s ({elapsed/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()