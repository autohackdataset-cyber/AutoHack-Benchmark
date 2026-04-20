import os
import time
import pandas as pd
from tqdm import tqdm

time_size = 10
window_size = f'{time_size}s'   # Time-based rolling window
over_window = time_size + 1

def hex_to_int(x, default=0):
    try:
        x = str(x).replace(" ", "").strip()
        if x == "" or x.lower() == "nan":
            return default
        return int(x, 16)
    except Exception:
        return default

def processing(df):
    steps = 8
    pbar = tqdm(total=steps, desc="  ▶ Processing steps", unit="step")

    pbar.set_description("Step 1: Mapping labels")
    df['Bus'] = df['Interface'].map({'B-CAN': 0, 'C-CAN': 1, 'P-CAN': 2})
    df.drop('Interface', axis=1, inplace=True)

    df['Label'] = df['Label'].astype(str)
    df['Class'] = df['Label'].map({
        'Normal': 0, 'DoS': 1, 'Spoofing': 1, 'Replay': 1, 'Fuzzing': 1
    }).fillna(1).astype('int32')

    df['Label'] = df['Label'].map({
        'Normal': 0, 'DoS': 1, 'Spoofing': 2, 'Replay': 3, 'Fuzzing': 4
    }).fillna(5).astype('int32')

    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    offset = df.groupby("Timestamp").cumcount().astype("float64")
    df["Timestamp"] = df["Timestamp"] + offset * 0.000003
    pbar.update(1)

    pbar.set_description("Step 2: Converting Data")
    df['Data'] = df['Data'].fillna('00').apply(hex_to_int)
    pbar.update(1)

    pbar.set_description("Step 3: Converting Arbitration_ID")
    df['Arbitration_ID'] = df['Arbitration_ID'].apply(hex_to_int)
    pbar.update(1)

    pbar.set_description("Step 4: Timestamp to datetime")
    df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('DateTime').sort_index()
    pbar.update(1)

    pbar.set_description("Step 5: Calculating intervals")
    ts_us = (df['Timestamp']*1_000_000).round()
    df["Prev_Interval"] = ts_us.diff().fillna(over_window)/1_000_000
    df["ID_Prev_Interval"] = ts_us.groupby(df["Arbitration_ID"]).diff().fillna(over_window * 1_000_000) / 1_000_000
    df["Data_Prev_Interval"] = ts_us.groupby([df["Arbitration_ID"], df["Data"]]).diff().fillna(over_window * 1_000_000) / 1_000_000
    pbar.update(1)

    pbar.set_description("Step 6: Rolling frequencies")
    df['ID_Frequency'] = (
        df.groupby('Arbitration_ID')['Arbitration_ID']
          .rolling(window_size)
          .count()
          .reset_index(level=0, drop=True)
    )
    df['Data_Frequency'] = (
        df.groupby(['Arbitration_ID', 'Data'])['Data']
          .rolling(window_size)
          .count()
          .reset_index(level=[0, 1], drop=True)
    )
    pbar.update(1)

    pbar.set_description("Step 7: Frequency diff & cleanup")
    df['Frequency_diff'] = df['ID_Frequency'] - df['Data_Frequency']
    df = df.reset_index(drop=True)
    df = df.drop(columns=['Data'], errors='ignore')
    pbar.update(1)

    pbar.set_description("Step 8: Done")
    pbar.update(1)
    pbar.close()
    return df

def main():
    _start = time.time()
    program_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(program_path)
    
    dataset_path = os.path.join(base_path, "Autohack2025_Dataset", "Interface")
    file_map = {
        "train": os.path.join(dataset_path, "train", "autohack_train_both_interface.csv"),
        "test":  os.path.join(dataset_path, "test",  "autohack_test_both_interface.csv"),
    }

    source_path = os.path.join(program_path, "source", "AutoHack")
    os.makedirs(source_path, exist_ok=True)

    for split, file_path in file_map.items():
        print(f"\n📂 Start processing `{split}` — Rolling window: {window_size}")
        with tqdm(total=1, desc="Processing files", unit="file") as pbar:
            pbar.set_description(f"📄 {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            processed = processing(df)
            pbar.update(1)

        output_file = os.path.join(source_path, f"{split}_proc.csv")
        processed.to_csv(output_file, index=False)
        print(f"✅ Saved processed data to: {output_file}")

    elapsed = time.time() - _start
    print(f"\nTotal execution time: {elapsed:.2f} s ({elapsed/60:.1f} min)")

if __name__ == "__main__":
    main()