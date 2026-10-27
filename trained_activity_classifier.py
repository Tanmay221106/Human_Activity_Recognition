# =============================================================================
# MPU6050 ACTIVITY RECOGNITION — RANDOM FOREST CLASSIFIER (v2 — RETRAINED & ENHANCED)
# Pipeline: Parse -> Clean -> Filter -> Magnitude -> Window -> Feature Extract
#           -> Label -> Combine -> Train Random Forest -> Evaluate -> Predict
#
# WHAT CHANGED IN THIS REVISED VERSION:
# 1) SMART DATA PATHS: Automatically resolves paths using the current script's
#    directory, avoiding hardcoded drive letter errors on different machines.
# 2) HYBRID PARSER: Seamlessly parses both raw serial lines (ax=... ay=...) and
#    standard CSV header files (time,ax,ay,az,gx,gy,gz).
# 3) FEATURE CACHING: If 'all_activities_features.csv' exists, it loads instantly
#    instead of recomputing for ~1 minute. Set FORCE_REEXTRACT = True to recompute.
# 4) MODEL PERSISTENCE: Exports the trained model to 'har_rf_model.joblib' for
#    instant reuse by the frontend UI and inference scripts.
# 5) DC COMPONENT FIX: Argmax dominant frequency calculation now ignores bin 0 (DC)
#    to properly capture true biomechanical oscillation frequencies.
# 6) SAFE EXECUTION & MODULARITY: Wrapped in 'if __name__ == "__main__":' so functions
#    can be imported cleanly by other modules without blocking on interactive prompts.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Required Libraries
# ─────────────────────────────────────────────────────────────────────────────
# pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Import Libraries
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import sys
import pandas as pd
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: File Locations & Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR + os.sep

FORCE_REEXTRACT = False  # Set to True if you want to re-parse raw recordings from scratch

ACTIVITY_FILES = {
    'sitting':  os.path.join(DATA_DIR, 'sitting.csv'),
    'sleeping': os.path.join(DATA_DIR, 'sleeping.csv'),
    'standing': os.path.join(DATA_DIR, 'standing.csv'),
    'walking':  os.path.join(DATA_DIR, 'walking.csv'),
    # 'running': os.path.join(DATA_DIR, 'running.csv'),
}

ACTIVITY_LABELS = {
    'walking':    0,
    'running':    1,
    'sitting':    2,
    'standing':   3,
    'upstairs':   4,
    'downstairs': 5,
    'sleeping':   6,
}

SENSOR_COLS  = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
FEATURE_COLS = SENSOR_COLS + ['acc_mag', 'gyro_mag']
WINDOW_SIZE  = 20   # rows per window (~10 seconds at ~2Hz)
STEP_SIZE    = 10   # 50% overlap


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Parse Raw Sensor File or Standard CSV into a Clean DataFrame
# ─────────────────────────────────────────────────────────────────────────────
def parse_raw_sensor_file(filepath):
    """
    Parses either:
    1) Standard CSV files with header ['time', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    2) Raw serial lines with 'ax=... ay=...' key-value pairs
    """
    if not os.path.exists(filepath):
        return pd.DataFrame()

    # Try standard CSV parsing first
    try:
        df_csv = pd.read_csv(filepath)
        cols_lower = [str(c).strip().lower() for c in df_csv.columns]
        if all(c in cols_lower for c in ['time'] + SENSOR_COLS):
            col_map = {orig: lower for orig, lower in zip(df_csv.columns, cols_lower)}
            df_csv = df_csv.rename(columns=col_map)
            return df_csv[['time'] + SENSOR_COLS]
    except Exception:
        pass

    # Fallback to key=value regex parsing
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = re.match(r'^(\S+)\s+(.*)$', line)
            if not match:
                continue
            time_str, sensor_str = match.groups()

            kv_pairs = re.findall(r'(ax|ay|az|gx|gy|gz)=(-?\d+\.?\d*)', sensor_str)
            row = {'time': time_str}
            for key, val in kv_pairs:
                try:
                    row[key] = float(val)
                except ValueError:
                    pass

            if all(k in row for k in SENSOR_COLS):
                rows.append(row)

    return pd.DataFrame(rows)


def load_any_sensor_csv(filepath):
    """Wrapper that reliably parses any sensor file or raises an informative error."""
    df = parse_raw_sensor_file(filepath)
    if len(df) == 0:
        raise ValueError(
            f"Couldn't parse file '{filepath}'. Ensure it has valid time, ax, ay, az, gx, gy, gz values."
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Low-Pass Filter (Denoising)
# ─────────────────────────────────────────────────────────────────────────────
def lowpass_filter(data, cutoff=0.3, order=4):
    padlen = 3 * order
    if len(data) <= padlen:
        return data
    b, a = butter(order, cutoff, btype='low')
    return filtfilt(b, a, data)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Feature Extraction for One Window (8 columns x 10 features = 80 total)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(window):
    features = {}
    for col in FEATURE_COLS:
        signal = window[col].values

        features[f'{col}_mean']   = float(np.mean(signal))
        features[f'{col}_std']    = float(np.std(signal))
        features[f'{col}_var']    = float(np.var(signal))
        features[f'{col}_min']    = float(np.min(signal))
        features[f'{col}_max']    = float(np.max(signal))
        features[f'{col}_rms']    = float(np.sqrt(np.mean(signal ** 2)))
        features[f'{col}_median'] = float(np.median(signal))

        # Frequency domain features via FFT
        fft_vals = np.abs(fft(signal))
        fft_vals = fft_vals[:len(fft_vals) // 2]
        if len(fft_vals) == 0:
            fft_vals = np.array([0.0])

        # Ignore bin 0 (DC offset) to find true oscillating dominant frequency
        if len(fft_vals) > 1:
            dom_freq_bin = int(np.argmax(fft_vals[1:])) + 1
        else:
            dom_freq_bin = int(np.argmax(fft_vals))

        features[f'{col}_dom_freq'] = dom_freq_bin
        features[f'{col}_spec_energy'] = float(np.sum(fft_vals ** 2))
        fft_norm = fft_vals / (np.sum(fft_vals) + 1e-10)
        features[f'{col}_spec_entropy'] = float(-np.sum(
            fft_norm * np.log(fft_norm + 1e-10)
        ))
    return features


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Process ONE Activity File End-to-End
# ─────────────────────────────────────────────────────────────────────────────
def process_activity_file(filepath, activity_name):
    df = parse_raw_sensor_file(filepath)
    if df.empty:
        print(f"  Warning: No data parsed for {activity_name} from {filepath}")
        return []

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in SENSOR_COLS:
        df[col] = lowpass_filter(df[col].values)

    df['acc_mag']  = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)

    feature_rows = []
    n = len(df)
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = df[FEATURE_COLS].iloc[start:end]
        feats = extract_features(window)
        feats['activity'] = activity_name
        feature_rows.append(feats)

    print(f"  {activity_name:9s}: {n:5d} raw rows -> {len(feature_rows):4d} windows")
    return feature_rows


# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 & 13 Helper: Predict New Window / Whole Recording
# ─────────────────────────────────────────────────────────────────────────────
def predict_activity_from_new_window(raw_window_df, model, feature_columns):
    df = raw_window_df.copy()
    for col in SENSOR_COLS:
        df[col] = lowpass_filter(df[col].values)
    df['acc_mag']  = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)

    window = df[FEATURE_COLS].iloc[-WINDOW_SIZE:]
    feats = extract_features(window)
    feats_df = pd.DataFrame([feats])[feature_columns]
    return model.predict(feats_df)[0]


def recognize_activity_with_timespan(filepath, model, feature_columns):
    df = load_any_sensor_csv(filepath)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < WINDOW_SIZE:
        print(f"\nThis file only has {len(df)} rows — need at least "
              f"{WINDOW_SIZE} rows to form one window. Can't predict.")
        return

    for col in SENSOR_COLS:
        df[col] = lowpass_filter(df[col].values)
    df['acc_mag']  = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)

    window_predictions = []
    n = len(df)
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = df[FEATURE_COLS].iloc[start:end]
        feats = extract_features(window)
        feats_df = pd.DataFrame([feats])[feature_columns]
        pred = model.predict(feats_df)[0]
        window_predictions.append({
            'start_time': df['time'].iloc[start],
            'end_time':   df['time'].iloc[end - 1],
            'activity':   pred,
        })

    if not window_predictions:
        print("\nNo windows could be evaluated.")
        return

    segments = []
    for wp in window_predictions:
        if segments and segments[-1]['activity'] == wp['activity']:
            segments[-1]['end_time'] = wp['end_time']
            segments[-1]['window_count'] += 1
        else:
            segments.append({
                'start_time': wp['start_time'],
                'end_time':   wp['end_time'],
                'activity':   wp['activity'],
                'window_count': 1,
            })

    print(f"\nFile: {filepath}")
    print(f"Total windows analyzed: {len(window_predictions)}\n")
    print(f"{'Start Time':<16} {'End Time':<16} {'Activity':<12} Windows")
    print("-" * 60)
    for seg in segments:
        print(f"{seg['start_time']:<16} {seg['end_time']:<16} "
              f"{seg['activity']:<12} {seg['window_count']}")

    modes = pd.Series([wp['activity'] for wp in window_predictions]).mode()
    overall = modes[0] if not modes.empty else "Unknown"
    print(f"\n>>> Overall activity detected in this file: {overall} <<<")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    FEATURE_OUTPUT = os.path.join(DATA_DIR, "all_activities_features.csv")
    MODEL_OUTPUT   = os.path.join(DATA_DIR, "har_rf_model.joblib")

    print("=" * 60)
    print("STEP 8: Feature Dataset Preparation")
    print("=" * 60)

    # Use cached feature dataset if available, otherwise extract from raw sessions
    if os.path.exists(FEATURE_OUTPUT) and not FORCE_REEXTRACT:
        print(f"Loading existing feature dataset from: {FEATURE_OUTPUT}")
        feature_df = pd.read_csv(FEATURE_OUTPUT)
        print(f"Loaded {len(feature_df)} windows across activities.")
    else:
        print("Parsing + windowing + feature extraction per activity...")
        all_feature_rows = []
        for activity_name, filepath in ACTIVITY_FILES.items():
            if os.path.exists(filepath):
                all_feature_rows.extend(process_activity_file(filepath, activity_name))
            else:
                print(f"Warning: File not found: {filepath}")

        feature_df = pd.DataFrame(all_feature_rows)
        feature_df.to_csv(FEATURE_OUTPUT, index=False)
        print(f"\nSaved combined feature dataset to: {FEATURE_OUTPUT}")

    print("\nClass distribution:")
    print(feature_df['activity'].value_counts())

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9: Train / Test Split (time-ordered per activity)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9: Splitting into train/test (time-ordered per activity)")
    print("=" * 60)

    train_parts, test_parts = [], []
    for activity_name in feature_df['activity'].unique():
        subset = feature_df[feature_df['activity'] == activity_name]
        split_idx = int(len(subset) * 0.75)
        train_parts.append(subset.iloc[:split_idx])
        test_parts.append(subset.iloc[split_idx:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df  = pd.concat(test_parts).reset_index(drop=True)

    X_train = train_df.drop(columns=['activity'])
    y_train = train_df['activity']
    X_test  = test_df.drop(columns=['activity'])
    y_test  = test_df['activity']

    print(f"Train windows: {len(X_train)}   Test windows: {len(X_test)}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 10: Train or Load Random Forest Classifier
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 10: Model Training & Persistence")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODEL_OUTPUT)
    print(f"Model trained and saved to: {MODEL_OUTPUT}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 11: Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 11: Evaluation")
    print("=" * 60)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    labels_order = sorted(list(feature_df['activity'].unique()))
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    # Save Confusion Matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_order, yticklabels=labels_order)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Activity Recognition")
    plt.tight_layout()
    cm_path = os.path.join(DATA_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # Save Feature Importance plot
    importance = pd.Series(rf.feature_importances_, index=X_train.columns)
    top_features = importance.sort_values(ascending=False).head(15)
    print("\nTop 15 most important features:")
    print(top_features.round(4).to_string())

    plt.figure(figsize=(8, 6))
    top_features.sort_values().plot(kind='barh', color='steelblue')
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    fi_path = os.path.join(DATA_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to: {fi_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 13: Interactive Testing
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 13: Test the Model on a CSV File")
    print("=" * 60)

    try:
        user_file_path = input(
            "\nEnter path to a CSV file to test (or press Enter to skip / test cleaned_sensor_data.csv): "
        ).strip().strip('"')

        if not user_file_path:
            sample_test = os.path.join(DATA_DIR, "cleaned_sensor_data.csv")
            if os.path.exists(sample_test):
                print(f"Defaulting to sample file: {sample_test}")
                user_file_path = sample_test
            else:
                print("No input provided. Exiting.")
                return

        recognize_activity_with_timespan(user_file_path, rf, X_train.columns)
    except (EOFError, KeyboardInterrupt):
        print("\nSession completed.")
    except Exception as e:
        print(f"\nError processing file: {e}")


if __name__ == '__main__':
    main()
