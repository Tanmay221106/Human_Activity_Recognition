# =============================================================================
# MPU6050 ACTIVITY RECOGNITION — RANDOM FOREST CLASSIFIER
# Pipeline: Parse -> Clean -> Filter -> Magnitude -> Window -> Feature Extract
#           -> Label -> Combine -> Train Random Forest -> Evaluate -> Predict
#
# WHAT CHANGED FROM YOUR ORIGINAL SCRIPT (read this first):
#
# 1) Your 4 activity files (sitting.csv, standing.csv, walking.csv, running.csv)
#    are NOT plain CSVs. They are raw serial-monitor lines like:
#       10:01:21.769 ax=1.016846<TAB>ay=-0.152344<TAB>az=-0.318115...
#    So Step 1 below is a custom parser (regex-based), not pd.read_csv().
#    'cleaned_sensor_data.csv' is a properly-parsed version of walking.csv
#    (same 110 rows) — it's not a 5th activity, so it is NOT used for
#    training. It's left in the folder purely as a reference of what a
#    "parsed" file should look like.
#
# 2) I REMOVED the StandardScaler normalization step (old Step 6).
#    Reason: StandardScaler was being fit SEPARATELY on each activity file.
#    That forces every file's ax/ay/az/... to have mean=0, std=1 -
#    which erases the exact differences (e.g. "running shakes a lot more
#    than sitting") that the classifier needs to tell activities apart.
#    Random Forest also does not need scaled features at all (unlike
#    KNN/SVM/Neural Nets), so skipping this step is strictly better here.
#
# 3) Your sensors are logging at ~2 Hz (one reading every ~0.5s), not the
#    50Hz assumed in your original comments. So window_size=100 was far
#    too large (each file only has 70-118 rows total!). I used a much
#    smaller window (WINDOW_SIZE=10 rows = ~5 seconds) so each file still
#    produces a reasonable number of windows. See the "IMPORTANT CAVEATS"
#    note at the bottom of this file about what this means for accuracy.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Install Required Libraries (run this once in your terminal, not here)
# ─────────────────────────────────────────────────────────────────────────────
# pip install pandas numpy scikit-learn scipy matplotlib seaborn

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Import Libraries
# ─────────────────────────────────────────────────────────────────────────────
import re                                   # for parsing the raw "ax=... ay=..." lines
import pandas as pd                         # dataframes
import numpy as np                          # numeric arrays / math
from scipy.signal import butter, filtfilt   # low-pass filter (denoising)
from scipy.fft import fft                   # frequency-domain features
from sklearn.ensemble import RandomForestClassifier   # our classifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: File Locations
#
# Change DATA_DIR to wherever sitting.csv / standing.csv / walking.csv /
# running.csv live on your machine (this is the folder you were already
# using for cleaned_sensor_data.csv).
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "D:/Software/PersonalWork/MozekHealthtech/adl/"

# Map: activity name -> raw file for that activity.
# Each file = one continuous recording session of ONE activity, exactly
# like your original Step 10 assumed.
ACTIVITY_FILES = {
    'sitting':  DATA_DIR + 'sitting.csv',
    'standing': DATA_DIR + 'standing.csv',
    'walking':  DATA_DIR + 'walking.csv',
    'running':  DATA_DIR + 'running.csv',
}

SENSOR_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']   # raw IMU columns
FEATURE_COLS = SENSOR_COLS + ['acc_mag', 'gyro_mag']  # + the 2 magnitude columns


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Parse a Raw Sensor File into a Clean DataFrame
#
# Your files look like this per line (tab-separated key=value pairs, with a
# leading timestamp, and Windows line endings \r\n):
#     10:01:21.769 ax=1.016846    ay=-0.152344    az=-0.318115 ...
#
# This function reads the file line-by-line and pulls out the timestamp and
# the 6 sensor values using a regular expression.
# ─────────────────────────────────────────────────────────────────────────────
def parse_raw_sensor_file(filepath):
    rows = []
    # 'utf-8-sig' automatically strips the invisible BOM character (﻿) that
    # shows up at the start of these files.
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()          # remove \r\n and stray whitespace
            if not line:
                continue                 # skip blank lines

            # Split "TIMESTAMP  restofline" into two pieces on the first
            # run of whitespace.
            match = re.match(r'^(\S+)\s+(.*)$', line)
            if not match:
                continue                 # line didn't match expected shape -> skip
            time_str, sensor_str = match.groups()

            # Find every "key=value" pair for our 6 known sensor keys.
            # This works regardless of whether they're separated by tabs
            # or spaces.
            kv_pairs = re.findall(r'(ax|ay|az|gx|gy|gz)=(-?\d+\.?\d*)', sensor_str)

            row = {'time': time_str}
            for key, val in kv_pairs:
                row[key] = float(val)

            # Only keep the row if all 6 sensor values were found — this
            # is our "cleaning" for malformed/incomplete lines.
            if all(k in row for k in SENSOR_COLS):
                rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Low-Pass Filter (Denoising)
#
# Same idea as your original script: a Butterworth low-pass filter smooths
# out fast random spikes while keeping the real, slower motion signal.
# filtfilt() runs it forward+backward so there's no time-shift.
# ─────────────────────────────────────────────────────────────────────────────
def lowpass_filter(data, cutoff=0.3, order=4):
    padlen = 3 * order              # filtfilt needs at least this many samples
    if len(data) <= padlen:
        return data                 # too short to filter safely -> leave as-is
    b, a = butter(order, cutoff, btype='low')
    return filtfilt(b, a, data)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Feature Extraction for One Window
#
# For each of the 8 columns (ax, ay, az, gx, gy, gz, acc_mag, gyro_mag) we
# compute 10 features -> 80 features total per window.
#   Time domain : mean, std, var, min, max, rms, median
#   Freq domain : dominant frequency, spectral energy, spectral entropy
# (Exactly the same feature set/formulas as your original Step 9.)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(window):
    features = {}
    for col in FEATURE_COLS:
        signal = window[col].values

        # ---- time domain ----
        features[f'{col}_mean']   = np.mean(signal)
        features[f'{col}_std']    = np.std(signal)
        features[f'{col}_var']    = np.var(signal)
        features[f'{col}_min']    = np.min(signal)
        features[f'{col}_max']    = np.max(signal)
        features[f'{col}_rms']    = np.sqrt(np.mean(signal ** 2))
        features[f'{col}_median'] = np.median(signal)

        # ---- frequency domain (FFT) ----
        fft_vals = np.abs(fft(signal))       # magnitude of each frequency component
        fft_vals = fft_vals[:len(fft_vals) // 2]   # FFT output is mirrored -> keep first half
        if len(fft_vals) == 0:
            fft_vals = np.array([0.0])       # safety net for tiny windows

        features[f'{col}_dom_freq'] = np.argmax(fft_vals)          # strongest frequency bin
        features[f'{col}_spec_energy'] = np.sum(fft_vals ** 2)     # total signal energy
        fft_norm = fft_vals / (np.sum(fft_vals) + 1e-10)           # normalize like probabilities
        features[f'{col}_spec_entropy'] = -np.sum(
            fft_norm * np.log(fft_norm + 1e-10)
        )
    return features


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Process ONE Activity File End-to-End
#   parse -> clean -> filter -> magnitude -> window -> extract features -> label
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 10   # rows per window  (~5 seconds, since sensor logs ~1 sample/0.5s)
STEP_SIZE   = 5    # rows to slide forward each time (50% overlap between windows)

def process_activity_file(filepath, activity_name):
    # 4a. Parse the raw serial-log file into a proper DataFrame
    df = parse_raw_sensor_file(filepath)

    # 4b. Clean: drop any incomplete rows (shouldn't be many, parser already
    #     filters malformed lines, but this also catches any leftover NaNs)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 4c. Denoise each of the 6 raw columns
    for col in SENSOR_COLS:
        df[col] = lowpass_filter(df[col].values)

    # 4d. Magnitude features — orientation-independent "total motion"
    df['acc_mag']  = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)

    # 4e. Slide a window across the file and extract features from each one
    feature_rows = []
    n = len(df)
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = df[FEATURE_COLS].iloc[start:end]
        feats = extract_features(window)
        feats['activity'] = activity_name    # label every window from this file
        feature_rows.append(feats)

    print(f"  {activity_name:9s}: {n:3d} raw rows -> {len(feature_rows):2d} windows")
    return feature_rows


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Build the Full Labeled Feature Dataset (all 4 activities combined)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 8: Parsing + windowing + feature extraction per activity")
print("=" * 60)

all_feature_rows = []
for activity_name, filepath in ACTIVITY_FILES.items():
    all_feature_rows.extend(process_activity_file(filepath, activity_name))

feature_df = pd.DataFrame(all_feature_rows)
print(f"\nTotal windows across all activities: {len(feature_df)}")
print(feature_df['activity'].value_counts())

# Save the combined, labeled feature dataset — this is your ML-ready file.
FEATURE_OUTPUT = DATA_DIR + "all_activities_features.csv"
feature_df.to_csv(FEATURE_OUTPUT, index=False)
print(f"\nSaved combined feature dataset to: {FEATURE_OUTPUT}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Train / Test Split
#
# NOTE ON THE SPLIT: because windows OVERLAP (50% overlap), a random
# train/test split can leak nearly-identical windows into both sets and
# make accuracy look artificially perfect. Instead, for each activity we
# take the FIRST 75% of its windows (in time order) for training and the
# LAST 25% for testing. This is a more honest test of "can the model
# generalize to a later part of the same recording it hasn't seen."
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Splitting into train/test (time-ordered per activity)")
print("=" * 60)

train_parts, test_parts = [], []
for activity_name in ACTIVITY_FILES:
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: Train the Random Forest Classifier
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Training Random Forest")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200,     # number of decision trees in the forest
    max_depth=None,       # let trees grow fully (dataset is small)
    random_state=42,      # reproducible results
    class_weight='balanced'  # compensate for uneven window counts per activity
)
rf.fit(X_train, y_train)
print("Model trained.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: Evaluate
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: Evaluation")
print("=" * 60)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Confusion matrix — shows which activities get confused with which
labels_order = list(ACTIVITY_FILES.keys())
cm = confusion_matrix(y_test, y_pred, labels=labels_order)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_order, yticklabels=labels_order)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Activity Recognition")
plt.tight_layout()
plt.savefig(DATA_DIR + "confusion_matrix.png", dpi=150)
plt.show()
print(f"\nSaved confusion matrix plot to: {DATA_DIR}confusion_matrix.png")

# Feature importance — which features matter most to the model
importance = pd.Series(rf.feature_importances_, index=X_train.columns)
top_features = importance.sort_values(ascending=False).head(15)
print("\nTop 15 most important features:")
print(top_features.round(4).to_string())

plt.figure(figsize=(8, 6))
top_features.sort_values().plot(kind='barh', color='steelblue')
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(DATA_DIR + "feature_importance.png", dpi=150)
plt.show()
print(f"Saved feature importance plot to: {DATA_DIR}feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 12: Predict a NEW, Unlabeled Recording
#
# IMPORTANT: because the model is trained on WINDOW FEATURES (80 numbers
# summarizing 10 rows), you can't feed it a single raw [ax, ay, az, gx, gy,
# gz] reading like in a generic guide. You need a short window (>=10 rows)
# of new raw sensor readings, run it through the SAME pipeline (filter ->
# magnitude -> extract_features), and predict on that.
# ─────────────────────────────────────────────────────────────────────────────
def predict_activity_from_new_window(raw_window_df):
    """
    raw_window_df: a DataFrame with at least WINDOW_SIZE rows and columns
                   ['ax','ay','az','gx','gy','gz'] of NEW, unlabeled sensor data.
    Returns: predicted activity name (string).
    """
    df = raw_window_df.copy()
    for col in SENSOR_COLS:
        df[col] = lowpass_filter(df[col].values)
    df['acc_mag']  = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)

    window = df[FEATURE_COLS].iloc[-WINDOW_SIZE:]   # use the most recent WINDOW_SIZE rows
    feats = extract_features(window)
    feats_df = pd.DataFrame([feats])[X_train.columns]  # keep same column order as training
    return rf.predict(feats_df)[0]

# Example usage (uncomment once you have a new file to test):
# new_df = parse_raw_sensor_file(DATA_DIR + "new_recording.csv")
# print("Predicted activity:", predict_activity_from_new_window(new_df))

print("\nAll steps complete. Model is trained and ready to use.")

# =============================================================================
# IMPORTANT CAVEATS FOR YOUR CAPSTONE WRITE-UP
# =============================================================================
# 1. SAMPLING RATE: your logger records ~1 sample every 0.5s (~2 Hz), not the
#    20-50 Hz typically used for gait/activity recognition. At 2 Hz, fast
#    dynamics of walking/running strides are under-sampled, so the FFT
#    features are far less informative than they'd be with a faster logger.
#    If your hardware/firmware can log faster, that will meaningfully
#    improve real-world accuracy.
#
# 2. DATASET SIZE: each activity is a SINGLE ~35-60 second recording (70-118
#    raw rows), giving only 13-22 windows per activity (76 windows total
#    across 4 classes). This is enough to demonstrate the pipeline but is
#    too small to draw strong conclusions about real-world accuracy —
#    record multiple separate sessions per activity (different times/people
#    if possible) before reporting a final number.
#
# 3. ACCURACY NUMBERS: with such a small dataset, don't be surprised if
#    accuracy looks very high (even ~100%) — that likely reflects the
#    limited variety in the data rather than a production-ready model.
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# STEP 13: Test the Model on a CSV File You Choose (interactive)
#
# Run the script, and when prompted, type/paste the path to any sensor CSV
# file (same "time ax=... ay=..." raw format as sitting/standing/walking/
# running.csv, OR an already-parsed CSV like cleaned_sensor_data.csv — this
# step auto-detects which format it is).
#
# Instead of just listing window numbers, this reports the PREDICTED
# ACTIVITY together with the actual TIME SPAN (from the file's own
# timestamps) it applies to, merging consecutive windows that got the same
# prediction into one readable block, e.g.:
#
#   12:34:12.710 -> 12:34:22.735   running   (3 windows)
#   12:34:22.735 -> 12:34:30.100   walking   (2 windows)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 13: Test the model on a CSV file")
print("=" * 60)

def load_any_sensor_csv(filepath):
    """
    Auto-detects the file format:
      - Raw serial-log format ("time ax=... ay=...")  -> use our regex parser
      - Already-parsed plain CSV (columns: time,ax,ay,az,gx,gy,gz) -> pd.read_csv
    Returns a DataFrame with columns: time, ax, ay, az, gx, gy, gz
    """
    df = parse_raw_sensor_file(filepath)
    if len(df) > 0:
        return df   # raw serial format worked

    # Fall back to a normal, already-parsed CSV (e.g. cleaned_sensor_data.csv)
    df = pd.read_csv(filepath)
    missing = [c for c in ['time'] + SENSOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Couldn't parse this file. Missing expected columns: {missing}"
        )
    return df[['time'] + SENSOR_COLS]


def recognize_activity_with_timespan(filepath):
    """
    Runs the full pipeline on one file, predicts an activity for every
    window, and merges consecutive same-activity windows into readable
    time-span blocks using the file's own 'time' column.
    """
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

    # Predict activity for every window, remembering each window's start/end time
    window_predictions = []
    n = len(df)
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = df[FEATURE_COLS].iloc[start:end]
        feats = extract_features(window)
        feats_df = pd.DataFrame([feats])[X_train.columns]  # match training column order
        pred = rf.predict(feats_df)[0]
        window_predictions.append({
            'start_time': df['time'].iloc[start],
            'end_time':   df['time'].iloc[end - 1],
            'activity':   pred,
        })

    # Merge consecutive windows that predicted the SAME activity into one
    # continuous time-span block, so the output reads like a timeline
    # instead of a long list of individual windows.
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

    # Overall verdict = majority vote across all windows in the file
    overall = pd.Series([wp['activity'] for wp in window_predictions]).mode()[0]
    print(f"\n>>> Overall activity detected in this file: {overall} <<<")


# ── Ask the user which file to test ──
user_file_path = input(
    "\nEnter the path to the CSV file you want to test "
    "(e.g. D:/Software/PersonalWork/MozekHealthtech/adl/new_recording.csv): "
).strip().strip('"')

try:
    recognize_activity_with_timespan(user_file_path)
except FileNotFoundError:
    print(f"\nCould not find the file: {user_file_path}")
except Exception as e:
    print(f"\nSomething went wrong while processing this file: {e}")