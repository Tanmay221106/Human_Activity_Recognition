# =============================================================================
# MPU6050 ACTIVITY RECOGNITION — RANDOM FOREST CLASSIFIER (v2 — RETRAINED)
# Pipeline: Parse -> Clean -> Filter -> Magnitude -> Window -> Feature Extract
#           -> Label -> Combine -> Train Random Forest -> Evaluate -> Predict
#
# WHAT CHANGED FROM YOUR PREVIOUS SCRIPT (read this first):
#
# 1) OLD DATA REMOVED. The previous sitting/standing/walking/running.csv
#    files were each only ~1 minute long (70-118 rows). Those are no longer
#    used anywhere in this script.
#
# 2) NEW DATA IN. You now have 4 large recordings (SITTTTTTTTING.csv,
#    SLEEEEEEEEING.csv, STANDDDDDDING.csv, WALKKKKKKKKKKKING.csv), each
#    ~7,000-9,000 rows (roughly an hour per activity at the same ~2Hz
#    logging rate). This is a MUCH better amount of data to train on.
#
# 3) NO 'running' DATA THIS ROUND. Your new batch swapped 'running' for a
#    new activity, 'sleeping'. I added 'sleeping' to the label map and
#    dropped 'running' from training (no data for it right now). 'running'
#    is still listed in ACTIVITY_LABELS in case you record it later —
#    just add it back to ACTIVITY_FILES below when you do.
#
# 4) WINDOW_SIZE increased from 10 -> 20 rows (~10 seconds instead of ~5).
#    With only ~100 rows per file before, a small window was necessary just
#    to get a handful of windows out of each file. Now that each file has
#    thousands of rows, a slightly larger window gives smoother, more
#    reliable features without sacrificing the number of windows (you'll
#    still get hundreds per activity).
#
# 5) Parser now also handles COMMA-separated "key=value" pairs (your new
#    files use commas, e.g. "ax=0.62,ay=-0.06,..." instead of tabs) — no
#    change needed on your end, the regex parser handles both formats.
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
# Change DATA_DIR to wherever your 4 recordings live on your machine.
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "D:/Software/PersonalWork/MozekHealthtech/adl/"

# Map: activity name -> raw file for that activity.
# Each file = one continuous recording session of ONE activity.
ACTIVITY_FILES = {
    'sitting':  DATA_DIR + 'sitting.csv',
    'sleeping': DATA_DIR + 'sleeping.csv',
    'standing': DATA_DIR + 'standing.csv',
    'walking':  DATA_DIR + 'walking.csv',
    # 'running':    DATA_DIR + 'running.csv',    # add back once you record it
    # 'upstairs':   DATA_DIR + 'upstairs.csv',
    # 'downstairs': DATA_DIR + 'downstairs.csv',
}

# Full label scheme (kept complete even though not every activity has data yet)
ACTIVITY_LABELS = {
    'walking':    0,
    'running':    1,
    'sitting':    2,
    'standing':   3,
    'upstairs':   4,
    'downstairs': 5,
    'sleeping':   6,
}

SENSOR_COLS  = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']   # raw IMU columns
FEATURE_COLS = SENSOR_COLS + ['acc_mag', 'gyro_mag']  # + the 2 magnitude columns


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Parse a Raw Sensor File into a Clean DataFrame
#
# Handles lines like:
#   10:01:21.769 ax=1.016846<TAB>ay=-0.152344<TAB>az=-0.318115 ...   (old files)
#   12:10:36.337 ax=0.626465,ay=-0.062500,az=-0.969971,...            (new files)
# The regex pulls out key=value pairs regardless of whether they're
# separated by tabs, commas, or spaces.
# ─────────────────────────────────────────────────────────────────────────────
def parse_raw_sensor_file(filepath):
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:   # utf-8-sig strips BOM
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
                row[key] = float(val)

            if all(k in row for k in SENSOR_COLS):
                rows.append(row)

    return pd.DataFrame(rows)


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
# STEP 6: Feature Extraction for One Window
# 8 columns x 10 features = 80 features per window.
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(window):
    features = {}
    for col in FEATURE_COLS:
        signal = window[col].values

        features[f'{col}_mean']   = np.mean(signal)
        features[f'{col}_std']    = np.std(signal)
        features[f'{col}_var']    = np.var(signal)
        features[f'{col}_min']    = np.min(signal)
        features[f'{col}_max']    = np.max(signal)
        features[f'{col}_rms']    = np.sqrt(np.mean(signal ** 2))
        features[f'{col}_median'] = np.median(signal)

        fft_vals = np.abs(fft(signal))
        fft_vals = fft_vals[:len(fft_vals) // 2]
        if len(fft_vals) == 0:
            fft_vals = np.array([0.0])

        features[f'{col}_dom_freq'] = np.argmax(fft_vals)
        features[f'{col}_spec_energy'] = np.sum(fft_vals ** 2)
        fft_norm = fft_vals / (np.sum(fft_vals) + 1e-10)
        features[f'{col}_spec_entropy'] = -np.sum(
            fft_norm * np.log(fft_norm + 1e-10)
        )
    return features


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Process ONE Activity File End-to-End
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 20   # rows per window  (~10 seconds at ~2Hz)
STEP_SIZE   = 10   # 50% overlap

def process_activity_file(filepath, activity_name):
    df = parse_raw_sensor_file(filepath)

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
# STEP 8: Build the Full Labeled Feature Dataset (all activities combined)
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

FEATURE_OUTPUT = DATA_DIR + "all_activities_features.csv"
feature_df.to_csv(FEATURE_OUTPUT, index=False)
print(f"\nSaved combined feature dataset to: {FEATURE_OUTPUT}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Train / Test Split (time-ordered per activity, avoids overlap leakage)
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
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight='balanced'
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
# STEP 12: Predict a NEW, Unlabeled Recording (single window helper)
# ─────────────────────────────────────────────────────────────────────────────
def predict_activity_from_new_window(raw_window_df):
    df = raw_window_df.copy()
    for col in SENSOR_COLS:
        df[col] = lowpass_filter(df[col].values)
    df['acc_mag']  = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)

    window = df[FEATURE_COLS].iloc[-WINDOW_SIZE:]
    feats = extract_features(window)
    feats_df = pd.DataFrame([feats])[X_train.columns]
    return rf.predict(feats_df)[0]

print("\nAll steps complete. Model is trained and ready to use.")

# =============================================================================
# NOTES FOR YOUR CAPSTONE WRITE-UP
# =============================================================================
# 1. SAMPLING RATE: still ~2 Hz (one reading every ~0.5s). This is fine for
#    coarse posture/activity classes (sitting/standing/sleeping/walking) but
#    limits how well fast dynamics (like running strides) could be captured
#    if you add that class back later.
#
# 2. DATASET SIZE: each of the 4 current activities now has roughly an hour
#    of continuous recording, which is a big improvement over the original
#    ~1 minute clips — you should see far more stable, trustworthy accuracy
#    numbers this time.
#
# 3. SINGLE SESSION PER ACTIVITY: all the data for each activity still comes
#    from ONE continuous recording session. For a stronger capstone result,
#    consider recording a second session per activity (different time of
#    day, or a different person if applicable) so your test set reflects a
#    genuinely different session, not just a later slice of the same one.
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# STEP 13: Test the Model on a CSV File You Choose (interactive)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 13: Test the model on a CSV file")
print("=" * 60)

def load_any_sensor_csv(filepath):
    df = parse_raw_sensor_file(filepath)
    if len(df) > 0:
        return df

    df = pd.read_csv(filepath)
    missing = [c for c in ['time'] + SENSOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Couldn't parse this file. Missing expected columns: {missing}"
        )
    return df[['time'] + SENSOR_COLS]


def recognize_activity_with_timespan(filepath):
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
        feats_df = pd.DataFrame([feats])[X_train.columns]
        pred = rf.predict(feats_df)[0]
        window_predictions.append({
            'start_time': df['time'].iloc[start],
            'end_time':   df['time'].iloc[end - 1],
            'activity':   pred,
        })

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

    overall = pd.Series([wp['activity'] for wp in window_predictions]).mode()[0]
    print(f"\n>>> Overall activity detected in this file: {overall} <<<")


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
