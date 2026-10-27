# =============================================================================
# SENSOR DATA PROCESSING — MPU6050 (Accelerometer + Gyroscope)
# Steps: Import → Clean → Separate → Convert → Filter → Normalize → Magnitude
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: Import Libraries
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import re
import os

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Import / Load the Data
# ─────────────────────────────────────────────────────────────────────────────

# ── Change this to your original raw serial file ──
FILE_PATH = "D:/Software/PersonalWork/MozekHealthtech/adl/cleaned_sensor_data.csv"

# Regex pattern to find ax=, ay=, az=, gx=, gy=, gz= in each line
print("=" * 60)
print("STEP 1: Loading raw data from file...")
print("=" * 60)

df = pd.read_csv(FILE_PATH)

print(f"\n✅ Loaded {len(df)} valid rows.")
print("Raw DataFrame (first 5 rows):")
print(df.head())
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

df = pd.read_csv(FILE_PATH)
print("After Step 1 (load):", df.shape)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Clean the Data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Cleaning the data...")
print("=" * 60)

rows_before = len(df)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
rows_after = len(df)

print(f"✅ Rows before cleaning: {rows_before}")
print(f"✅ Rows after cleaning:  {rows_after}")
print(f"🗑  Removed {rows_before - rows_after} null/incomplete rows.\n")


df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print("After Step 2 (dropna):", df.shape)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Separate Sensor Values into Clear Columns
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Verifying column structure...")
print("=" * 60)

SENSOR_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

print("Columns in dataset:", list(df.columns))
print("\nSensor columns:")
print("  Accelerometer → ax, ay, az")
print("  Gyroscope     → gx, gy, gz")
print("\nData sample:")
print(df[['time'] + SENSOR_COLS].head())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Convert Values to Numeric
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Converting sensor values to numeric (float)...")
print("=" * 60)

print("Data types BEFORE conversion:")
print(df[SENSOR_COLS].dtypes)

for col in SENSOR_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=SENSOR_COLS, inplace=True)
df.reset_index(drop=True, inplace=True)

print("\nData types AFTER conversion:")
print(df[SENSOR_COLS].dtypes)
print("\nBasic statistics of the sensor data:")
print(df[SENSOR_COLS].describe().round(3))


# ... after Step 4's numeric conversion + dropna:
print("After Step 4 (to_numeric):", df.shape)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Remove Noise with a Low-Pass Filter
#
# A Butterworth low-pass filter:
#   - Keeps slow, smooth changes (real motion)
#   - Removes fast random spikes (noise)
# filtfilt() applies it forward + backward so signal isn't shifted in time.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Applying Low-Pass Filter to remove noise...")
print("=" * 60)

def lowpass_filter(data, cutoff=0.1, order=4):
    """
    Butterworth low-pass filter.
    cutoff=0.1 means keep only smooth/slow changes.
    padlen check avoids crash when dataset is too small.
    """
    padlen = 3 * order          # needs at least 12 rows to work
    if len(data) <= padlen:
        return data             # skip filter if too few rows
    b, a = butter(order, cutoff, btype='low')
    return filtfilt(b, a, data)

for col in SENSOR_COLS:
    df[col + '_raw'] = df[col].copy()       # save original for comparison
    df[col] = lowpass_filter(df[col].values)

print("✅ Low-pass filter applied to: ax, ay, az, gx, gy, gz")
print("   (Originals saved as ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw)\n")



# ... after Step 5's filtering loop:
print("After Step 5 (filter):", df.shape)
# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Normalize the Data (StandardScaler)
#
# Why normalize?
#   Accelerometer values are small (e.g. 0.4 g)
#   Gyroscope values are large (e.g. -101 °/s)
#   Machine learning gets confused when features have very different ranges.
#
# StandardScaler transforms each column so that:
#   Mean  = 0   (centers the data around zero)
#   Std   = 1   (all columns have the same spread)
#
# Formula for each value:  z = (x - mean) / std
#
# This means: "how many standard deviations away from average is this reading?"
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 6: Normalizing sensor data (StandardScaler)...")
print("=" * 60)

print("Statistics BEFORE normalization:")
print(df[SENSOR_COLS].describe().round(3))

# Save pre-normalization values for comparison
for col in SENSOR_COLS:
    df[col + '_filtered'] = df[col].copy()

# StandardScaler: fit_transform calculates mean & std, then applies the formula
scaler = StandardScaler()
df[SENSOR_COLS] = scaler.fit_transform(df[SENSOR_COLS])

print("\nStatistics AFTER normalization (mean≈0, std≈1):")
print(df[SENSOR_COLS].describe().round(3))
print("\n✅ All 6 sensor columns normalized.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Calculate Magnitude Features
#
# Why magnitude?
#   Individual x, y, z values depend on HOW the sensor is oriented.
#   If you rotate the device, x and y swap — but the total motion doesn't change.
#
#   Magnitude = sqrt(x² + y² + z²)
#
#   This gives a single number representing the TOTAL strength of motion,
#   regardless of orientation. Very useful for:
#     - Activity Detection (walking, running, standing)
#     - Fall Detection
#     - Gesture Recognition
#
# acc_mag  = total acceleration strength
# gyro_mag = total rotation speed
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 7: Calculating Magnitude Features...")
print("=" * 60)

# Accelerometer magnitude: how strong is the total linear acceleration?
df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

# Gyroscope magnitude: how fast is the total rotation?
df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)

print("✅ acc_mag  = sqrt(ax² + ay² + az²)  → total acceleration strength")
print("✅ gyro_mag = sqrt(gx² + gy² + gz²)  → total rotation speed")
print(f"\nacc_mag  — mean: {df['acc_mag'].mean():.4f}, max: {df['acc_mag'].max():.4f}")
print(f"gyro_mag — mean: {df['gyro_mag'].mean():.4f}, max: {df['gyro_mag'].max():.4f}")
print("\nFirst 5 rows with magnitude columns:")
print(df[['time', 'ax', 'ay', 'az', 'acc_mag', 'gx', 'gy', 'gz', 'gyro_mag']].head().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# FINAL RESULT: Save the processed dataset
# ─────────────────────────────────────────────────────────────────────────────
final_cols = ['time', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'acc_mag', 'gyro_mag']
df_final = df[final_cols].copy()

print("\n" + "=" * 60)
print("FINAL DATASET (first 10 rows):")
print("=" * 60)
print(df_final.head(10).to_string(index=True))
print(f"\nTotal rows in final dataset: {len(df_final)}")

OUTPUT_PATH = "D:/Software/PersonalWork/MozekHealthtech/adl/processed_sensor_data.csv"
df_final.to_csv(OUTPUT_PATH, index=False)
print(f"\n💾 Saved processed data to: {OUTPUT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS: Raw vs Filtered + Magnitude over time
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊 Generating plots...")

# --- Plot 1: Raw vs Filtered (before normalization) ---
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("Raw vs Filtered Sensor Data\n(Blue = Raw, Orange = Filtered)", fontsize=14)

plot_pairs = [
    ('ax', 'Accelerometer X (ax)'), ('ay', 'Accelerometer Y (ay)'),
    ('az', 'Accelerometer Z (az)'), ('gx', 'Gyroscope X (gx)'),
    ('gy', 'Gyroscope Y (gy)'),     ('gz', 'Gyroscope Z (gz)'),
]

for ax_plot, (col, title) in zip(axes.flat, plot_pairs):
    ax_plot.plot(df[col + '_raw'].values,      color='steelblue', alpha=0.5, linewidth=0.8, label='Raw')
    ax_plot.plot(df[col + '_filtered'].values, color='orangered', linewidth=1.5,            label='Filtered')
    ax_plot.set_title(title, fontsize=11)
    ax_plot.set_xlabel("Sample Index")
    ax_plot.legend(fontsize=8)
    ax_plot.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sensor_raw_vs_filtered.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 1 saved: sensor_raw_vs_filtered.png")

# --- Plot 2: Magnitude over time ---
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
fig2.suptitle("Sensor Magnitude Over Time", fontsize=14)

ax1.plot(df['acc_mag'].values,  color='steelblue', linewidth=1.5)
ax1.set_title("Accelerometer Magnitude (acc_mag)")
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Magnitude")
ax1.grid(True, alpha=0.3)

ax2.plot(df['gyro_mag'].values, color='orangered', linewidth=1.5)
ax2.set_title("Gyroscope Magnitude (gyro_mag)")
ax2.set_xlabel("Sample Index")
ax2.set_ylabel("Magnitude")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sensor_magnitude.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 2 saved: sensor_magnitude.png")

print("\n🎉 All 7 steps complete!")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Segment Data into Windows
#
# Why windows?
#   A single sensor reading (one row) tells you almost nothing about activity.
#   Is the person walking? Standing? You can't tell from one point.
#
#   But if you look at 100 readings together (~2 seconds of data at 50Hz),
#   patterns become visible:
#     - Walking has a repeating up-down rhythm
#     - Running has faster, stronger rhythms
#     - Standing is nearly flat/still
#
# How windowing works:
#   window_size = 100  → each window has 100 rows (~2 seconds at 50Hz)
#   step_size   = 50   → each new window starts 50 rows after the last
#                        (50% overlap — windows share half their data)
#
# Example with 200 rows:
#   Window 1: rows 0–99
#   Window 2: rows 50–149   ← starts 50 rows after Window 1
#   Window 3: rows 100–199
#   ...and so on
#
# Overlap is important because activities don't start exactly at window edges.
# Overlapping ensures no transition between activities is missed.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Segmenting data into sliding windows...")
print("=" * 60)

# The columns we want to extract features from
FEATURE_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'acc_mag', 'gyro_mag']

window_size = 100   # number of rows per window (2 seconds if sensor runs at 50Hz)
step_size   = 50    # how many rows to move forward before starting next window
                    # step_size = window_size/2 → 50% overlap

windows = []        # will hold each window as a small DataFrame chunk

total_rows = len(df)

# Slide the window across all rows
# range(start, stop, step):
#   start = 0 (begin at row 0)
#   stop  = total_rows - window_size + 1 (stop when window would go out of bounds)
#   step  = step_size (move forward by step_size each time)
for start in range(0, total_rows - window_size + 1, step_size):
    end = start + window_size          # end index of this window (exclusive)
    window = df[FEATURE_COLS].iloc[start:end]  # slice the DataFrame
    windows.append(window)             # add this window to our list

print(f"✅ Total rows in dataset     : {total_rows}")
print(f"✅ Window size               : {window_size} rows")
print(f"✅ Step size (overlap)       : {step_size} rows (50% overlap)")
print(f"✅ Total windows created     : {len(windows)}")
print(f"\nExample — Window 1 (rows 0–{window_size-1}):")
print(windows[0].describe().round(3))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Extract Features from Each Window
#
# For each window we calculate two types of features:
#
# ── TIME DOMAIN FEATURES ──────────────────────────────────────────────────
#   These describe the "shape" and "spread" of values within the window.
#   All calculated directly on the raw numbers.
#
#   mean     → average value (e.g. is the sensor tilted?)
#   std      → standard deviation (how much does it vary?)
#   var      → variance = std² (another measure of spread)
#   min      → smallest value in the window
#   max      → largest value in the window
#   rms      → Root Mean Square = sqrt(mean of squares)
#              RMS captures the "energy" of the signal, especially useful
#              when values go negative (e.g. gyroscope)
#   median   → middle value (robust to outliers unlike mean)
#
# ── FREQUENCY DOMAIN FEATURES (FFT) ──────────────────────────────────────
#   FFT = Fast Fourier Transform
#   It converts the time-series signal into a list of frequencies.
#   Think of it like splitting white light through a prism into colors —
#   FFT splits a signal into its "frequency components".
#
#   dominant_freq  → which frequency has the most energy?
#                    Walking: ~2Hz, Running: ~3Hz, Standing: ~0Hz
#   spectral_energy → total energy across all frequencies
#                    High energy = active motion
#   spectral_entropy → how "random" or "spread out" the frequencies are
#                    Low entropy = rhythmic (walking)
#                    High entropy = chaotic/random
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Extracting features from each window...")
print("=" * 60)

from scipy.fft import fft   # FFT tool from scipy

def extract_features(window):
    """
    Takes one window (DataFrame of shape window_size × 8) and returns
    a flat dictionary of features — one value per feature per column.
    
    For 8 columns × 10 features = 80 features per window.
    """
    features = {}

    for col in FEATURE_COLS:
        signal = window[col].values   # get the column as a numpy array

        # ── Time Domain ──────────────────────────────────────
        features[f'{col}_mean']   = np.mean(signal)
        # mean: sum of all values / count
        # tells you the "center" of the signal

        features[f'{col}_std']    = np.std(signal)
        # std: how spread out the values are around the mean
        # high std = lots of movement, low std = steady/still

        features[f'{col}_var']    = np.var(signal)
        # var = std² — another way to measure spread

        features[f'{col}_min']    = np.min(signal)
        # smallest value seen in this window

        features[f'{col}_max']    = np.max(signal)
        # largest value seen in this window

        features[f'{col}_rms']    = np.sqrt(np.mean(signal**2))
        # RMS: square all values, average them, take sqrt
        # captures energy even when signal oscillates around 0

        features[f'{col}_median'] = np.median(signal)
        # middle value when sorted — not affected by extreme spikes

        # ── Frequency Domain (FFT) ────────────────────────────
        fft_vals = np.abs(fft(signal))
        # fft() returns complex numbers representing each frequency component
        # np.abs() gives us the magnitude (strength) of each frequency
        # Result: array where index = frequency bin, value = strength

        fft_vals = fft_vals[:len(fft_vals)//2]
        # FFT output is symmetric — we only need the first half
        # (the second half is a mirror of the first)

        # Dominant frequency: which frequency bin has the most energy?
        features[f'{col}_dom_freq'] = np.argmax(fft_vals)
        # argmax() returns the INDEX of the largest value
        # higher index = higher frequency = faster motion

        # Spectral energy: total power across all frequencies
        features[f'{col}_spec_energy'] = np.sum(fft_vals**2)
        # square each frequency's strength, sum them all
        # high = lots of energy = active movement

        # Spectral entropy: how "spread out" or "random" are the frequencies?
        fft_norm = fft_vals / (np.sum(fft_vals) + 1e-10)
        # normalize so values sum to 1 (like probabilities)
        # +1e-10 avoids division by zero

        features[f'{col}_spec_entropy'] = -np.sum(
            fft_norm * np.log(fft_norm + 1e-10)
        )
        # entropy formula: -sum(p * log(p))
        # low entropy  → energy concentrated at one frequency (walking rhythm)
        # high entropy → energy spread across many frequencies (random noise)

    return features

# Apply extract_features to every window
feature_list = []
for i, window in enumerate(windows):
    feat = extract_features(window)
    feat['window_id'] = i           # label which window this is
    feature_list.append(feat)

# Convert list of feature dicts into a DataFrame
# Each row = one window, each column = one feature
df_features = pd.DataFrame(feature_list)

# Move window_id to be the first column (just for readability)
cols = ['window_id'] + [c for c in df_features.columns if c != 'window_id']
df_features = df_features[cols]

print(f"✅ Features extracted from {len(windows)} windows")
print(f"✅ Features per window: {len(df_features.columns) - 1}")
print(f"   (8 columns × 10 features each = {8*10} total features)")
print(f"\nFeature DataFrame shape: {df_features.shape}")
print(f"\nFirst 3 windows (first 10 features shown):")
print(df_features.iloc[:3, :11].to_string(index=False))

# Save the feature dataset
FEATURE_OUTPUT = "D:/Software/PersonalWork/MozekHealthtech/adl/sensor_features.csv"
df_features.to_csv(FEATURE_OUTPUT, index=False)
print(f"\n💾 Saved feature dataset to: {FEATURE_OUTPUT}")
print(f"   This file is ready for machine learning! 🎯")

# ── Plot: Feature comparison across windows ───────────────────────────────
print("\n📊 Plotting feature trends across windows...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Extracted Features Across Windows", fontsize=14)

# Plot mean of acc_mag per window
axes[0,0].plot(df_features['acc_mag_mean'], color='steelblue', linewidth=1.5)
axes[0,0].set_title("acc_mag — Mean per Window")
axes[0,0].set_xlabel("Window Index")
axes[0,0].set_ylabel("Mean Value")
axes[0,0].grid(True, alpha=0.3)

# Plot std of acc_mag per window
axes[0,1].plot(df_features['acc_mag_std'], color='orangered', linewidth=1.5)
axes[0,1].set_title("acc_mag — Std Dev per Window")
axes[0,1].set_xlabel("Window Index")
axes[0,1].set_ylabel("Std Dev")
axes[0,1].grid(True, alpha=0.3)

# Plot dominant frequency of acc_mag per window
axes[1,0].plot(df_features['acc_mag_dom_freq'], color='green', linewidth=1.5)
axes[1,0].set_title("acc_mag — Dominant Frequency per Window")
axes[1,0].set_xlabel("Window Index")
axes[1,0].set_ylabel("Frequency Bin")
axes[1,0].grid(True, alpha=0.3)

# Plot spectral energy of acc_mag per window
axes[1,1].plot(df_features['acc_mag_spec_energy'], color='purple', linewidth=1.5)
axes[1,1].set_title("acc_mag — Spectral Energy per Window")
axes[1,1].set_xlabel("Window Index")
axes[1,1].set_ylabel("Energy")
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sensor_features.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Feature plot saved: sensor_features.png")

print("\n🎉 All 9 steps complete! Your data is ready for ML classification.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: Label Activities
#
# Each CSV file is assumed to be ONE recording session of ONE activity.
# So every window extracted from this file gets the SAME activity label.
#
# Activity      Label
# Walking       0
# Running       1
# Sitting       2
# Standing      3
# Upstairs      4
# Downstairs    5
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Labeling activity...")
print("=" * 60)

ACTIVITY_LABELS = {
    'walking':    0,
    'running':    1,
    'sitting':    2,
    'standing':   3,
    'upstairs':   4,
    'downstairs': 5,
}

# ── Set this to whatever activity THIS file's recording actually is ──
CURRENT_ACTIVITY = 'walking'   # <-- change this per file/session

label = ACTIVITY_LABELS[CURRENT_ACTIVITY]
df_features['activity'] = CURRENT_ACTIVITY
df_features['label'] = label

print(f"✅ Activity for this session : {CURRENT_ACTIVITY}")
print(f"✅ Numeric label assigned    : {label}")
print(f"✅ Applied to all {len(df_features)} window(s)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: Save Preprocessed (Labeled) Dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: Saving labeled feature dataset...")
print("=" * 60)

# Save one file per activity/session — you'll combine them later
FEATURE_OUTPUT = f"D:/Software/PersonalWork/MozekHealthtech/adl/features_{CURRENT_ACTIVITY}.csv"
df_features.to_csv(FEATURE_OUTPUT, index=False)

print(f"✅ Saved {len(df_features)} labeled window(s) to: {FEATURE_OUTPUT}")
print(f"   Columns: {list(df_features.columns)}")