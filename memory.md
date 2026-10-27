# Capstone Project Memory & State Tracking

## Project Overview
- **Project Name:** Human Activity Recognition (HAR) Sentinel
- **Domain:** Wearable Healthtech (MPU6050 6-Axis IMU sensor — Accelerometer + Gyroscope)
- **Capstone Subjects:**
  1. **MLF (Machine Learning Fundamentals):** Sliding windowing (20 samples, 10s at 2Hz), 80 time & spectral features (Butterworth low-pass filtering, FFT dominant frequency, spectral energy/entropy, RMS, variance), Random Forest Classifier (200 trees, balanced class weights, time-ordered train/test split to prevent leakage).
  2. **INS (Information & Network Security):** Cryptographic SHA-256 telemetry digest calculation, biomechanical boundary checks (preventing sensor spoofing / synthetic signals exceeding ±16g or ±2000°/s), sensor flatline/replay attack detection, timestamp monotonicity.
  3. **DAV (Data Analysis & Visualization):** Multi-channel sensor time-series waveforms (`ax`, `ay`, `az`, `|a|`, `gx`, `gy`, `gz`, `|ω|`) with interactive scrub/tooltip, chronological activity timeline segmentation bar, and model performance metrics (Confusion Matrix & Top Feature Importances).

## Styling & Design Constraints
- **Color Palette:** Strictly restricted to a **triple-tone** scheme:
  - Tone 1: Dark Slate Neutral (`#0d121b`, `#141b27`, `#222d3f`) — surface and structural contrast
  - Tone 2: Pale Slate Neutral (`#edf2f7`, `#8c9ba8`, `#576575`) — typography and readability
  - Tone 3: Subdued Steel Slate Blue (`#3a7596`, `#4687ab`, `rgba(58, 117, 150, 0.14)`) — accents and waveforms
- **No flashy colors, no neon, no vibrant gradients.** High-legibility modern sans-serif typography.
- **Easy to work on & clutter-free:** Core operational metrics visible directly, secondary metrics accessible via evaluation modal.

## Current State of Built Components
- `har_rf_model.joblib`: Pretrained Random Forest model trained on `all_activities_features.csv` (2,800 windows).
- `model_metadata.json`: Stored evaluation metrics (85.02% test accuracy, Confusion Matrix, Top 15 Feature Importances, Precision/Recall/F1 per class).
- `har_engine.py`: Core processing engine handling raw file parsing (both regex `ax=...` format and standard CSV), signal filtering, feature extraction, RF inference, and INS security audit.
- `trained_activity_classifier.py`: Core capstone classifier script, now updated and fixed:
  - Dynamic path resolution using `BASE_DIR` (replaces hardcoded drive letters).
  - Feature dataset caching (`all_activities_features.csv`) saving ~1 minute on runs.
  - Hybrid parsing supporting both raw serial key-value pairs and standard CSV files.
  - Frequency domain fix: FFT dominant frequency calculation ignores the DC component bin 0.
  - Non-blocking execution: `plt.close()` instead of hanging on `plt.show()`, plus safe `if __name__ == '__main__':` modular packaging.
  - Model persistence: Saves trained model to `har_rf_model.joblib`.
- **GitHub Sync:** Successfully initialized git, configured `.gitignore`, resolved initial remote README merge, and pushed to `https://github.com/Tanmay221106/Human_Activity_Recognition.git` on branch `main`.
- `server.py`: Standard library Python HTTP daemon serving static files and JSON API endpoints (`/api/sample-files`, `/api/model-info`, `/api/analyze-file`, `/api/analyze-upload`).
- `run_ui.py`: One-command launcher script that starts the server and opens the browser automatically.
- `index.html`: Clean, semantic HTML dashboard.
- `style.css`: Triple-tone aesthetic stylesheet.
- `app.js`: Client-side logic for real-time inference, canvas waveform plotting, interactive timeline, and modal metrics.

## Last Prompts & Context
- **Prompt:**
  > "go through the code of "trained_activity_classifier" this my 3rd capstone project and based on subjects MLF(machine learning fundamentals) , INS(information of network security) and DAV(Data analysis and visulazation) and my project is based on Human Activity Recognition (HAR) which recoginise human activity based on sensor data , please but a frontend for my project so make me a ui which should be only dual or triple tone not more than that and make sure that the color and font are not tooo flashy and neon make sure the things are not too vibrant and the overall ui that you are building for this particular is easy to work on and rest you cna make the changes which you think have advantages in this particular field don't add tooo much extra things and not required feedback also make sure only main things are visible mainly and rest of the thing can be ignored"
- **Resume Directive:**
  > "/continue and add a memory.md file to store the last prompt to contiune when error"
