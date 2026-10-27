import re
import hashlib
import json
import io
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import joblib

MODEL_PATH = "har_rf_model.joblib"
METADATA_PATH = "model_metadata.json"

SENSOR_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
FEATURE_COLS = SENSOR_COLS + ['acc_mag', 'gyro_mag']
WINDOW_SIZE = 20
DEFAULT_STEP = 10

def lowpass_filter(data, cutoff=0.3, order=4):
    padlen = 3 * order
    if len(data) <= padlen:
        return data
    b, a = butter(order, cutoff, btype='low')
    return filtfilt(b, a, data)

def extract_features(window):
    features = {}
    for col in FEATURE_COLS:
        signal = window[col].values
        features[f'{col}_mean'] = float(np.mean(signal))
        features[f'{col}_std'] = float(np.std(signal))
        features[f'{col}_var'] = float(np.var(signal))
        features[f'{col}_min'] = float(np.min(signal))
        features[f'{col}_max'] = float(np.max(signal))
        features[f'{col}_rms'] = float(np.sqrt(np.mean(signal ** 2)))
        features[f'{col}_median'] = float(np.median(signal))

        fft_vals = np.abs(fft(signal))
        fft_vals = fft_vals[:len(fft_vals) // 2]
        if len(fft_vals) == 0:
            fft_vals = np.array([0.0])

        if len(fft_vals) > 1:
            dom_idx = int(np.argmax(fft_vals[1:])) + 1
        else:
            dom_idx = int(np.argmax(fft_vals))

        features[f'{col}_dom_freq'] = dom_idx
        features[f'{col}_spec_energy'] = float(np.sum(fft_vals ** 2))
        fft_norm = fft_vals / (np.sum(fft_vals) + 1e-10)
        features[f'{col}_spec_entropy'] = float(-np.sum(fft_norm * np.log(fft_norm + 1e-10)))
    return features

def parse_sensor_content(raw_text):
    if not raw_text or not raw_text.strip():
        return pd.DataFrame()

    lines = raw_text.strip().splitlines()

    # Attempt 1: Universal CSV parsing via pandas
    try:
        # Detect delimiter (comma, tab, semicolon)
        sample = lines[0] if len(lines) > 0 else ""
        delim = '\t' if '\t' in sample and ',' not in sample else (';' if ';' in sample and ',' not in sample else ',')
        
        df = pd.read_csv(io.StringIO(raw_text), sep=delim, skipinitialspace=True, on_bad_lines='skip')
        col_clean = {c: re.sub(r'[^a-zA-Z0-9_]', '', str(c).strip().lower()) for c in df.columns}
        df = df.rename(columns=col_clean)

        alias_map = {
            'acc_x': 'ax', 'accx': 'ax', 'x_acc': 'ax', 'accel_x': 'ax', 'accelerometer_x': 'ax',
            'acc_y': 'ay', 'accy': 'ay', 'y_acc': 'ay', 'accel_y': 'ay', 'accelerometer_y': 'ay',
            'acc_z': 'az', 'accz': 'az', 'z_acc': 'az', 'accel_z': 'az', 'accelerometer_z': 'az',
            'gyro_x': 'gx', 'gyrox': 'gx', 'x_gyro': 'gx', 'gyroscope_x': 'gx',
            'gyro_y': 'gy', 'gyroy': 'gy', 'y_gyro': 'gy', 'gyroscope_y': 'gy',
            'gyro_z': 'gz', 'gyroz': 'gz', 'z_gyro': 'gz', 'gyroscope_z': 'gz',
            'timestamp': 'time', 'datetime': 'time', 't': 'time'
        }
        df = df.rename(columns=alias_map)

        if all(c in df.columns for c in ['ax', 'ay', 'az']):
            # Auto-fill gyros with 0.0 if not recorded
            for g in ['gx', 'gy', 'gz']:
                if g not in df.columns:
                    df[g] = 0.0

            if 'time' not in df.columns:
                df['time'] = [f"{i*0.5:.1f}s" for i in range(len(df))]

            for col in SENSOR_COLS:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna(subset=SENSOR_COLS).reset_index(drop=True)
            if len(df) >= WINDOW_SIZE:
                return df[['time'] + SENSOR_COLS]
    except Exception:
        pass

    # Attempt 2: Key=Value Regex (e.g. "12:10:36.337 ax=0.626465,ay=-0.062500,...")
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^(\S+)\s+(.*)$', line)
        if not match:
            continue
        time_str, sensor_str = match.groups()
        kv_pairs = re.findall(r'(ax|ay|az|gx|gy|gz)=(-?\d+\.?\d*)', sensor_str)
        if len(kv_pairs) >= 3:
            row = {'time': time_str, 'gx': 0.0, 'gy': 0.0, 'gz': 0.0}
            for k, v in kv_pairs:
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
            if all(k in row for k in ['ax', 'ay', 'az']):
                rows.append(row)

    if rows:
        return pd.DataFrame(rows)

    return pd.DataFrame()

class HAREngine:
    def __init__(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get('feature_names', [])
            self.classes = self.metadata.get('classes', [])
        except Exception as e:
            print(f"HAREngine initialization warning: {e}")
            self.model = None
            self.metadata = {}
            self.feature_names = []
            self.classes = []

    def security_audit(self, raw_bytes, df):
        sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
        anomalies = []
        is_spoofed = False

        if df.empty:
            return {
                "sha256": sha256_hash,
                "packet_count": 0,
                "integrity_status": "Empty Stream",
                "anomalies": ["Zero sensor readings found."],
                "spoofing_detected": True
            }

        n_packets = len(df)
        max_acc = max(df['ax'].abs().max(), df['ay'].abs().max(), df['az'].abs().max())
        max_gyro = max(df['gx'].abs().max(), df['gy'].abs().max(), df['gz'].abs().max())

        if max_acc > 16.0:
            anomalies.append(f"Acceleration exceeded physical sensor limits (>16g): {max_acc:.2f}g")
            is_spoofed = True

        if max_gyro > 2000.0:
            anomalies.append(f"Angular velocity exceeded sensor limits (>2000°/s): {max_gyro:.1f}°/s")
            is_spoofed = True

        if n_packets > 30 and (df['ax'].var() < 1e-6 and df['ay'].var() < 1e-6 and df['az'].var() < 1e-6):
            anomalies.append("Sensor Flatline: Zero variance detected (possible disconnected probe or static replay).")
            is_spoofed = True

        return {
            "sha256": sha256_hash,
            "packet_count": n_packets,
            "integrity_status": "Verified Safe" if not is_spoofed else "Security Anomaly",
            "anomalies": anomalies if anomalies else ["All physical boundary and integrity checks passed."],
            "spoofing_detected": is_spoofed
        }

    def analyze(self, raw_bytes):
        raw_text = raw_bytes.decode('utf-8', errors='ignore')
        df = parse_sensor_content(raw_text)

        sec_audit = self.security_audit(raw_bytes, df)

        if df.empty or len(df) < WINDOW_SIZE:
            return {
                "success": False,
                "error": f"Uploaded file has only {len(df)} valid sensor rows. At least {WINDOW_SIZE} rows (~10s) are needed for feature windowing.",
                "security": sec_audit
            }

        df = df.dropna().reset_index(drop=True)

        for col in SENSOR_COLS:
            df[col] = lowpass_filter(df[col].values)

        df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)

        # High-speed window evaluation (cap at max 120 windows for instant sub-second response)
        n = len(df)
        step_size = max(DEFAULT_STEP, (n - WINDOW_SIZE) // 120)

        window_preds = []
        window_confs = []

        for start in range(0, n - WINDOW_SIZE + 1, step_size):
            end = start + WINDOW_SIZE
            win = df[FEATURE_COLS].iloc[start:end]
            feats = extract_features(win)

            feats_df = pd.DataFrame([feats])[self.feature_names]
            probs = self.model.predict_proba(feats_df)[0]
            pred_idx = np.argmax(probs)
            pred_class = self.classes[pred_idx]
            conf = float(probs[pred_idx])

            window_preds.append({
                'start_idx': start,
                'end_idx': end,
                'start_time': str(df['time'].iloc[start]),
                'end_time': str(df['time'].iloc[end - 1]),
                'activity': pred_class,
                'confidence': round(conf * 100, 1),
                'probabilities': {cls: round(float(p) * 100, 1) for cls, p in zip(self.classes, probs)}
            })
            window_confs.append(conf)

        if not window_preds:
            return {
                "success": False,
                "error": "Could not extract sufficient feature windows.",
                "security": sec_audit
            }

        # Contiguous timeline segments
        segments = []
        for wp in window_preds:
            if segments and segments[-1]['activity'] == wp['activity']:
                segments[-1]['end_time'] = wp['end_time']
                segments[-1]['window_count'] += 1
                segments[-1]['confidence_list'].append(wp['confidence'])
            else:
                segments.append({
                    'start_time': wp['start_time'],
                    'end_time': wp['end_time'],
                    'activity': wp['activity'],
                    'window_count': 1,
                    'confidence_list': [wp['confidence']]
                })

        for s in segments:
            s['avg_confidence'] = round(float(np.mean(s['confidence_list'])), 1)
            del s['confidence_list']

        # Activity Distribution
        act_counts = {}
        for wp in window_preds:
            act_counts[wp['activity']] = act_counts.get(wp['activity'], 0) + 1
        total_w = len(window_preds)
        distribution = {
            act: round((count / total_w) * 100, 1)
            for act, count in act_counts.items()
        }

        primary_activity = max(act_counts, key=act_counts.get)
        overall_conf = round(float(np.mean(window_confs)) * 100, 1)

        # Downsample waveform for ultra-smooth canvas rendering (~150 points)
        step = max(1, len(df) // 150)
        downsampled = df.iloc[::step]
        waveform = {
            "time": downsampled['time'].tolist(),
            "ax": [round(float(v), 3) for v in downsampled['ax']],
            "ay": [round(float(v), 3) for v in downsampled['ay']],
            "az": [round(float(v), 3) for v in downsampled['az']],
            "acc_mag": [round(float(v), 3) for v in downsampled['acc_mag']],
            "gyro_mag": [round(float(v), 3) for v in downsampled['gyro_mag']]
        }

        # Approximate recording duration in seconds (~0.5s per row)
        duration_sec = round(len(df) * 0.5)
        duration_str = f"{duration_sec // 60}m {duration_sec % 60}s" if duration_sec >= 60 else f"{duration_sec}s"

        return {
            "success": True,
            "primary_activity": primary_activity,
            "overall_confidence": overall_conf,
            "total_samples": len(df),
            "duration": duration_str,
            "total_windows": total_w,
            "distribution": distribution,
            "segments": segments,
            "waveform": waveform,
            "security": sec_audit
        }
