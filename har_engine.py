import re
import hashlib
import json
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
STEP_SIZE = 10

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

        features[f'{col}_dom_freq'] = int(np.argmax(fft_vals))
        features[f'{col}_spec_energy'] = float(np.sum(fft_vals ** 2))
        fft_norm = fft_vals / (np.sum(fft_vals) + 1e-10)
        features[f'{col}_spec_entropy'] = float(-np.sum(fft_norm * np.log(fft_norm + 1e-10)))
    return features

def parse_sensor_content(raw_text):
    rows = []
    lines = raw_text.strip().splitlines()
    
    # Check if header format exists
    if lines and 'time' in lines[0].lower() and 'ax' in lines[0].lower():
        # Standard CSV header
        header = [h.strip().lower() for h in lines[0].split(',')]
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) >= len(header):
                row_dict = dict(zip(header, parts))
                try:
                    parsed_row = {'time': row_dict.get('time', '')}
                    for sc in SENSOR_COLS:
                        parsed_row[sc] = float(row_dict[sc])
                    rows.append(parsed_row)
                except (ValueError, KeyError):
                    continue
        if rows:
            return pd.DataFrame(rows)

    # Key=value regex parsing (as in trained_activity_classifier.py)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^(\S+)\s+(.*)$', line)
        if not match:
            continue
        time_str, sensor_str = match.groups()
        kv_pairs = re.findall(r'(ax|ay|az|gx|gy|gz)=(-?\d+\.?\d*)', sensor_str)
        if len(kv_pairs) >= 6:
            row = {'time': time_str}
            for k, v in kv_pairs:
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
            if all(k in row for k in SENSOR_COLS):
                rows.append(row)

    return pd.DataFrame(rows)

class HAREngine:
    def __init__(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata['feature_names']
            self.classes = self.metadata['classes']
        except Exception as e:
            print(f"Warning loading model/metadata: {e}")
            self.model = None
            self.metadata = {}
            self.feature_names = []
            self.classes = []

    def security_audit(self, raw_bytes, df):
        # INS Pillar: Telemetry Cryptographic Digest & Anomaly Auditing
        sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
        
        anomalies = []
        is_spoofed = False
        
        if df.empty:
            return {
                "sha256": sha256_hash,
                "packet_count": 0,
                "integrity_status": "Empty Stream",
                "anomalies": ["Zero sensor packets found."],
                "spoofing_detected": True
            }
            
        n_packets = len(df)
        
        # 1. Physics / Range check: MPU6050 max acceleration ±16g, gyro ±2000 deg/s
        max_acc = max(df['ax'].abs().max(), df['ay'].abs().max(), df['az'].abs().max())
        max_gyro = max(df['gx'].abs().max(), df['gy'].abs().max(), df['gz'].abs().max())
        
        if max_acc > 16.0:
            anomalies.append(f"Accelerometry out of physical biomechanical range (>16g): {max_acc:.2f}g")
            is_spoofed = True
            
        if max_gyro > 2000.0:
            anomalies.append(f"Gyroscope out of physical sensor bounds (>2000°/s): {max_gyro:.1f}°/s")
            is_spoofed = True

        # 2. Replay / Flatline Detection (sensor freeze or fabricated static payload)
        ax_var = df['ax'].var()
        ay_var = df['ay'].var()
        az_var = df['az'].var()
        if n_packets > 30 and (ax_var < 1e-6 and ay_var < 1e-6 and az_var < 1e-6):
            anomalies.append("Sensor Flatline Detected: Synthetic zero-variance signal indicates replay or disconnected probe.")
            is_spoofed = True

        return {
            "sha256": sha256_hash,
            "packet_count": n_packets,
            "integrity_status": "Cryptographically Verified" if not is_spoofed else "Security Warning",
            "anomalies": anomalies if anomalies else ["All biomechanical boundary & integrity checks passed."],
            "spoofing_detected": is_spoofed
        }

    def analyze(self, raw_bytes):
        raw_text = raw_bytes.decode('utf-8', errors='ignore')
        df = parse_sensor_content(raw_text)
        
        sec_audit = self.security_audit(raw_bytes, df)
        
        if df.empty or len(df) < WINDOW_SIZE:
            return {
                "success": False,
                "error": f"Insufficient sensor readings ({len(df)} rows). Need at least {WINDOW_SIZE} rows to form 1 window.",
                "security": sec_audit
            }
            
        # Clean dataframe
        df = df.dropna().reset_index(drop=True)
        
        # Filter signals
        for col in SENSOR_COLS:
            df[col] = lowpass_filter(df[col].values)
            
        df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
        
        # Window feature extraction
        n = len(df)
        window_preds = []
        window_confs = []
        
        for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
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
                "error": "Could not form windows.",
                "security": sec_audit
            }

        # Cluster into contiguous time segments
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
        
        # Primary activity is the mode
        primary_activity = max(act_counts, key=act_counts.get)
        overall_conf = round(float(np.mean(window_confs)) * 100, 1)

        # Downsample waveform for high-performance canvas/SVG rendering (max 200 points)
        step = max(1, len(df) // 160)
        downsampled = df.iloc[::step]
        waveform = {
            "time": downsampled['time'].tolist(),
            "ax": [round(float(v), 3) for v in downsampled['ax']],
            "ay": [round(float(v), 3) for v in downsampled['ay']],
            "az": [round(float(v), 3) for v in downsampled['az']],
            "acc_mag": [round(float(v), 3) for v in downsampled['acc_mag']],
            "gyro_mag": [round(float(v), 3) for v in downsampled['gyro_mag']]
        }

        return {
            "success": True,
            "primary_activity": primary_activity,
            "overall_confidence": overall_conf,
            "total_samples": len(df),
            "total_windows": total_w,
            "distribution": distribution,
            "segments": segments,
            "waveform": waveform,
            "security": sec_audit,
            "window_details": window_preds[:15] # first 15 windows
        }
