import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = "all_activities_features.csv"

def train_and_export():
    print("Loading feature dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Time-ordered train/test split per activity (75/25) matching trained_activity_classifier.py
    train_parts, test_parts = [], []
    activities = df['activity'].unique()
    
    for act in activities:
        subset = df[df['activity'] == act]
        split_idx = int(len(subset) * 0.75)
        train_parts.append(subset.iloc[:split_idx])
        test_parts.append(subset.iloc[split_idx:])
        
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    
    feature_cols = [c for c in df.columns if c != 'activity']
    X_train = train_df[feature_cols]
    y_train = train_df['activity']
    X_test = test_df[feature_cols]
    y_test = test_df['activity']
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    labels_order = sorted(list(activities))
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    importance = pd.Series(rf.feature_importances_, index=feature_cols)
    top_features = importance.sort_values(ascending=False).head(15)
    
    metadata = {
        "accuracy": round(float(acc) * 100, 2),
        "total_train_samples": len(X_train),
        "total_test_samples": len(X_test),
        "classes": labels_order,
        "feature_names": feature_cols,
        "confusion_matrix": {
            "labels": labels_order,
            "matrix": cm.tolist()
        },
        "top_features": [
            {"feature": k, "importance": round(float(v), 4)}
            for k, v in top_features.items()
        ],
        "metrics_per_class": {
            cls: {
                "precision": round(report[cls]["precision"] * 100, 1),
                "recall": round(report[cls]["recall"] * 100, 1),
                "f1": round(report[cls]["f1-score"] * 100, 1),
                "support": int(report[cls]["support"])
            }
            for cls in labels_order if cls in report
        }
    }
    
    joblib.dump(rf, "har_rf_model.joblib")
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("Exported har_rf_model.joblib and model_metadata.json successfully!")

if __name__ == "__main__":
    train_and_export()
