import pandas as pd
import joblib

from pothole_detection import PotholeDetector

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Data" / "pothole_ai_model.pkl"

# -----------------------------
# Load dataset
# -----------------------------

data = pd.read_csv(BASE_DIR / "Data" / "synthetic_pothole_dataset.csv")


# -----------------------------
# Load trained AI model
# -----------------------------

ai_model = joblib.load(BASE_DIR  / "Data" / "pothole_ai_model.pkl")


# -----------------------------
# Initialize physics detector
# -----------------------------

detector = PotholeDetector()


physics_detections = 0
ai_confirmed_detections = 0

true_positives = 0
false_positives = 0
false_negatives = 0


print("\nRunning pothole detection with AI filtering...\n")


# -----------------------------
# Process dataset
# -----------------------------

for _, row in data.iterrows():

    result = detector.process_sample(
        timestamp=row["timestamp"],
        ax=row["ax"],
        ay=row["ay"],
        az=row["az"],
        gx=row["gx"],
        gy=row["gy"],
        gz=row["gz"],
        speed=row["speed"]
    )

    # Physics detection
    if result["pothole_detected"]:

        physics_detections += 1

        # AI input features
        features = pd.DataFrame([{
            "ax": row["ax"],
            "ay": row["ay"],
            "az": row["az"],
            "gx": row["gx"],
            "gy": row["gy"],
            "gz": row["gz"],
            "speed": row["speed"]
        }])

        prediction = ai_model.predict(features)[0]

        if prediction == 1:

            ai_confirmed_detections += 1

            print("Pothole confirmed at time:", row["timestamp"])
            print(result)
            print()

    # -----------------------------
    # Evaluation metrics
    # -----------------------------

    actual = row["label"]

    features = pd.DataFrame([{
        "ax": row["ax"],
        "ay": row["ay"],
        "az": row["az"],
        "gx": row["gx"],
        "gy": row["gy"],
        "gz": row["gz"],
        "speed": row["speed"]
    }])

    prediction = ai_model.predict(features)[0]

    if prediction == 1 and actual == 1:
        true_positives += 1

    elif prediction == 1 and actual == 0:
        false_positives += 1

    elif prediction == 0 and actual == 1:
        false_negatives += 1


# -----------------------------
# Accuracy calculation
# -----------------------------

accuracy = true_positives / (true_positives + false_positives + false_negatives)


# -----------------------------
# Summary
# -----------------------------

print("\nDetection Summary")
print("------------------------")
print("Physics detections:", physics_detections)
print("AI confirmed detections:", ai_confirmed_detections)


print("\nEvaluation Metrics")
print("-------------------")
print("True Positives:", true_positives)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)
print("Accuracy:", accuracy)