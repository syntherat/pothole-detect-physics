import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Data" / "pothole_ai_model.pkl"


# Load dataset
data = pd.read_csv(BASE_DIR / "Data" / "synthetic_pothole_dataset.csv")

# Features used for AI
X = data[[
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "speed"
]]

# Labels
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train AI model
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

joblib.dump(model, MODEL_PATH)
# Evaluate model
predictions = model.predict(X_test)

print("\nAI Model Performance:\n")
print(classification_report(y_test, predictions))