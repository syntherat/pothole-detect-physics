import numpy as np
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


# Constants
g = 9.81
sampling_rate = 400
dt = 1 / sampling_rate

# Total simulation time
total_time = 20
samples = int(total_time * sampling_rate)

timestamps = np.arange(0, total_time, dt)

# Normal driving noise
ax = np.random.normal(0, 0.2, samples)
ay = np.random.normal(0, 0.2, samples)
az = np.random.normal(g, 0.2, samples)

gx = np.random.normal(0, 0.02, samples)
gy = np.random.normal(0, 0.02, samples)
gz = np.random.normal(0, 0.02, samples)

speed = np.random.uniform(10, 18, samples)

# Labels
labels = np.zeros(samples)

# Insert pothole events
num_potholes = 6

for _ in range(num_potholes):

    start = np.random.randint(500, samples - 500)

    # Drop phase
    az[start:start+5] = 5

    # Free fall
    az[start+5:start+10] = 0.5

    # Impact spike
    az[start+10:start+12] = 22

    labels[start:start+12] = 1

# Create dataset
data = pd.DataFrame({
    "timestamp": timestamps,
    "ax": ax,
    "ay": ay,
    "az": az,
    "gx": gx,
    "gy": gy,
    "gz": gz,
    "speed": speed,
    "label": labels
})

# Save dataset
data.to_csv(BASE_DIR / "Data" / "synthetic_pothole_dataset.csv", index=False)

print("Dataset generated successfully!")
print("Total samples:", samples)
print("Potholes inserted:", num_potholes)