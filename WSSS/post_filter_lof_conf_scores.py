import json
import os

# File paths
input_file = 'result/sem_seg/instances_predictions.json'
output_file = 'result/sem_seg/filtered_predictions.json'

# Set your confidence threshold (e.g., 0.5 means keep 50% confidence and above)
confidence_threshold = 0.50

print("Loading massive JSON file... (this might take a moment)")
with open(input_file, 'r') as f:
    predictions = json.load(f)

print(f"Original total predictions: {len(predictions)}")

# Filter the list: keep only items where the 'score' is above the threshold
# We use .get('score', 1.0) just in case some entries don't have a score key
filtered_preds = [
    pred for pred in predictions
    if pred.get('score', 1.0) >= confidence_threshold
]

print(f"Filtered total predictions: {len(filtered_preds)}")
print(f"Removed {len(predictions) - len(filtered_preds)} low-confidence masks.")

print(f"Saving to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(filtered_preds, f)

print("Done! You can now run COCOeval on the filtered file.")