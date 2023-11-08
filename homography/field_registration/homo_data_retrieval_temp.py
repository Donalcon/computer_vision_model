import pickle
import pandas as pd
import numpy as np
import csv

# Unpickle dataset_dict_stacked
with open('dataset_dict_stacked.pkl', 'rb') as f:
    dataset_dict_stacked = pickle.load(f)

# Print the content types and shapes
for key, value in dataset_dict_stacked.items():
    print(f"{key}: {type(value)}, Shape: {np.array(value).shape}")

# Create DataFrame from dataset_dict_stacked
df = pd.DataFrame.from_dict(dataset_dict_stacked)

# Assuming 'homography' and 'meta' are keys in dataset_dict_stacked:
# Extract homographies and file names
homographies = [h[0] for h in df["homography"]]
file_names = [meta['file_name'] for meta in df["meta"]]

# Write to CSV file
csv_file = "homographies.csv"
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_name", "homography_matrix"])  # Writing header

    for idx, file_name in enumerate(file_names):
        homography_str = str(homographies[idx])
        writer.writerow([file_name, homography_str])

print(f"CSV file '{csv_file}' written successfully.")
