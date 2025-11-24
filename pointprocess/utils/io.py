import yaml
import os
import csv

def load_config(path="config.yaml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)
        
def save_results_to_csv(result, csv_path):
    """
    Append one result dictionary to a CSV file,
    creating the directory and header if needed.
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "generator",
        "method",
        "alpha_level",
        "M",
        "KS",
        "CvM",
        "AD",
        "time_seconds",
    ]

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)
