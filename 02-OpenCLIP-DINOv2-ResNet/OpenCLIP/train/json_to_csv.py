import csv
import json
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python json_to_csv.py <json_file>")
    sys.exit(1)

json_file = sys.argv[1]
csv_file = os.path.splitext(json_file)[0] + ".csv"

if not os.path.exists(json_file):
    print(f"Error: File {json_file} does not exist.")
    sys.exit(1)

with open(json_file, "r") as f:
    data = json.load(f)

if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
    print("Error: JSON file should contain a list of dictionaries.")
    sys.exit(1)

if len(data) == 0:
    print("Warning: JSON file is empty.")
    headers = []
else:
    headers = list(data[0].keys())

with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(data)

print(f"Successfully converted {json_file} to {csv_file}")
