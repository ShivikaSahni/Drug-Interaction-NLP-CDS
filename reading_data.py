import csv

# Replace 'your_large_file.csv' with the actual path to your CSV file
file_path = 'NOTEEVENTS.csv'

row_count = 0

with open(file_path, "r") as file:
    reader = csv.reader(file)
    for _ in reader:
        row_count += 1

print(f"Total Rows (including header): {row_count}")
print(f"Total Data Rows (excluding header): {row_count - 1}")