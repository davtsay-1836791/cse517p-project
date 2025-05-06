import pandas as pd

# 
# Utility script to split dataset into files <=100MB
# 
# Input file
input_file = 'src/data/mldd_dataset.csv'

# Number of target splits
num_splits = 6

# Target size per file (in bytes)
target_size = 100 * 1024 * 1024  # 100 MB

# Read CSV
df = pd.read_csv(input_file)

# Calculate approx size of each row (based on UTF-8 encoding)
row_sizes = df['conversations'].apply(lambda x: len(str(x).encode('utf-8')))
total_size = row_sizes.sum()

print(f"Total size in memory estimate: {total_size / (1024 * 1024):.2f} MB")

# Initialize splits
splits = [[] for _ in range(num_splits)]
split_sizes = [0] * num_splits

current_split = 0

for idx, size in row_sizes.items():
    if split_sizes[current_split] + size > target_size and current_split < num_splits - 1:
        current_split += 1
    splits[current_split].append(idx)
    split_sizes[current_split] += size

# Write out splits
for i, indices in enumerate(splits):
    split_df = df.loc[indices]
    output_file = f'src/data/mldd_split_dataset_{i + 1}.csv'
    split_df.to_csv(output_file, index=False)
    print(f"Wrote {output_file}: {split_sizes[i]/(1024*1024):.2f} MB, {len(indices)} rows")
