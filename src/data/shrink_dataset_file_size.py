import pandas as pd

#
# Utility script to split dataset into files <=95MB (to stay under GitHub 100MB limit)
#

# Input file
input_file = 'src/data/mldd_dataset.csv'

# Number of target splits (adjust as needed)
num_splits = 6

# Target size per file (in bytes) — leave headroom
target_size = 95 * 1024 * 1024  # 95 MB

# Read CSV
df = pd.read_csv(input_file)

# Calculate approx size of each full row (all columns, as string)
row_sizes = df.apply(lambda row: len(','.join([str(x) for x in row]).encode('utf-8')), axis=1)
total_size = row_sizes.sum()

print(f"Total estimated dataset size: {total_size / (1024 * 1024):.2f} MB")

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
    actual_file_size = split_df.to_csv(None, index=False).encode('utf-8')
    print(f"Wrote {output_file}: ~{len(actual_file_size)/(1024*1024):.2f} MB, {len(indices)} rows")
