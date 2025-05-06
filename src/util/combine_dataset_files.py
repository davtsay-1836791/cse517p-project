import pandas as pd
import glob

# Output combined file name
output_file = 'output/mldd_dataset.csv'

# Find all split files in order
split_files = sorted(glob.glob('mldd_split_dataset_*.csv'))

print(f"Files to combine: {split_files}")

# Read and concatenate all splits
dfs = []
for file in split_files:
    df = pd.read_csv(file)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# Write combined CSV
combined_df.to_csv(output_file, index=False)

print(f"Combined file written to {output_file} with {len(combined_df)} rows")
