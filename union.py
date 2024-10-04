import pandas as pd
import os

# Directory where the files are stored
directory = "tools"

# List of the paths of the files to union
files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

# Initialize an empty list to store the CpG sites from all files
all_cpg_sites = []

# Load the CpG sites from each file and add them to the list
for file_path in files:
    cpg_sites = pd.read_csv(file_path, header=None, squeeze=True)
    all_cpg_sites.append(cpg_sites)

# Concatenate all the CpG sites together into a single Series
all_cpg_sites = pd.concat(all_cpg_sites)

# Drop duplicates to get the unique CpG sites
unique_cpg_sites = all_cpg_sites.drop_duplicates()

# Save the unique CpG sites to a CSV file
unique_cpg_sites.to_csv("cleans_and_unions/union_out.csv", index=False, header=False)
