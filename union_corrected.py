import pandas as pd
import os

# List of the paths of the specific input files to union
files = [
    "altum_cpgs_list.txt",
    "grim_coeffs_list.txt",
    "hannum_coeffs_list.txt",
    "horvath_coeffs_list.txt",
    "horvath_SB_coeff_list.txt",
    "pheno_coeffs_list.txt",
]

# Initialize an empty set to store the CpG sites from all files
all_cpg_sites = set()

# Function to read CpG sites from a file and convert them to a set
def read_cpg_sites(file_path):
    with open(file_path, 'r') as file:
        cpg_sites = set(line.strip() for line in file.readlines() if line.startswith("cg"))
    print(f"Sample CpG sites from {file_path}: {list(cpg_sites)[:5]}")
    return cpg_sites

# Load the CpG sites from each file and add them to the set
for file_path in files:
    cpg_sites = read_cpg_sites(file_path)
    print(f"Loading CpG sites from {file_path}. Number of CpG sites: {len(cpg_sites)}")
    all_cpg_sites = all_cpg_sites.union(cpg_sites)

# Convert the set to a pandas Series
unique_cpg_sites = pd.Series(list(all_cpg_sites))

# Inspect the resulting unique CpG sites
print(f"Total unique CpG sites: {len(unique_cpg_sites)}")
print("Sample unique CpG sites:")
print(unique_cpg_sites.head(10))

# Save the unique CpG sites to a CSV file: change to your preferred path
output_path = "/Users/josephanthony/Desktop/eac_folder/union_out_corrected.csv"
unique_cpg_sites.to_csv(output_path, index=False, header=False)
