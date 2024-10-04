import pandas as pd
import os
print(os.getcwd())

def extract_cpg_sites(file_path):
    # List of possible headers
    possible_headers = ['CpGmarker', 'var', 'Marker', 'CpG', 'ID'] #you will have to add more header column names as you discover different names

    # Load the file. If it's a .pkl file, use pd.read_pickle. Otherwise, use pd.read_csv
    if file_path.endswith('.pkl'):
        df = pd.read_pickle(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # If the data is a DataFrame, find the column that contains CpG sites
    if isinstance(df, pd.DataFrame):
        for header in df.columns:
            if header in possible_headers:
                cpg_sites = df[header]
                break
    # If the data is a Series (as in the case of the .pkl file), it directly contains CpG sites
    else:
        cpg_sites = df

    # Filter to keep only entries starting with 'cg'
    cpg_sites = cpg_sites[cpg_sites.str.startswith('cg')]

    return cpg_sites

# Define file paths as needed
altum_file = 'altum_cpgs (1).pkl'
grim_file = 'grim_coeffs (1).csv'
hannum_file = 'hannum_coeffs (1).csv'
horvath_file = 'horvath_coeffs (1).csv'
horvath_sb_file = 'horvath_skin_blood_coeff (1).csv'
pheno_file = 'pheno_coeffs (1).csv'

# Run script for each file and write them to the 'eac_folder' directory
altum_cpgs = extract_cpg_sites(altum_file)
altum_cpgs.to_csv('altum_cpgs_list.txt', index=False, header=False)

grim_cpgs = extract_cpg_sites(grim_file)
grim_cpgs.to_csv('grim_coeffs_list.txt', index=False, header=False)

hannum_cpgs = extract_cpg_sites(hannum_file)
hannum_cpgs.to_csv('hannum_coeffs_list.txt', index=False, header=False)

horvath_cpgs = extract_cpg_sites(horvath_file)
horvath_cpgs.to_csv('horvath_coeffs_list.txt', index=False, header=False)

horvath_sb_cpgs = extract_cpg_sites(horvath_sb_file)
horvath_sb_cpgs.to_csv('horvath_SB_coeff_list.txt', index=False, header=False)

pheno_cpgs = extract_cpg_sites(pheno_file)
pheno_cpgs.to_csv('pheno_coeffs_list.txt', index=False, header=False)
