import pandas as pd 
import numpy as np
from app import get_altum_age, get_hannum_age, get_Han2020_age, get_horvath_age, get_horvath_sb_age, get_pheno_age, get_YingCaus_age, get_ZhangEn_age
import re
import pandas as pd
from clocks import han2020_coefs


zhangen_params = pd.read_table('clocks/zhangen.coef', sep=' ')
zhangen_params['feature'] = zhangen_params['probe']
zhangen_params['coefficient'] = zhangen_params['coef']
zhangen_cpgs = zhangen_params['feature'][1:].tolist()

yingCausAge_params = pd.read_csv('clocks/YingCausAge.csv')
yingCausAge_params['feature'] = yingCausAge_params['term']
yingCausAge_params['coefficient'] = yingCausAge_params['estimate']
yingCausAge_cpgs = yingCausAge_params['feature'][1:].tolist()

han2020_cpgs = han2020_coefs.cpgs
cpgs = pd.read_csv("tools/union_out.csv")
cpgs_list = list(set(cpgs.iloc[:, 0].tolist() + yingCausAge_cpgs + zhangen_cpgs + han2020_coefs.cpgs))

# Define the file path
file_path = '/Users/yashagarwal/Downloads/GSE134379_processedSamples_cblAD.txt'

# Initialize empty lists to store data
cg_ids = []
methylation_values = []

# Read the file line by line
with open(file_path, 'r') as file:
    for line in file:
        # Split each line by whitespace to separate CG ID and values
        data = line.strip().split()
        
        # First element is the CG ID
        cg_id = data[0]
        cg_ids.append(cg_id)
        print(cg_id)
        # Remaining elements are the methylation values
        values = list(map(float, data[1:]))  # Convert values to float
        print(values)
        methylation_values.append(values)
        

# Create a DataFrame
df = pd.DataFrame(methylation_values, columns=cg_ids)

# Display the DataFrame (optional)
print(df.head())  # Display the first few rows to verify






    





'''altum_pred = np.expand_dims(np.float32(get_altum_age(df)), axis=0)
hannum_pred = np.expand_dims(np.float32(get_hannum_age(df)), axis=0)
horvath_pred = np.expand_dims(np.float32(get_horvath_age(df)), axis=0)
pheno_pred = np.expand_dims(np.float32(get_pheno_age(df)), axis=0)
horvath_sb_pred = np.expand_dims(np.float32(get_horvath_sb_age(df)), axis=0)
pred_zhangEn = np.expand_dims(np.float32(get_ZhangEn_age(df)), axis=0)
pred_han2020 = np.expand_dims(np.float32(get_Han2020_age(df)), axis=0)
pred_yingCaus = np.expand_dims(np.float32(get_YingCaus_age(df)), axis=0)

ensembleLR_pred = round(19.275324793823557 + (0.3437195049507392 * float(altum_pred)) + (-0.20633851531241806 * float(pred_han2020)) + (-0.5173259165634722 * float(hannum_pred)) + (0.2836010547077375 * float(horvath_pred)) + (0.1857810346592119 * float(horvath_sb_pred)) + (-0.02208310995460718 * float(pheno_pred)) + (-0.08526975203798473 * float(pred_yingCaus)) + (0.5640149630320757 * float(pred_ensembleLR_pred = round(19.275324793823557 + (0.3437195049507392 * float(altum_pred)) + (-0.20633851531241806 * float(pred_han2020)) + (-0.5173259165634722 * float(hannum_prediction)) + (0.2836010547077375 * float(horvath_prediction)) + (0.1857810346592119 * float(horvath_sb_pred)) + (-0.02208310995460718 * float(pheno_prediction)) + (-0.08526975203798473 * float(yingCausal_pred)) + (0.5640149630320757 * float(pred_zhangEn)), 2)
)), 2)


results = np.concatenate(np.mean([altum_pred, hannum_pred, horvath_pred, pheno_pred, horvath_sb_pred, pred_zhangEn, pred_han2020, pred_yingCaus],axis=0), ).T
        #print(results.shape)
columns = ['EnsembleAge','HannumClock', 'HorvathClock', 'PhenoAge', 'HorvathSkinBlood', 'AltumAge', "ZhangEn", "Han2020", "YingCausal"]
df = pd.DataFrame(data = results, columns = columns, index=df.columns[1:])    
def convert_df(df):
        return df.to_csv(index=True).encode('utf-8')

csv = convert_df(df)'''


