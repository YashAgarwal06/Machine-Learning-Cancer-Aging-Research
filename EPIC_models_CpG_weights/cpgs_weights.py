
import pickle
import joblib
import pandas as pd

# Load the data from the .pkl file


heavy_model = joblib.load('EPIC_models_CpG_weights/EPICheavy_weights.pkl')

heavy_cpgs = pd.read_csv("EPIC_models_CpG_weights/EPICheavy_cpgs.txt", index_col=None)


# Access the coefficients
heavy_coefficients = heavy_model.coef_

heavy_coeff = []
# Print the coefficients
print("Coefficients:")
for i, coef in enumerate(heavy_coefficients):
    heavy_coeff.append([coef])
    
heavy_df_coeff = pd.DataFrame(heavy_coeff)

    
heavy_final_df = pd.concat([heavy_cpgs, heavy_df_coeff], axis=1)
heavy_final_df.columns = ["CpG" , "Weight"]




# Load the model and data
light_model = joblib.load('EPIC_models_CpG_weights/EPIClight_weights.pkl')
light_cpgs = pd.read_csv("EPIC_models_CpG_weights/EPIClight_cpgs.txt", index_col=None)

# Access the coefficients
light_coefficients = light_model.coef_

light_coeff = []
# Print the coefficients
print("Coefficients:")
for i, coef in enumerate(light_coefficients):
    light_coeff.append([coef])
    
light_df_coeff = pd.DataFrame(light_coeff)

# Combine the data
light_final_df = pd.concat([light_cpgs, light_df_coeff], axis=1)
light_final_df.columns = ["CpG" , "Weight"]
print(light_final_df)

heavy_final_df.to_csv('/Users/yashagarwal/Downloads/heavy_final_df.csv', index=False)
light_final_df.to_csv('/Users/yashagarwal/Downloads/light_final_df.csv', index=False)

    





