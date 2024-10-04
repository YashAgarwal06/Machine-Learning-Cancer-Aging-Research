import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Assuming you have a DataFrame called df with missing values
# Example:
df = pd.read_csv('stem_cell_samples/stem_sample_clean.csv')

print('1')
# Separate methylation values and labels
labels = df.iloc[:, 0]  # Assuming the first column contains the CPG labels
methylation_data = df.iloc[:, 1:]  # Assuming the methylation data starts from the second column


# Impute missing values using Linear Regression
imputer_simple = SimpleImputer(strategy='mean')  # You can use other strategies like 'median' or 'most_frequent'
imputed_data_simple = imputer_simple.fit_transform(methylation_data)

# Impute missing values using IterativeImputer
imputer_iterative = IterativeImputer(max_iter=100, random_state=0)  # You can adjust max_iter as needed
imputed_data_iterative = imputer_iterative.fit_transform(methylation_data)


# Convert imputed data back to DataFrame
# Create the DataFrame
imputed_df_simple = pd.DataFrame(imputed_data_simple, columns=methylation_data.columns)
imputed_df_iterative = pd.DataFrame(imputed_data_iterative, columns=methylation_data.columns)

# Add the labels as the first column
imputed_df_simple.insert(0, 'Labels', labels)
imputed_df_iterative.insert(0, 'Labels', labels)


#print(imputed_df)
print("Original DataFrame:")
#df.to_csv('/Users/yashagarwal/Downloads/original_data.csv', index=False)
#print(df)
print("\nImputed DataFrame:")
imputed_df_simple.to_csv('/Users/yashagarwal/Downloads/stem_imputed_data_simple.csv', index=False)
imputed_df_iterative.to_csv('/Users/yashagarwal/Downloads/100_stem_imputed_data_iterative.csv', index=False)

#print(imputed_df)