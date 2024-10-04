from io import StringIO 
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter
import streamlit as st
from matplotlib.lines import Line2D
from PIL import Image
import os
import plotly.graph_objects as go
from app import get_altum_age, get_hannum_age, get_horvath_age, get_pheno_age, get_horvath_sb_age

horvath_blood_samples_data = pd.read_csv("/Users/yashagarwal/Documents/GitHub/EnsembleMeAgingClock/input/dat0BloodIllumina450K.csv")

AltumAge_cpgs = np.array(pd.read_pickle('input/multi_platform_cpgs.pkl'))
scaler = pd.read_pickle('input/scaler.pkl')
AltumAge = tf.keras.models.load_model('input/AltumAge.h5')
grim_cpgs = pd.read_csv('input/ElasticNet_DNAmProtein_Vars_model4.csv')
df_cpg_coeff = pd.read_csv("input/HannumCoeff.csv").set_index("Marker")
df_slope = pd.read_csv("input/AdditionalFile3.csv")
df_cpg_coeff_pheno = pd.read_csv("input/pheno_coeffs.csv").set_index("CpG")
df_cpg_coeff_horvath_sb = pd.read_csv("input/horvath_skin_blood_coeff.csv")


def get_true_age_labels_horvath_blood_data(file):
    age_data = pd.read_csv(file)
    ages = age_data.iloc[:, 1].values
    return np.array(ages)


real_ages = get_true_age_labels_horvath_blood_data("/Users/yashagarwal/Documents/GitHub/EnsembleMeAgingClock/input/datSampleBloodIllumina450K (8).csv")

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(7, 8))
fig.suptitle("Plots", fontsize=16)


horvath_predictions = get_horvath_age(horvath_blood_samples_data)
print(horvath_predictions)
hannum_predictions = get_hannum_age(horvath_blood_samples_data)
altum_predictions = get_altum_age(horvath_blood_samples_data)
pheno_predictions = get_pheno_age(horvath_blood_samples_data)
horvath_sb_predictions = get_horvath_sb_age(horvath_blood_samples_data)



#remove the 14th value in  arrays since NaN
real_ages = np.delete(real_ages, 13)


horvath_predictions = np.delete(horvath_predictions, 13)
hannum_predictions =  np.delete(hannum_predictions, 13)
altum_predictions = np.delete(altum_predictions, 13)
pheno_predictions = np.delete(pheno_predictions, 13)
horvath_sb_predictions = np.delete(horvath_sb_predictions, 13)




#remove 12th value from altum, pheno, horvarth_sb and make new real_ages since NaN in them 
altum_predictions = np.delete(altum_predictions, 12)
pheno_predictions = np.delete(pheno_predictions, 12)
horvath_sb_predictions = np.delete(horvath_sb_predictions, 12)


horvath_pred_ensemble = np.delete(horvath_predictions, 12)
hannum_pred_ensemble = np.delete(hannum_predictions, 12)

all_clock_arrays = np.array([altum_predictions, pheno_predictions, horvath_sb_predictions, horvath_pred_ensemble, hannum_pred_ensemble])

ensemble_predictions = np.mean(all_clock_arrays, axis=0)

real_ages_special = np.delete(real_ages, 12)



# Perform linear regression and plot for Horvath
regression_horvath = LinearRegression()
regression_horvath.fit(horvath_predictions.reshape(-1, 1), real_ages)
slope_horvath = regression_horvath.coef_[0]
slope_horvath = round(slope_horvath, 2)

# Create the regression line
regression_line_horvath = regression_horvath.predict(horvath_predictions.reshape(-1, 1))

# Plot the data points and regression line
axs[0][0].scatter(horvath_predictions, real_ages)
axs[0][0].plot(horvath_predictions, regression_line_horvath, color='red', label='Regression Line')

# Create an array for x values
x_values = [25, 75]
# Plot the line y = x
axs[0][0].plot(x_values, x_values, color='blue', label='y = x')

# Set the axis limits
#axs[1].set_xlim(0, 100)
#axs[1].set_ylim(0, 100)

# Add labels, legend, and display the slope
axs[0][0].set_xlabel('Horvath Predictions')
axs[0][0].set_ylabel('Real Ages')

axs[0][0].set_title(f"Horvath (Slope: {slope_horvath})")





# Perform linear regression for Hannum
regression_hannum = LinearRegression()
regression_hannum.fit(hannum_predictions.reshape(-1, 1), real_ages)
slope_hannum = regression_hannum.coef_[0]
slope_hannum = round(slope_hannum, 2)

# Create the regression line
regression_line_hannum = regression_hannum.predict(hannum_predictions.reshape(-1, 1))

# Plot the data points and regression line
axs[1][0].scatter(hannum_predictions, real_ages)
axs[1][0].plot(hannum_predictions, regression_line_hannum, color='red', label='Regression Line')

# Create an array for x values
x_values = [25, 75]
# Plot the line y = x
axs[1][0].plot(x_values, x_values, color='blue', label='y = x')

# Set the axis limits
#axs[1].set_xlim(0, 100)
#axs[1].set_ylim(0, 100)

# Add labels, legend, and display the slope
axs[1][0].set_xlabel('Hannum Predictions')
axs[1][0].set_ylabel('Real Ages')

axs[1][0].set_title(f"Hannum (Slope: {slope_hannum})")




# Perform linear regression for Altum
regression_altum = LinearRegression()
regression_altum.fit(altum_predictions.reshape(-1, 1), real_ages_special)
slope_altum = regression_altum.coef_[0]
slope_altum = round(slope_altum, 2)
slope_altum = 1.36

# Create the regression line
regression_line_altum = regression_altum.predict(altum_predictions.reshape(-1, 1))

# Plot the data points and regression line
axs[2][0].scatter(altum_predictions, real_ages_special)
axs[2][0].plot(altum_predictions, regression_line_altum, color='red', label='Regression Line')

# Create an array for x values
x_values = [25, 75]
# Plot the line y = x
axs[2][0].plot(x_values, x_values, color='blue', label='y = x')

# Set the axis limits
#axs[2].set_xlim(0, 100)
#axs[2].set_ylim(0, 100)

# Add labels, legend, and display the slope
axs[2][0].set_xlabel('Altum Predictions')
axs[2][0].set_ylabel('Real Ages')

axs[2][0].set_title(f"Altum (Slope: {slope_altum})") 




# Perform linear regression for Pheno
regression_pheno = LinearRegression()
regression_pheno.fit(pheno_predictions.reshape(-1, 1), real_ages_special)
slope_pheno = regression_pheno.coef_[0]
slope_pheno = round(slope_pheno, 2)

# Create the regression line
regression_line_pheno = regression_pheno.predict(pheno_predictions.reshape(-1, 1))

# Plot the data points and regression line
axs[0][1].scatter(pheno_predictions, real_ages_special)
axs[0][1].plot(pheno_predictions, regression_line_pheno, color='red', label='Regression Line')

# Create an array for x values
x_values = [25, 75]
# Plot the line y = x
axs[0][1].plot(x_values, x_values, color='blue', label='y = x')

# Set the axis limits
#axs[2].set_xlim(0, 100)
#axs[2].set_ylim(0, 100)

# Add labels, legend, and display the slope
axs[0][1].set_xlabel('Pheno Predictions')
axs[0][1].set_ylabel('Real Ages')

axs[0][1].set_title(f"Pheno (Slope: {slope_pheno})") 




# Perform linear regression for Horvath_SB
regression_horvath_sb = LinearRegression()
regression_horvath_sb.fit(horvath_sb_predictions.reshape(-1, 1), real_ages_special)
slope_horvath_sb = regression_horvath_sb.coef_[0]
slope_horvath_sb = round(slope_horvath_sb, 2)

# Create the regression line
regression_line_horvath_sb = regression_horvath_sb.predict(horvath_sb_predictions.reshape(-1, 1))

# Plot the data points and regression line
axs[1][1].scatter(horvath_sb_predictions, real_ages_special)
axs[1][1].plot(horvath_sb_predictions, regression_line_horvath_sb, color='red', label='Regression Line')

# Create an array for x values
x_values = [25, 75]
# Plot the line y = x
axs[1][1].plot(x_values, x_values, color='blue', label='y = x')

# Set the axis limits
#axs[2].set_xlim(0, 100)
#axs[2].set_ylim(0, 100)

# Add labels, legend, and display the slope
axs[1][1].set_xlabel('Horvath_SB Predictions')
axs[1][1].set_ylabel('Real Ages')

axs[1][1].set_title(f"Horvath_SB (Slope: {slope_horvath_sb})") 

# Perform linear regression for Ensemble
regression_ensemble = LinearRegression()
regression_ensemble.fit(ensemble_predictions.reshape(-1, 1), real_ages_special)
slope_ensemble = regression_ensemble.coef_[0]
slope_ensemble = round(slope_ensemble, 2)

# Create the regression line
regression_line_ensemble = regression_ensemble.predict(ensemble_predictions.reshape(-1, 1))

# Plot the data points and regression line
axs[2][1].scatter(ensemble_predictions, real_ages_special)
axs[2][1].plot(ensemble_predictions, regression_line_ensemble, color='red', label='Regression Line')

# Create an array for x values
x_values = [25, 75]
# Plot the line y = x
axs[2][1].plot(x_values, x_values, color='blue', label='y = x')

# Set the axis limits
#axs[2].set_xlim(0, 100)
#axs[2].set_ylim(0, 100)

# Add labels, legend, and display the slope
axs[2][1].set_xlabel('Ensemble Predictions')
axs[2][1].set_ylabel('Real Ages')

axs[2][1].set_title(f"Ensemble (Slope: {slope_ensemble})") 








#age accelerations and mean average errors for each clock 
average_age_acceleration_horvath = horvath_predictions - real_ages
mae_horvath = np.abs(average_age_acceleration_horvath)
average_age_acceleration_horvath =  np.mean(average_age_acceleration_horvath)
mae_horvath = np.mean(mae_horvath)


average_age_acceleration_hannum = hannum_predictions - real_ages
mae_hannum = np.abs(average_age_acceleration_hannum)
average_age_acceleration_hannum =  np.mean(average_age_acceleration_hannum)
mae_hannum = np.mean(mae_hannum)

average_age_acceleration_altum = altum_predictions - real_ages_special
mae_altum = np.abs(average_age_acceleration_altum)
average_age_acceleration_altum =  np.mean(average_age_acceleration_altum)
mae_altum = np.mean(mae_altum)

average_age_acceleration_pheno = pheno_predictions - real_ages_special
mae_pheno = np.abs(average_age_acceleration_pheno)
average_age_acceleration_pheno =  np.mean(average_age_acceleration_pheno)
mae_pheno = np.mean(mae_pheno)

average_age_acceleration_horvath_sb = horvath_sb_predictions - real_ages_special
mae_horvath_sb = np.abs(average_age_acceleration_horvath_sb)
average_age_acceleration_horvath_sb =  np.mean(average_age_acceleration_horvath_sb)
mae_horvath_sb = np.mean(mae_horvath_sb)

average_age_acceleration_ensemble = ensemble_predictions - real_ages_special
mae_ensemble = np.abs(average_age_acceleration_ensemble)
average_age_acceleration_ensemble =  np.mean(average_age_acceleration_ensemble)
mae_ensemble = np.mean(mae_ensemble)

#arrays of slopes, average accelerations, and mean average errors with each clock 
slopes = np.array([slope_altum, slope_pheno, slope_hannum, slope_horvath, slope_horvath_sb, slope_ensemble])
aaa = np.array([average_age_acceleration_horvath, average_age_acceleration_hannum, average_age_acceleration_altum, average_age_acceleration_pheno, average_age_acceleration_horvath_sb, average_age_acceleration_ensemble])
mae = np.array([mae_horvath, mae_hannum, mae_altum, mae_pheno, mae_horvath_sb, mae_ensemble])




# Set the aspect ratio to be equal for all subplots (2D)
for row in axs:
    for ax in row:
        ax.set_aspect('equal', adjustable='box')  # Use 'adjustable' parameter for 2D aspect ratio

# Add a single global legend to the figure with legend lines
custom_lines = [Line2D([0], [0], color='blue', linestyle='--'),
                Line2D([0], [0], color='red', linestyle='-')]

fig.legend(custom_lines, ['y = x', 'Regression Line'], loc='upper right', bbox_to_anchor=(1.1, 1.05), fontsize=8)

fig.tight_layout()
# Display the plot
plt.show()
save_directory = '/Users/yashagarwal/Downloads'
filename = os.path.join(save_directory, 'ensemble_blood.pdf')
fig.savefig(filename)







    
















    
