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
import plotly.subplots as sp

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
fig, axs = plt.subplots(3, 3, figsize=(9, 9))
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

# Perform linear regression for Altum
regression_altum = LinearRegression()
regression_altum.fit(altum_predictions.reshape(-1, 1), real_ages_special)
slope_altum = regression_altum.coef_[0]
slope_altum = f"{slope_altum:.2f}"
slope_altum = 1.36
 

regression_hannum = LinearRegression()
regression_hannum.fit(hannum_predictions.reshape(-1, 1), real_ages)
slope_hannum = regression_hannum.coef_[0]
slope_hannum = round(slope_hannum, 2)

# Perform linear regression for Pheno
regression_pheno = LinearRegression()
regression_pheno.fit(pheno_predictions.reshape(-1, 1), real_ages_special)
slope_pheno = regression_pheno.coef_[0]
slope_pheno = round(slope_pheno, 2)


# Perform linear regression and plot for Horvath
regression_horvath = LinearRegression()
regression_horvath.fit(horvath_predictions.reshape(-1, 1), real_ages)
slope_horvath = regression_horvath.coef_[0]
slope_horvath = round(slope_horvath, 2)


# Perform linear regression for Horvath_SB
regression_horvath_sb = LinearRegression()
regression_horvath_sb.fit(horvath_sb_predictions.reshape(-1, 1), real_ages_special)
slope_horvath_sb = regression_horvath_sb.coef_[0]
slope_horvath_sb = round(slope_horvath_sb, 2)

regression_ensemble = LinearRegression()
regression_ensemble.fit(ensemble_predictions.reshape(-1, 1), real_ages_special)
slope_ensemble = regression_ensemble.coef_[0]
slope_ensemble = round(slope_ensemble, 2)

#arrays of slopes, average accelerations, and mean average errors with each clock 
slopes = np.array([slope_altum, slope_pheno, slope_hannum, slope_horvath, slope_horvath_sb, slope_ensemble])
aaa = np.array([average_age_acceleration_horvath, average_age_acceleration_hannum, average_age_acceleration_altum, average_age_acceleration_pheno, average_age_acceleration_horvath_sb, average_age_acceleration_ensemble])
mae = np.array([mae_horvath, mae_hannum, mae_altum, mae_pheno, mae_horvath_sb, mae_ensemble])

graph_metrics = ['Slopes', 'AAA', 'MAE']

metrics = {}
metrics["Slopes"] = slopes
metrics["AAA"] = aaa
metrics["MAE"] = mae
print(metrics)

# Define the slopes and labels
clock_labels = ['Hannum', 'Horvath', 'Pheno', 'Skin Blood', 'AltumAge', 'EnsembleAge']

metric = 'Age Acceleration'
title = 'Blood Data Clock Comparisons'
def draw_plot(arr):
    data = []
    for metric in graph_metrics:
        data.append(arr[metric])
    data = np.array(data)
    print(data)
    layout = go.Layout(
        paper_bgcolor='white',  # Set the background color to white
        plot_bgcolor='white',   # Set the plot area background color to white
        width=1200,               # Set the width of the entire plot (in pixels)
        height=1200,              # Set the height of the entire plot (in pixels)
        #title= title,
        xaxis=dict(title='graph_metrics'),
        yaxis=dict(title=metric),
        boxgap=0.5,       # Adjust the box gap
        boxgroupgap=0.5 ,  # Adjust the box group gap
        shapes=[
            # Rectangle shape for the plot area outline
            dict(
                type='rect',
                xref='paper',  # The x reference is set to 'paper' for relative coordinates
                yref='paper',  # The y reference is set to 'paper' for relative coordinates
                x0=0.02,          # x-coordinate of the lower left corner
                y0=0.02,          # y-coordinate of the lower left corner
                x1=0.98,          # x-coordinate of the upper right corner
                y1=0.98,          # y-coordinate of the upper right corner
                line=dict(
                    color='black',  # Border color (you can use any valid color name or HEX code)
                    width=1         # Border width in pixels
                )
            )
        ]
    )

    # Create the box plot
    fig = go.Figure(layout=layout)

    

    

    for i, values in enumerate(data):
        point_colors = np.random.rand(7, 3)
        fig.add_trace(go.Box(y=values, name=graph_metrics[i], showlegend=False, boxpoints=False, line=dict(color='gray')))
    

    point_colors = ['red', 'green', 'blue', 'orange', 'pink', 'magenta', 'cyan']
    y_coords = []
    print(len(graph_metrics))
    for i in range(len(graph_metrics)):
        for j in range(len(clock_labels)):
            if j != len(clock_labels) - 1:
                symbol = ['circle']
                size=8
                width=0
            else:
                symbol = ['diamond']
                size=11
                width=0
                
            if i == 0:
                fig.add_trace(go.Scatter(
                    x=[graph_metrics[i]],
                    y=[arr[graph_metrics[i]][j]],
                    mode='markers',
                    marker=dict(color=[point_colors[j]], symbol=symbol, size=size, line=dict(width=width)),
                    name=clock_labels[j],
                    showlegend=True
                ))
                y_coords.append(arr[graph_metrics[i]][j])
            else:
                fig.add_trace(go.Scatter(
                    x=[graph_metrics[i]],
                    y=[arr[graph_metrics[i]][j]],
                    mode='markers',
                    marker=dict(color=[point_colors[j]], symbol=symbol, size=size, line=dict(width=width)),
                    name=clock_labels[j],
                    showlegend=False
                ))
                y_coords.append(arr[graph_metrics[i]][j])    

    fig.show()
draw_plot(metrics)

