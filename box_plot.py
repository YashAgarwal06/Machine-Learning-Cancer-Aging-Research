import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import plotly.graph_objects as go


organs = ['Whole Blood', 'Breast', 'Kidney', 'Lung', 'Muscle', 'Ovary', 'Prostate', 'Testis', 'Colon', 'All Data']

slopes = {}
errors = {}
accels = {}
clock_labels = ['Hannum', 'Horvath', 'Pheno', 'Skin Blood', 'GrimAge', 'AltumAge', 'EnsembleAge']

slopes["Whole Blood"] = [0.89, 0.92, 1.01, 1.15, 0.78, 1.72, 1.08]
slopes["Breast"] = [0.33, 0.51, 0.48, 0.68, 0.66, 1.2, 0.64]
slopes["Kidney"] = [0.48, 0.69, 0.53, 1.12, 0.76, 1.48, 0.84]
slopes["Lung"] = [0.56, 0.68, 0.64, 0.95, 0.7, 1.24, 0.8]
slopes["Muscle"] = [0.22, 0.31, 0.2, 0.39, 0.66, 1.33, 0.52]
slopes["Ovary"] = [0.19, 0.28, 0.12, 0.44, 0.64, 0.38, 0.34]
slopes["Prostate"] = [0.38, 0.67, 0.68, 0.91, 0.74, 1.35, 0.79]
slopes["Testis"] = [0.25, 0.34, 0.4, 0.46, 0.7, 0.71, 0.48]
slopes["Colon"] = [0.46, 0.72, 0.53, 0.84, 0.71, 1.58, 0.81]
slopes["All Data"] = [0.51, 0.64, 0.57, 0.82, 0.72, 1.27, 0.76]

errors["Whole Blood"] = [16.57, 7.03, 4.98, 10.6, 18.54, 13.24, 3.19]
errors["Breast"] = [39.85, 7.31, 28.23, 15.8, 19.33, 14.72, 8.24]
errors["Kidney"] = [32.35, 11.77, 38.51, 3.74, 14.39, 8.38, 10.46]
errors["Lung"] = [21.21, 10.12, 29.83, 13.47, 15.81, 20.0, 8.33]
errors["Muscle"] = [57.34, 17.05, 57.52, 24.69, 6.23, 8.52, 25.53]
errors["Ovary"] = [54.37, 28.25, 62.47, 16.03, 11.0, 36.49, 31.63]
errors["Prostate"] = [34.07, 9.81, 31.43, 12.37, 9.47, 10.06, 14.46]
errors["Testis"] = [57.59, 18.95, 44.63, 39.7, 4.56, 30.13, 29.92]
errors["Colon"] = [26.67, 7.19, 21.48, 10.86, 21.49, 8.75, 6.71]
errors["All Data"] = [31.32, 10.78, 33.04, 13.22, 14.95, 16.47, 11.06]

accels["Whole Blood"] = [-16.26, -6.86, -0.5, 11.34, 18.57, -9.43, -0.52]
accels["Breast"] = [-39.44, -5.53, -28.49, 12.6, 20.1, -11.3, -8.68]
accels["Kidney"] = [-32.13, -11.29, -38.02, 2.93, 14.12, 5.73, -9.77]
accels["Lung"] = [-21.22, -9.84, -29.71, 13.22, 16.11, -18.52, -8.33]
accels["Muscle"] = [-56.2, -16.27, -55.64, -24.9, 6.53, -6.28, -25.46]
accels["Ovary"] = [-53.15, -26.39, -60.18, -13.67, 11.71, -35.63, -29.55]
accels["Prostate"] = [-33.53, -9.39, -31.24, -12.18, 10.2, -8.32, -14.08]
accels["Testis"] = [-56.13, -17.8, -44.01, -38.11, 4.32, -27.94, -29.95]
accels["Colon"] = [-26.07, -7.3, -21.84, 10.26, 21.57, -7.1, -5.08]
accels["All Data"] = [-33.84, -12.35, -33.9, -0.16, 15.07, -15.29, -13.41]

metric = 'Age Acceleration'
#title = 'Median Average Error Comparison'
def draw_plot(arr):
    data = []
    for organ in organs:
        data.append(arr[organ])
    data = np.array(data)
    layout = go.Layout(
        paper_bgcolor='white',  # Set the background color to white
        plot_bgcolor='white',   # Set the plot area background color to white
        width=1200,               # Set the width of the entire plot (in pixels)
        height=1200,              # Set the height of the entire plot (in pixels)
        #title= title,
        xaxis=dict(title='Organs'),
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
        fig.add_trace(go.Box(y=values, name=organs[i], showlegend=False, boxpoints=False, line=dict(color='gray')))
    

    point_colors = ['red', 'green', 'blue', 'orange', 'pink', 'magenta', 'cyan']
    y_coords = []
    for i in range(len(organs)):
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
                    x=[organs[i]],
                    y=[arr[organs[i]][j]],
                    mode='markers',
                    marker=dict(color=[point_colors[j]], symbol=symbol, size=size, line=dict(width=width)),
                    name=clock_labels[j],
                    showlegend=True
                ))
                y_coords.append(arr[organs[i]][j])
            else:
                fig.add_trace(go.Scatter(
                    x=[organs[i]],
                    y=[arr[organs[i]][j]],
                    mode='markers',
                    marker=dict(color=[point_colors[j]], symbol=symbol, size=size, line=dict(width=width)),
                    name=clock_labels[j],
                    showlegend=False
                ))
                y_coords.append(arr[organs[i]][j])    

    fig.show()
draw_plot(accels)

