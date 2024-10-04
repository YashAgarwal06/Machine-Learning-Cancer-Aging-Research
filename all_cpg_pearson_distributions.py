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
import pandas as pd
from IPython.display import display
from plotly.subplots import make_subplots
from matplotlib.ticker import FuncFormatter


samples = [54, 52, 50, 223, 47, 164, 123, 50, 224]

df = pd.read_csv("input/PearsonCoeffEnsemble.csv").set_index("cpgs")
df = df.iloc[:, 1:]
df = pd.DataFrame(df)
# Get the columns of the DataFrame
columns = df.columns.tolist()

# Move the last column to the first position
columns = [columns[-1]] + columns[:-1]

# Reorder the DataFrame columns
df = df[columns]


#Create subplots for each column
num_columns = len(df.columns)
num_rows = 3
num_cols = 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 4))


# Flatten the axes for easier indexing
axes = axes.flatten()

# Iterate through columns and create histograms
# Define custom colors
color_map = ['royalblue', 'lightsteelblue', 'lightcoral', 'indianred', 'limegreen', 'palegreen', 'khaki', 'gold']

#custom_xticks = np.array([-1, -0.5] + list(np.arange(-0.5, 0.6, 0.1)) + [1])
'''custom_xticks = np.array(list(np.arange(-1, 1.1, 0.1)))
# Customize the x tick labels
new_xtick_labels = []

for label in custom_xticks:
    if label == 0:
        new_xtick_labels.append("0")
        
    elif abs(label) < 1:
        formatted_label = '{:.1f}'.format(label).replace('0.', '.')
        new_xtick_labels.append(formatted_label)
    else:
        new_xtick_labels.append('{:.0f}'.format(label))
        
new_xtick_labels[6] = 0


# Format the tick labels to display as ".1" instead of "0.1" and remove leading "0"
formatted_xtick_labels = [("{:.1f}".format(x).lstrip("0") if x != 0 else "0") for x in custom_xticks]'''
for idx, column_name in enumerate(df.columns[1:]):
    ax = axes[idx]
    values = df[column_name]
    #ax.hist(values, bins=np.arange(-1, 1.1, 0.1), color='blue', alpha=0.7, edgecolor='black')
    ax.hist(values[(values >= -0.5) & (values <= 0.5)], bins=np.arange(-1, 1.1, 0.1), color= 'lightgray', alpha=0.7, edgecolor='black', width = 0.1)
    
    title = column_name.replace("pearson coeff", "")
    title = title + "(N = " + str(samples[idx]) + ")"
    ax.set_title(title, fontsize=10)  # Adjust title font size
    
    ax.set_xlabel('Value', fontsize=8)  # Adjust label font size
    ax.set_ylabel('Frequency', fontsize=8)  # Adjust label font size
    ax.set_xlim(-1, 1)  # Set x-axis range
    ax.set_ylim(0, 8000)  # Set y-axis range
    #ax.set_xticks(np.arange(-0.5, 0.6, 0.1))  # 0.1 increments from -0.5 to 0.5
    #ax.set_xticks(np.arange(-1, -0.4, 0.5), minor=True)  # 0.5 increments from -1 to -0.5 (minor ticks)
    #ax.set_xticks(np.arange(0.5, 1.1, 0.5), minor=True)  # 0.5 increments from 0.5 to 1 (minor ticks)
    #ax.set_xticks(np.arange(-1, 1.1, 0.2))  # Set x-axis ticks
    ax.tick_params(labelsize=5)  # Adjust tick label font size
    ax.set_xticks(np.arange(-1, 1.1, 0.1))
    #ax.set_xticklabels(new_xtick_labels)  # Set formatted tick labels
    #ax.set_xticklabels(new_xtick_labels)


    
    
    #ax.legend()

    

# Hide any empty subplots
for idx in range(num_columns, num_rows * num_cols):
    axes[idx].axis('off')

plt.savefig("pearson_distribution.pdf")

plt.savefig("pearson_distribution.png")

plt.tight_layout()  # Adjust spacing between subplots
plt.show()
