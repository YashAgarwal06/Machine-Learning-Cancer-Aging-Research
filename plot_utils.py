import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LinearRegression
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import median_absolute_error
from app import get_altum_age, get_hannum_age, get_Han2020_age, get_horvath_age, get_horvath_sb_age, get_pheno_age, get_YingCaus_age, get_ZhangEn_age

def get_mae(path, organs=['Blood', 'Breast', 'Kidney', 'Lung', 'Muscle', 'Ovary', 'Prostate', 'Testis', 'Colon']):
    errors = {}
    df = pd.read_csv(path, index_col='Organ')
    for organ in organs:
        array = df.loc[organ].to_numpy()
        # Calculate the number of prediction columns (m)
        m = array.shape[1] - 1  # Number of columns minus the last column (ground truth)

        # Initialize an array to store absolute errors for each row
        absolute_errors = np.zeros((array.shape[0], m))

        # Compute absolute errors for each row
        for i in range(array.shape[0]):
            predictions = array[i, :m]  # First m columns are predictions
            ground_truth = array[i, -1]  # Last column is ground truth
            
            # Compute absolute errors
            absolute_errors[i] = np.abs(predictions - ground_truth)
        errors[organ] = np.mean(absolute_errors, axis=0).tolist()
    return errors

def get_aa(path, organs=['Blood', 'Breast', 'Kidney', 'Lung', 'Muscle', 'Ovary', 'Prostate', 'Testis', 'Colon']):
    errors = {}
    df = pd.read_csv(path, index_col='Organ')
    for organ in organs:
        array = df.loc[organ].to_numpy()
        # Calculate the number of prediction columns (m)
        m = array.shape[1] - 1  # Number of columns minus the last column (ground truth)

        # Initialize an array to store absolute errors for each row
        absolute_errors = np.zeros((array.shape[0], m))

        # Compute absolute errors for each row
        for i in range(array.shape[0]):
            predictions = array[i, :m]  # First m columns are predictions
            ground_truth = array[i, -1]  # Last column is ground truth
            
            # Compute absolute errors
            absolute_errors[i] = predictions - ground_truth
        errors[organ] = np.mean(absolute_errors, axis=0).tolist()
    return errors


def draw_error_chart(data_path):
    metric = 'Median Average Error'
    clock_labels = ['Altum', 'Han2020', 'Hannum', 'Horvath', 'Skin Blood', 'Pheno', 'YingCausal', 'ZhangEn', 'EnsembleAge']
    organs = ['Whole Blood', 'Breast', 'Kidney', 'Lung', 'Muscle', 'Ovary', 'Prostate', 'Testis', 'Colon', 'All Data']
    data = []
    color_labels = ['red', 'green', 'blue', 'orange', 'purple', 'magenta', 'cyan', 'black', 'yellow']
    arr = {clock_labels[i]: [row[i] for row in get_mae(data_path).values()] for i in range(len(clock_labels))}
    for clock in clock_labels:
        data.append(arr[clock])
    data = np.array(data)

    ensemble_age_index = clock_labels.index('EnsembleAge')
    organs = [x for _,x in sorted(zip(data[:, ensemble_age_index], organs))]
    layout = go.Layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=1000,
        height=800,
        xaxis=dict(title='Clocks'),
        yaxis=dict(title=metric),
        shapes=[
            dict(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color='black', width=1)
            )
        ]
    )

    # Create figure object
    fig = go.Figure(layout=layout)

    # Loop through each row of the data array and create scatter traces
    for i in range(len(data)):

        marker_symbol = 'circle-open'
        
        fig.add_trace(go.Scatter(
            x=clock_labels,
            y=data[i],
            mode='lines+markers',
            name=organs[i],
            line=dict(color=color_labels[i]),  # Assigning a specific color from color_labels
            marker=dict(symbol=marker_symbol, size=10, color=color_labels[i])
        ))

    fig.show()

def draw_accel_plot(data_path):
    metric = 'Age Acceleration'
    clock_labels = ['Altum', 'Han2020', 'Hannum', 'Horvath', 'Skin Blood', 'Pheno', 'YingCausal', 'ZhangEn', 'EnsembleAge']

    point_colors = ['red', 'green', 'blue', 'orange', 'purple', 'magenta', 'lime', 'navy', 'brown', 'cyan']
    organs = ['Whole Blood', 'Breast', 'Kidney', 'Lung', 'Muscle', 'Ovary', 'Prostate', 'Testis', 'Colon']
    data = []
    
    arr = {clock_labels[i]: [row[i] for row in get_aa(data_path).values()] for i in range(len(clock_labels))}
    for organ in clock_labels:
        data.append(arr[organ])
    data = np.array(data)
    layout = go.Layout(
        paper_bgcolor='white',  # Set the background color to white
        plot_bgcolor='white',   # Set the plot area background color to white
        width=1000,               # Set the width of the entire plot (in pixels)
        height=800,              # Set the height of the entire plot (in pixels)
        #title= title,
        xaxis=dict(title='Clocks'),
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
            ),
            #dict(
            #    type='line',
            #    xref='paper',
            #    x0=0.02,  # You can set the x-coordinate range for the line if needed
            #    x1=0.98,
            #    y0=0,
            #    y1=0,
            #    line=dict(color='red', width=2),  # Set the color and width of the line
            #)
        ]
    )

    # Create the box plot
    fig = go.Figure(layout=layout)

    

    for i, values in enumerate(data):
        fig.add_trace(go.Box(y=values, name=clock_labels[i], showlegend=False, boxpoints=False, line=dict(color='gray')))
    

    
    y_coords = []
    for i in range(len(clock_labels)):
        for j in range(len(organs)):
            # if j != len(organs) - 1:
            symbol = ['circle']
            size=7
            width=2
            # else:
            #     symbol = ['diamond']
            #     size=9
            #     width=2
                
            if i == 0:
                fig.add_trace(go.Scatter(
                    x=[clock_labels[i]],
                    y=[arr[clock_labels[i]][j]],
                    mode='markers',
                    marker=dict(color='rgba(0, 0, 0, 0)',symbol=symbol, size=size, line=dict(width=width, color=[point_colors[j]])),
                    name=organs[j],
                    showlegend=True,
                ))
                y_coords.append(arr[clock_labels[i]][j])
            else:
                fig.add_trace(go.Scatter(
                    x=[clock_labels[i]],
                    y=[arr[clock_labels[i]][j]],
                    mode='markers',
                    marker=dict(color='rgba(0, 0, 0, 0)',symbol=symbol, size=size, line=dict(width=width, color=[point_colors[j]])),
                    name=organs[j],
                    showlegend=False,
                ))
                y_coords.append(arr[clock_labels[i]][j]) 

    fig.update_layout(legend=dict(x=1.02, y=0.96, font=dict(size=14)))
    fig.update_xaxes(tickfont=dict(size=20), titlefont=dict(size=25)) 
    fig.update_yaxes(tickfont=dict(size=20), titlefont=dict(size=25)) 
    fig.show()

def draw_plot_matrix(data_path, output_path):
    df = pd.read_csv(data_path, index_col='Organ')
    clocks = ['altum', 'han', 'hannum', 'horvath', 'horvath_SB', 'pheno', 'ying', 'zhang', 'EnsembleAge']
    organs = ['Blood', 'Breast', 'Colon', 'Kidney', 'Lung', 'Muscle', 'Ovary', 'Prostate', 'Testis']
    for j in range(len(clocks)):
            for i in range(len(organs)):
                pred = df.loc[organs[i]].to_numpy()[:,j]
                truth = df.loc[organs[i]].to_numpy()[:,len(clocks)]
                plt.plot(truth, pred, 'o', color='black', mfc='none', markersize=10, markeredgewidth=5)
                lin_reg = LinearRegression(fit_intercept=True).fit(truth.reshape(-1,1), pred.reshape(-1,1))
                plt.plot(np.arange(0,100), (np.arange(0,100) * lin_reg.coef_[0]) + lin_reg.intercept_, color='red', label='slope = ' + str(round(float(lin_reg.coef_[0]), 2)), linewidth=5)
                plt.plot(np.arange(0,100), np.arange(0,100), color='blue', label='y=x', linewidth=5)
                plt.title(clocks[j] + ' on ' + organs[i] + ' (N = ' + str(len(pred))+ ')')
                plt.xlim(0,100)
                plt.ylim(0,100)
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.tick_params(axis='both', which='major', length=20, width=10)
                plt.gca().spines['left'].set_linewidth(10) 
                plt.gca().spines['bottom'].set_linewidth(10) 
                plt.gca().spines['top'].set_linewidth(10) 
                plt.gca().spines['right'].set_linewidth(10) 
                plt.legend(fontsize=10, loc='upper left')
                #plt.xlabel("Chronological Age", fontsize=10)
                #plt.ylabel("Predicted Age", fontsize=10)
                plt.savefig('plots_dir/' + str(i) + '_' + str(j) + '.png')
                plt.clf()

    # Dimensions of each image (assuming they are all the same size)
    image_width = 640  # Adjust according to your image size
    image_height = 480  # Adjust according to your image size

    # Number of rows and columns in the grid
    rows = len(organs)  # 0 to 8 inclusive
    cols = len(clocks)  # 0 to 7 inclusive

    # Create a blank canvas for the final image grid
    final_image = Image.new('RGB', (cols * image_width, rows * image_height))

    # Iterate over each image and paste it into the final grid
    for n in range(rows):
        for m in range(cols):
            # Load each image
            image_path = f"plots_dir/{n}_{m}.png"  # Replace with your actual directory path
            img = Image.open(image_path)
            
            # Calculate the position to paste the current image
            x_offset = m * image_width
            y_offset = n * image_height
            
            # Paste the current image into the final image grid
            final_image.paste(img, (x_offset, y_offset))

    # Save the final image grid
    final_image.save(output_path)  # Replace with your desired save path


plot_matrix_output_path = "plots_dir/grid_image.png"
data_path = 'all_clocks_res.csv'
#draw_error_chart(data_path)
#draw_accel_plot(data_path)
draw_plot_matrix(data_path, plot_matrix_output_path)
