import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

def check_directory_name(target_name) -> bool:
    """
    Check if the current directory name matches the target_name.
    If not, move up a directory and repeat the check.
    
    Args:
        target_name (str): The directory name to match.
        
    Returns:
        bool: True if the current directory name matches the target_name, False otherwise.
    """
    # Get the current directory path
    current_dir = os.getcwd()
    
    # Extract the directory name from the path
    current_dir_name = os.path.basename(current_dir)
    
    # Check if the current directory name matches the target_name
    if current_dir_name == target_name:
        print(f'Directory set to {current_dir}, matches target dir sting {target_name}.')
        return True
    else:
        # Move up a directory
        os.chdir('..')
        # Check if we have reached the root directory
        if os.getcwd() == current_dir:
            return False
        # Recursively call the function to check the parent directory
        return check_directory_name(target_name)

def apply_default_styling(fig, title, xaxis_title, yaxis_title, 
                          legend_title=None, 
                          labels=None):
    """ Function to update layout with consistent styling and flexible parameters

    Args:
        fig (plotly.graph_objects.Figure): Figure for formatting
        title (str): Main title for graph
        xaxis_title (str): Title for x-axis of graph
        yaxis_title (str): Title for y-axis of graph
        legend_title (str, optional): Title for figure legend. Defaults to None.

    Returns:
        plotly.graph_objects.Figure: Plotly figure with updated formatting
    """    
    # Update layout for title and fonts
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="black"
        ),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title_font=dict(size=24)
    )
    
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    
    return fig

def add_bar_totals(fig, df, col, y_offset=1000):
    """Adds sum of stacked bar graph to each column

    Args:
        fig (plotly.graph_objects.Figure): Figure to update
        df (pandas.DataFrame): DF with totals to add
        col (str): Column name from df with relevant totals
        y_offset (int, optional): How far to move title vertically. Defaults to 1000.

    Returns:
        plotly.graph_objects.Figure: Plotly stacked bar graph with totals annotated
    """    
    total_counts = df[col].to_list()
    for i, total in enumerate(total_counts):
        fig.add_annotation(
            x=i,
            y=total + y_offset,
            yanchor='top',
            text=str(total),
            showarrow=False,
            font=dict(
                family="Arial, sans-serif",
                size=14,
                color="black"
            ),
        )
    return fig

def show_example_images(df, group_col='emotion', image_col='image', save_path=None, samples=1):  
    """Displays and optionally saves exampled images from each category of expression

    Args:
        df (pd.DataFrame): df with input data
        group_col (str, optional): Column with groups. Defaults to 'emotion'.
        image_col (str, optional): Column with image arrays. Defaults to 'image'.
        save_path (_type_, optional): Path to save plot. If None figure not saved. Defaults to None.
        samples (int, optional): N of images to use. Defaults to 1.
    """    
    n_cols = df[group_col].nunique()
    n_rows = samples
    fig_width = 10
    fig_height = samples * 2
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, figsize=(fig_width, fig_height))

    for i in range(n_rows):
        samples_df = df.groupby(group_col).sample(n=1).reset_index()
        samples_df.sort_values(by=group_col, inplace=True)
        for idx, row in samples_df.iterrows():
            ax = axes[i, idx]
            ax.imshow(np.array(row[image_col]), cmap='gray')
            ax.axis('off')

    # Add titles to each column
    emo_labels = samples_df[group_col].to_list()
    for col_idx, label in enumerate(emo_labels):
        axes[0, col_idx].set_title(f"{label}")

    plt.tight_layout()    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def convert_pixels_to_array(pixels):
    'Reshape pixel arrays into correct format'
    array = np.array([int(x) for x in pixels.split(' ')]).reshape(48,48)
    array = np.array(array, dtype='uint8')
    return array

# class EmoFaceClassifier():
#     '''
#     Class for ingesting, analyzing, and predicting emotional facial expressions
#     '''

#     def __init__(self) -> None:
#         pass
