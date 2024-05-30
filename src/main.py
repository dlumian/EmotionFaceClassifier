import os
import pandas as pd
import numpy as np
import plotly.express as px

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

# Function to update layout with consistent styling and flexible parameters
def apply_default_styling(fig, title, xaxis_title, yaxis_title, 
                          legend_title=None, 
                        #   color_discrete_sequence=px.colors.qualitative.Bold, 
                          labels=None):
    # Apply the color sequence
    # fig.update_traces(marker=dict(color=color_discrete_sequence))
    # fig.update_traces(textfont_size = 14, textangle = 0, textposition = "outside")

    # Update layout for title and fonts
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            # 'y': 1.0,
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
    
    if labels is not None:
        fig.update_layout(xaxis_title=labels.get('x', xaxis_title), yaxis_title=labels.get('y', yaxis_title))
    
    return fig

def add_bar_totals(fig, df, col, y_offset=1000):
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

class EmoFaceClassifier():
    '''
    Class for ingesting, analyzing, and predicting emotional facial expressions
    '''

    def __init__(self) -> None:
        pass
