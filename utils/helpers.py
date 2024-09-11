import os
import json
import cv2
import numpy as np

def load_config(file_path):
    'Opens and loads json file'
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def to_json(data, file_path):
    'Writes data to filepath with nice formatting'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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
    

def apply_default_plotly_styling(fig, title, xaxis_title=None, 
                          yaxis_title=None, legend_title=None):
    """ Function to update layout with consistent styling and flexible parameters

    Args:
        fig (plotly.graph_objects.Figure): Figure for formatting
        title (str): Main title for graph
        xaxis_title (str): Title for x-axis of graph
        yaxis_title (str): Title for y-axis of graph

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
        title_font=dict(size=24)
    )

    if xaxis_title is not None:
        fig.update_layout( xaxis_title=xaxis_title)

    if yaxis_title is not None:
        fig.update_layout(yaxis_title=yaxis_title)

    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    
    return fig


def add_bar_totals(fig, df, col, y_offset=1000):
    """Adds sum of stacked bar graph to each column

    Args:
        fig (plotly.graph_objects.Figure): Figure to update
        df (pd.DataFrame): DF with totals to add
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


def load_images_and_labels(folder):
    images = []
    labels = []
    for emotion in os.listdir(folder):
        # Skip hidden files and directories (those that start with a dot)
        if emotion.startswith('.'):
            continue
        label_folder = os.path.join(folder, emotion)
        for filename in os.listdir(label_folder):
            # Skip hidden files
            if filename.startswith('.'):
                continue
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = cv2.imread(os.path.join(label_folder, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = img.flatten()  # Flatten the image
                    img = img / 255.0  # Normalize
                    images.append(img)
                    labels.append(emotion)
    return np.array(images), np.array(labels)