import os
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.patches as patches

from .helpers import (
    apply_default_plotly_styling,
    add_bar_totals
)

def convert_pixels_to_array(pixels):
    'Reshape pixel arrays into correct format for FER2013 csv input'
    array = np.array([int(x) for x in pixels.split(' ')]).reshape(48,48)
    array = np.array(array, dtype='uint8')
    return array

def create_img(row):
    base_path = os.path.join('data', 'fer2013')
    usage = row['usage']
    emot = row['emotion']
    dir_path = os.path.join(base_path, usage, emot)
    os.makedirs(dir_path, exist_ok=True)

    f_name = f"{emot}-{row['emo_count_id']}.jpg"

    # Convert the array to grayscale
    arr = row['image']
    tmp_img = Image.fromarray(arr.astype('uint8'), 'L')

    # Combined path
    final_path = os.path.join(dir_path, f_name)
    # Save the grayscale image as a JPG
    tmp_img.save(final_path)

def generate_file_dataframe(root_dir):
    data = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                # Get the relative path and split into parts
                rel_path = os.path.relpath(root, root_dir)
                parts = rel_path.split(os.sep)
                
                # Collect data
                record = parts + [file, os.path.join(root, file)]
                data.append(record)
    # Define the columns based on the maximum directory depth + 1 for the file name + 1 for the full path
    max_depth = max(len(record) - 2 for record in data)
    columns = [f'Level {i}' for i in range(1, max_depth + 1)] + ['Filename', 'Full Path']
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    df.rename(columns={
            'Level 1': 'train_test_split', 
            'Level 2': 'emotion'
        },
        inplace=True
    )
    
    return df

def emotion_count_piv(df, gby_cols=['emotion', 'train_test_split'], agg_col='Filename',
                      count_cols=['Training', 'Testing']):
    emo_gby = df.groupby(gby_cols)[agg_col].count()
    emo_piv = emo_gby.reset_index().pivot(
        index=gby_cols[0], 
        columns=gby_cols[1], 
        values=agg_col).reset_index(drop=False, names='Emotion')    
    emo_piv['TotalImages'] = emo_piv[count_cols].sum(axis=1)

    # For each count column calculage a percentage col
    for cc in count_cols:
        new_col = f'{cc}Perc'
        emo_piv[new_col] = (emo_piv[cc]/emo_piv['TotalImages'])*100
        emo_piv[new_col] = emo_piv[new_col].round(decimals=2)
    
    return emo_piv

def piv_stacked_bar(df, label):
    # Stacked bar graph showing counts by emotion
    fig = px.bar(df, x="Emotion", y=["Training", "Testing"], 
                color_discrete_sequence=px.colors.qualitative.Bold)

    fig = apply_default_plotly_styling(fig, title="Image Counts by Emotion", xaxis_title="Emotion", yaxis_title="Count", legend_title="Usage")
    fig = add_bar_totals(fig, df, 'TotalImages')

    fig_path = os.path.join('imgs', f'{label}_count_bar.png')
    fig.show()
    fig.write_image(fig_path)

def waffle_chart(
        df, group_col='Emotion', data_col='TotalImages', 
        save_path=None, display=False, total_squares=100,
        color_dict=None
    ):
    
    waffle_df = df.sort_values(by=data_col, ascending=False)
    # Calculate the number of squares for each category
    waffle_df.loc[:, 'Squares'] = (waffle_df[data_col] / waffle_df[data_col].sum() * total_squares).round().astype(int)

    # Calculate the number of images represented by each box
    images_per_box = waffle_df[data_col].sum() / total_squares

    # Create the waffle chart
    squares = []
    for _, row in waffle_df.iterrows():
        squares.extend([row[group_col]] * row['Squares'])

    # Create grid positions
    grid_size = int(total_squares ** 0.5)
    grid = [(i % grid_size, i // grid_size) for i in range(total_squares)]

    # Create the plot
    fig = go.Figure()

    # Track categories added to the legend
    added_categories = set()

    for i, (x, y) in enumerate(grid):
        category = squares[i] if i < len(squares) else None
        color = color_dict[category] if category else 'rgba(0, 0, 0, 0)'
        show_legend = category not in added_categories if category else False
        if show_legend:
            added_categories.add(category)
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=20, color=color, symbol='square', line=dict(width=2)),
            name=category,
            showlegend=show_legend
        ))

    # Add a text box with the number of images per box
    fig.add_trace(go.Scatter(
        x=[grid_size - 1], y=[-2],
        text=[f'Approximately {images_per_box:.1f} images represented per box.'],
        mode='text',
        showlegend=False
    ))

    # Update layout
    apply_default_plotly_styling(fig=fig, title="Image Distribution by Emotion", xaxis_title=None, 
                            yaxis_title=None, legend_title="Emotion")

    fig.update_layout(
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor='y',  # Ensure the x-axis and y-axis are scaled equally
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False
        ),
        showlegend=True,
        legend_traceorder="reversed",
        autosize=False,
        width=600,  # Adjust the width of the figure
        height=600  # Adjust the height of the figure
    )

    if save_path:
        fig.write_image(save_path)
    if display:
        fig.show()
    plt.close()

def apply_default_matplotlib_styling(fig, axs, title, xaxis_title=None, 
                          yaxis_title=None, legend_title=None):
    # Set the figure title
    fig.suptitle(title, fontsize=24, fontname="Arial", color="black", 
                 y=0.95, x=0.5, ha='center', va='top')
    
    # Check if axs is a single Axes object or an array of Axes
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    # Flatten the array if it is multi-dimensional
    axs = np.ravel(axs)

    # Set axis labels and customize tick labels for each subplot
    for ax in axs:
        if xaxis_title:
            ax.set_xlabel(xaxis_title, fontsize=14, 
                          fontname="Arial", color="black")
        if yaxis_title:
            ax.set_ylabel(yaxis_title, fontsize=14, 
                          fontname="Arial", color="black")
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname("Arial")
            label.set_fontsize(14)
            label.set_color("black")
    
    return fig, axs

def show_example_images(df, group_col='emotion', image_col='image', 
                        col_col='color', save_path=None, samples=1,
                        title='Emotion Category Examples', display=False):  
    """Displays and optionally saves exampled images from each category of expression

    Args:
        df (pd.DataFrame): df with input data
        group_col (str, optional): Column with groups. Defaults to 'emotion'.
        image_col (str, optional): Column with image arrays. Defaults to 'image'.
        save_path (_type_, optional): Path to save plot. If None figure not saved. Defaults to None.
        samples (int, optional): N of images to use. Defaults to 1.
        title (str, optional): Overall title. Defaults to 'Example FER2013 Faces'.

    Returns:
        plt.Figure, plt.axs: Matplotlib figure and axes from subplot figure
    """    
    n_cols = df[group_col].nunique()
    n_rows = samples
    fig_width = 10
    fig_height = samples * 2

    # Dictionary to store emotion titles and corresponding subplots
    sorted_df = df.sort_values(by=group_col)
    emo_labels = sorted_df[group_col].unique().tolist()
    emotion_axes = {emotion: [] for emotion in emo_labels}
    emotion_color_dict = sorted_df[[group_col, col_col]].drop_duplicates().set_index(group_col)[col_col].to_dict()

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, figsize=(fig_width, fig_height))

    # For each number of samples requested, pull 1 example of each emotion
    for i in range(n_rows):
        samples_df = df.groupby(group_col).sample(n=1).reset_index()
        samples_df.sort_values(by=group_col, inplace=True)
        for idx, row in samples_df.iterrows():
            ax = axes[i, idx]

            if image_col == 'Full Path':
                ax.imshow(Image.open(row[image_col]), cmap='gray')
            else:
                ax.imshow(np.array(row[image_col]), cmap='gray')

            ax.axis('off')
            emotion_axes[row[group_col]].append(ax)

    # Add column labels (emotion) in defined color
    for col_idx, label in enumerate(emo_labels):
        axes[0, col_idx].set_title(f"{label}", color=emotion_color_dict[label])

    # Format fonts for output to be consistent and match plotly
    fig, axes = apply_default_matplotlib_styling(
            fig=fig, axs=axes, 
            title=title
        )

    # Improves layout structure for saving images
    # MUST be called before adding color frames (changes layout)
    plt.tight_layout()    

    # Add colored frames to each example by emo label
    for emotion, e_axs in emotion_axes.items():
        for ax in e_axs:
            # Get the position of the current subplot
            pos = ax.get_position()
            # Create a rectangle with the same position and add it to the figure
            rect = patches.Rectangle((pos.x0, pos.y0), pos.width, pos.height,
                                     linewidth=5, 
                                     edgecolor=emotion_color_dict[emotion], 
                                     facecolor='none', 
                                     transform=fig.transFigure
                                )
            fig.patches.append(rect)

    if save_path:
        plt.savefig(save_path)
    if display:
        plt.show()
    plt.close()
    return fig, axes