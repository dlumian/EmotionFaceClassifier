import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotly.graph_objects import Figure

def show_example_images(df, group_col='emotion', image_col='image', 
                        col_col='color', save_path=None, samples=1,
                        title='Example FER2013 Faces'):  
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

    # Dictionary to store emotion titles and corresponding subplots
    sorted_df = df.sort_values(by='emotion')
    emo_labels = sorted_df['emotion'].unique().tolist()
    emotion_axes = {emotion: [] for emotion in emo_labels}
    emotion_color_dict = sorted_df[['emotion', 'color']].drop_duplicates().set_index('emotion')['color'].to_dict()

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, figsize=(fig_width, fig_height))

    # For each number of samples requested, pull 1 example of each emotion
    for i in range(n_rows):
        samples_df = df.groupby(group_col).sample(n=1).reset_index()
        samples_df.sort_values(by=group_col, inplace=True)
        for idx, row in samples_df.iterrows():
            ax = axes[i, idx]
            ax.imshow(np.array(row[image_col]), cmap='gray')
            ax.axis('off')
            emotion_axes[row['emotion']].append(ax)

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
    # plt.show()
    plt.close()
    return fig, axes

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