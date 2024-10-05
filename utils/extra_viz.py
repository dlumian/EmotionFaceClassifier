import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches


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