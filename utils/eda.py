import pandas as pd
from pywaffle import Waffle
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotly.graph_objects as go

# Function to create a plotly bar chart with facet columns
def plot_emotion_counts(
        dataframe, 
        x_axis='emotion', 
        y_axis='Count',
        color_by='usage', 
        plot_title='Emotion Counts by Usage (Train/Test)', 
        output_path=None, 
        is_stacked=False,
        styling_dict=None,
        legend_note=None,
        auto_text=False,
        order_categories=None
    ):
    """
    Creates a bar chart displaying emotion counts with options for grouping, stacking, and styling.

    Parameters:
    - dataframe: pandas.DataFrame
        The data source for the plot.
    - x_axis: str, default 'emotion'
        The column name for the x-axis.
    - y_axis: str, default 'Count'
        The column name for the y-axis values.
    - color_by: str, default 'usage'
        The column name used for color coding the bars.
    - plot_title: str, default 'Emotion Counts by Usage (Train/Test)'
        The title of the plot.
    - output_path: str or None, default None
        The file path to save the plot as an image. If None, the plot is not saved.
    - is_stacked: bool, default False
        Determines if the bars should be stacked or grouped.
    - styling_dict: dict or None, default None
        A dictionary of styling options to apply to the plot traces.
    - legend_note: str or None, default None
        Text to display in the legend.
    - auto_text: bool, default False
        Automatically adds text labels to the bars.
    - order_categories: list or None, default None
        A list specifying the order of categories on the x-axis.

    Returns:
    - figure: plotly.graph_objects.Figure
        The created bar chart as a Plotly figure object.
    """
    bar_mode = 'stack' if is_stacked else 'group'

    figure = px.bar(
        dataframe, 
        x=x_axis, 
        y=y_axis,
        color=color_by, 
        barmode=bar_mode, 
        title=plot_title,
        text_auto=auto_text
    )
    
    if order_categories:
        figure.update_xaxes(categoryorder='array', categoryarray=order_categories)

    if styling_dict:
        figure = apply_trace_style(figure, styling_dict)

    figure = format_figure(figure, title=plot_title, x_axis_label='Emotion', y_axis_label='Count')

    if legend_note:
        figure = add_legend_annotation(figure, legend_note)
    else:
        figure.update_layout(legend_title_text=color_by.capitalize())

    if output_path:
        figure.write_image(output_path)

    return figure

def format_figure(fig, title, x_axis_label, y_axis_label):
    """
    Standardizes the appearance of a figure with a title, x-axis label, and y-axis label.

    Parameters:
    - fig: plotly.graph_objects.Figure
        The figure object to format.
    - title: str
        The title of the figure.
    - x_axis_label: str
        The label for the x-axis.
    - y_axis_label: str
        The label for the y-axis.

    Returns:
    - fig: plotly.graph_objects.Figure
        The formatted figure object.
    """
    # Update the layout with the specified title and font properties
    fig.update_layout(
        title=title,
        title_font={"size": 20, "family": "Arial, sans-serif"},
        title_x=0.5,  # Center the title horizontally
        title_y=0.9,  # Position title slightly above the top
        title_xanchor="center",
        title_yanchor="top",
        xaxis_title=x_axis_label,
        xaxis_title_font={"size": 16},
        xaxis_tickfont={"size": 14},
        yaxis_title=y_axis_label,
        yaxis_title_font={"size": 16},
        yaxis_tickfont={"size": 14},
    )
    return fig

def apply_trace_style(fig, style_config):
    """
    Applies custom trace styles from a dictionary to a figure's traces.

    Parameters:
    - fig: plotly.graph_objects.Figure
        The figure object to update.
    - style_config: dict
        A dictionary mapping trace names (e.g., "Angry") to their style properties (e.g., color, opacity, line).

    Returns:
    - fig: plotly.graph_objects.Figure
        The updated figure object.
    """
    if not isinstance(style_config, dict):
        raise ValueError("style_config must be a dictionary")

    for trace in fig.data:
        trace_name = trace.name

        if trace_name in style_config:
            trace_style = style_config[trace_name]

            # Set color for each emotion dynamically
            colors = []
            for emotion in trace.x:
                color = trace_style.get('color', {}).get(emotion)
                if color:
                    colors.append(color)
                else:
                    colors.append('grey')  # Default color if not specified

            # Prepare marker settings dynamically
            marker_settings = {
                'color': colors,
                'opacity': trace_style.get('opacity', 1.0)
            }

            # Add optional nested attributes like line and pattern if they exist in the config
            if 'line' in trace_style:
                marker_settings['line'] = trace_style['line']

            if 'pattern' in trace_style:
                marker_settings['pattern'] = trace_style['pattern']

            # Update the trace with marker settings
            trace.update(
                marker=marker_settings,
            )
    return fig

def add_legend_annotation(figure, annotation_text):
    """
    Adds a custom annotation to the legend area of the figure.
    
    Parameters:
    - figure: plotly.graph_objects.Figure
        The figure object to update.
    - annotation_text: str
        The text to display in the annotation.
        
    Returns:
    - figure: plotly.graph_objects.Figure
        The updated figure object with the annotation.
    """
    # Disable the default legend display
    figure.update_layout(showlegend=False)

    # Add a custom annotation near the legend area
    figure.add_annotation(
        text=annotation_text,
        x=0.98, y=-0.42,
        xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=12),
        align="left"
    )
    return figure

def calculate_waffle_counts(dataframe, count_column, n_rows=20, n_cols=20):
    """
    Calculates the number of waffle chart tiles each category should occupy 
    based on its count relative to the total count.

    Parameters:
    - df: pandas.DataFrame
        The dataframe containing the data.
    - count_col: str
        The column in the dataframe representing the counts for each category.
    - n_rows: int, optional, default=20
        The number of rows in the waffle chart.
    - n_cols: int, optional, default=20
        The number of columns in the waffle chart.

    Returns:
    - df: pandas.DataFrame
        The updated dataframe with an additional column 'waffle_tiles' 
        indicating the number of tiles for each category.
    - items_per_tile: float
        The average number of items each tile represents.
    """
    total_count = dataframe[count_column].sum()
    total_tiles = n_rows * n_cols
    dataframe['waffle_tiles'] = (dataframe[count_column] / total_count * total_tiles).round().astype(int)
    items_per_tile = total_count / total_tiles
    return dataframe, items_per_tile

def create_subplot_data(dataframe, group_column, tile_count_column, title, color_column):
    """
    Creates a dictionary with the necessary information for creating a subplot in a waffle chart.

    Parameters:
    - dataframe: pandas.DataFrame
        The dataframe containing the data for the subplot.
    - group_column: str
        The column name for the categorical values in the subplot.
    - tile_count_column: str
        The column name for the number of tiles each category should occupy.
    - title: str
        The title of the subplot.
    - color_column: str
        The column name for the color to use for each category.

    Returns:
    - subplot_data: dict
        A dictionary with the necessary information for creating a subplot in a waffle chart.
    """
    subplot_data = {
        'values': dataframe[tile_count_column].tolist(),
        'labels': dataframe[group_column].tolist(),
        'title': {'text': title, 'loc': 'center', 'fontsize': 20},
        'colors': dataframe[color_column].tolist()
    }
    return subplot_data

def plot_emotion_waffle(
        dataframe: pd.DataFrame,
        count_column: str,
        group_column: str,
        split_column: str = None,
        color_column: str = None,
        rows: int = 20,
        columns: int = 20,
        output_path: str = None
) -> None:
    """
    Plot a waffle chart to visualize emotion expression image counts by usage.

    Parameters:
        dataframe (pd.DataFrame): The source data containing counts for each emotion and usage category.
        count_column (str): The column name representing the count of images for each category-usage pair.
        group_column (str): The column name representing categories (e.g., emotion) in the dataframe.
        split_column (str, optional): The column name to split the data into subplots. Defaults to None.
        color_column (str, optional): The column name for the color to use for each category in the dataframe. Defaults to None.
        rows (int, optional): The number of rows in the waffle chart. Defaults to 20.
        columns (int, optional): The number of columns in the waffle chart. Defaults to 20.
        output_path (str, optional): The file path to save the heatmap image. Defaults to None.
    """
    if split_column and len(dataframe[split_column].unique()) > 2:
        raise ValueError("Only 1 or 2 subplots are supported.")

    group_titles = list(dataframe[split_column].unique()) if split_column else ['All Data']
    # Prepare plot configurations
    plot_configs = {}
    subplot_ids = [111, 122]
    items_per_tile_info = {}

    # Generate waffle chart data for each subplot group
    for i, group in enumerate(group_titles):
        group_df = dataframe[dataframe[split_column] == group].copy() if split_column else dataframe.copy()
        subplot_df, items_per_tile = calculate_waffle_counts(
            dataframe=group_df, 
            count_column=count_column, 
            n_rows=rows, 
            n_cols=columns
        )
        items_per_tile_info[group] = items_per_tile
        waffle_info = create_subplot_data(
            dataframe=subplot_df, 
            group_column=group_column, 
            tile_count_column='waffle_tiles', 
            title=group,  # Use the group name as the title
            color_column=color_column
        )
        plot_configs[subplot_ids[i]] = {
            'values': waffle_info['values'],
            'labels': waffle_info['labels'],
            'colors': waffle_info['colors'],
            'label': group  # Use the group name as the title
        }

    # Create the waffle figure
    fig = plt.figure(
        FigureClass=Waffle,
        plots=plot_configs,
        rows=rows,
        columns=columns,
        rounding_rule='ceil',
        figsize=(20, 10)
    )
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Add a unified legend to the figure
    legend_handles = [Patch(facecolor=color, label=label) for label, color in zip(waffle_info['labels'], waffle_info['colors'])]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(waffle_info['labels']),
        prop={'size': 20,
            'weight': 'bold'}
    )

    # Add main title and subtitles
    fig.suptitle('Emotion Counts by Expression', fontsize=26, y=1.05, fontweight='bold')

    # Remove individual subplot legends
    for idx, ax in enumerate(fig.axes):
        ax.get_legend().remove()
        ax.set_title(group_titles[idx], fontsize=20, fontweight='bold', pad=10) 

    fig.tight_layout()
    # Save the figure
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f'Figure saved to {output_path}')

def plot_emotion_heatmap(
    dataframe: pd.DataFrame, 
    category_col: str, 
    usage_col: str, 
    count_col: str, 
    output_path: str = None, 
    order_categories: list = None
) -> None:
    """
    Generates a heatmap to visualize emotion expression image counts by usage.

    Parameters:
        dataframe (pd.DataFrame): The source data containing counts for each emotion and usage category.
        category_col (str): The column name representing categories (e.g., emotion) in the dataframe.
        usage_col (str): The column name representing usage types (e.g., Training, Testing) in the dataframe.
        count_col (str): The column name representing the count of images for each category-usage pair.
        output_path (str, optional): The file path to save the heatmap image. Defaults to None.
        order_categories (list, optional): A list specifying the order of categories on the y-axis. Defaults to None.

    Returns:
        None
    """
    heatmap_df = dataframe.pivot(index=category_col, columns=usage_col, values=count_col)
    usage_order = ['Training', 'Testing']
    heatmap_df = heatmap_df[usage_order]

    # Create the heatmap figure using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        colorscale='Plasma',
        text=heatmap_df.values,
        hoverinfo='text',
        showscale=True,
        zmin=0,
        zmax=heatmap_df.values.max(),
    ))

    # Update y-axis order if a specific category order is provided
    if order_categories:
        fig.update_yaxes(categoryorder='array', categoryarray=order_categories)

    # Configure the layout and appearance of the heatmap
    fig.update_traces(
        xgap=1,
        ygap=1,
        hoverongaps=False,
    )

    fig.update_layout(
        title='Emotion Expression Image Counts by Usage Heatmap',
        xaxis_title='Usage',
        yaxis_title='Category',
        yaxis=dict(autorange='reversed'),
        width=600,
        height=400,
        paper_bgcolor='white',
        plot_bgcolor='white',
        title_x=0.5,  # Center the top title
        title_font_size=20,  # Increase font size of top title
        xaxis_title_font_size=16,  # Increase font size of x-axis title
        yaxis_title_font_size=16,  # Increase font size of y-axis title
        font_size=14  # Increase font size of row/col labels
    )

    # Save the heatmap as an image if output path is provided
    if output_path:
        fig.write_image(output_path)

    # Display the heatmap
    fig.show()
