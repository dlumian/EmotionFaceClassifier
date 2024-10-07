import json
from pywaffle import Waffle
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotly.graph_objects as go

# Function to create a plotly bar chart with facet columns
def plot_emotion_counts(
        df, 
        x='emotion', 
        y='Count',
        color='usage', 
        title='Emotion Counts by Usage (Train/Test)', 
        save_path=None, 
        stacked=False,
        style_dict=None,
        legend_text=None,
        text_auto=False
    ):
    # Use 'stack' or 'group' for bar mode depending on the stacked argument
    barmode = 'stack' if stacked else 'group'

    fig = px.bar(
        df, 
        x=x, 
        y=y,
        color=color, 
        barmode=barmode, 
        title=title,
        text_auto=text_auto
    )
    
    # Update trace settings if style dict available
    if style_dict:
        fig = apply_trace_style(fig, style_dict)

    # Update general formatting for consistency
    fig = apply_formatting(fig, title=title, xlabel='Emotion', ylabel='Count')

    if legend_text:
        fig = annotate_legend(fig, legend_text)
    else:
        # Update the layout to capitalize the legend title
        fig.update_layout(legend_title_text=color.capitalize())

    # Save the plot if save_path is provided
    if save_path:
        fig.write_image(save_path)

    return fig

def apply_formatting(fig, title, xlabel, ylabel):
    fig.update_layout(
        title={
            'text': title,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        title_font=dict(size=20, family='Arial, sans-serif'),
        xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
    )
    return fig

def apply_trace_style(fig, style_dict):
    # Verify that the style_config is properly structured
    if not isinstance(style_dict, dict):
        raise ValueError("style_dict must be a dictionary")

    # Update traces using the style dictionary
    for trace in fig.data:
        dataset_type = trace.name  
    
        # Ensure that dataset_type exists in style_config
        if dataset_type in style_dict:
            # Extract the configuration for the dataset type
            config = style_dict[dataset_type]
            
            # Set color for each emotion dynamically
            colors = []
            for emotion in trace.x:
                color = config.get('color', {}).get(emotion)
                if color:
                    colors.append(color)
                else:
                    colors.append('grey')  # Default color if not specified
            
            # Prepare marker settings dynamically
            marker_settings = {
                'color': colors,
                'opacity': config.get('opacity', 1.0)
            }
            
            # Add optional nested attributes like line and pattern if they exist in the config
            if 'line' in config:
                marker_settings['line'] = config['line']
            
            if 'pattern' in config:
                marker_settings['pattern'] = config['pattern']
            
            # Update the trace with marker settings
            trace.update(
                marker=marker_settings,
            )
    return fig

def annotate_legend(fig, text):
    # Hide the default legend
    fig.update_layout(showlegend=False)

    # Add custom note as an annotation
    fig.add_annotation(
        text=text,
        x=0.98, y=-0.42,
        xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=12),
        align="left"
    )
    return fig

def get_waffle_counts(df, count_col, rows=20, cols=20):
    total_count = df[count_col].sum()
    total_tiles = rows*cols
    df['waffle_tiles'] = (df[count_col] / total_count * total_tiles).round().astype(int)
    items_per_tile = total_count / total_tiles
    return df, items_per_tile

def subplot_dict(df, group_col, tile_count, title, color_col):
    temp_dict = {
        'values': df[tile_count].tolist(),  # Convert actual number to a reasonable block number
        'labels': df[group_col].tolist(),
        'legend': {},
        'title': {'label': title, 'loc': 'center', 'fontsize': 20},
        'colors': df[color_col].tolist()
    }
    return temp_dict

def plot_emotion_waffle(
        df,
        count_col, 
        group_col, 
        split_col=None,
        color_col=None,
        rows=20, 
        cols=20,
        save_path=None):
    if split_col:
        subplot_groups = list(df[split_col].unique())
        subplot_count = len(subplot_groups)
        if subplot_count > 2:
            print(f"{subplot_count} found! {', '.join(subplot_groups)}")
            raise ValueError("Current implementation only works for 1 or 2 subplots.")
    else:
        subplot_count = 1
        subplot_groups ='All Data'
    plots_dict ={}
    subplot_ids = [111, 122]
    items_per_tile_dict = {}
    for i, subgroup in enumerate(subplot_groups):
        temp_df = df[df[split_col]==subgroup].copy()
        subplot_df, items_per_tile = get_waffle_counts(df=temp_df, count_col=count_col, rows=rows, cols=cols)
        items_per_tile_dict[subgroup] = items_per_tile
        waffle_dict = subplot_dict(
            df=subplot_df, 
            group_col=group_col, 
            tile_count='waffle_tiles', 
            title=subgroup, 
            color_col=color_col)
        plots_dict[subplot_ids[i]] = waffle_dict

    fig = plt.figure(
        FigureClass=Waffle,
        plots=plots_dict,
        rows=rows,
        columns=cols,
        rounding_rule='ceil',  # Change rounding rule, so value less than 1000 will still have at least 1 block
        figsize=(20, 10)
    )

    # Explicitly remove any legends from the figure's axes
    for i, ax in enumerate(fig.get_axes()):
        legend = ax.get_legend()
        if legend:
            legend.remove()
        pos = ax.get_position()  # Get current position
        new_pos = [pos.x0, pos.y0, pos.width * 50, pos.height]  # Increase width to reduce spacing
        ax.set_position(new_pos)  # Set new position

    # Manually create legend handles and labels for overall legend
    legend_handles = [
        Patch(facecolor=color, label=label)
        for label, color in zip(waffle_dict['labels'], waffle_dict['colors'])
    ]

    # Add a single legend for the entire figure with a title
    legend = fig.legend(
        handles=legend_handles,
        loc='lower center',  # Position the legend at the bottom center of the figure
        bbox_to_anchor=(0.5, -0.1),  # Adjust the location to fit below the plot
        ncol=len(waffle_dict['labels']),  # Number of columns for legend entries
        fontsize=16,
        title="Emotion Expression",  # Title for the legend
        title_fontsize=16  # Font size for the legend title
    )

    training_data_annotation = f"For training data, each square represents {items_per_tile_dict['Training']:.1f} images."
    testing_data_annotation = f"For testing data, each square represents {items_per_tile_dict['Testing']:.1f} images."
    # Bottom left
    fig.text(0.03, 0.01, training_data_annotation, ha='left', va='top', fontsize=12)
    # Bottom right
    fig.text(0.95, 0.01, testing_data_annotation, ha='right', va='top', fontsize=12)
    # Adjust layout to ensure the subplots do not overlap with annotations
    plt.tight_layout(rect=[0, 0.05, 1, 1])  

    fig.suptitle('Image Counts by Emotion Expression', fontsize=24, fontweight='bold')

    # Save the combined chart
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    # Close the figure to free memory
    plt.close(fig)

def plot_emotion_heatmap(df, group_col, split_col, value_col, save_path=None):
    heatmap_data = df.pivot(index=group_col, columns=split_col, values=value_col)
    new_order = ['Training', 'Testing']
    hm_reordered = heatmap_data[new_order]

    fig = go.Figure(data=go.Heatmap(
        z=hm_reordered.values,
        x=hm_reordered.columns,
        y=hm_reordered.index,
        colorscale='Plasma',  
        text=hm_reordered.values,
        hoverinfo='text',
        showscale=True,
        zmin=0,
        zmax=hm_reordered.values.max(),
    ))

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
    )

    if save_path:
        fig.write_image(save_path)
    fig.show()
