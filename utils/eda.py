import plotly.express as px

# Function to create a plotly bar chart with facet columns
def plot_emotion_counts(
        df, 
        x='emotion', 
        y='Count',
        color='usage', 
        title='Emotion Counts by Usage (Train/Test)', 
        save_path=None, 
        stacked=False):
    # Use 'stack' or 'group' for bar mode depending on the stacked argument
    barmode = 'stack' if stacked else 'group'

    fig = px.bar(
        df, 
        x=x, 
        y=y,
        color=color, 
        barmode=barmode, 
        title=title,
        color_discrete_sequence=px.colors.qualitative.Plotly)

    fig = apply_plotly_formatting(fig, title=title, xlabel='Emotion', ylabel='Count')

    # Update the layout to capitalize the legend title
    fig.update_layout(legend_title_text=color.capitalize())

    # Save the plot if save_path is provided
    if save_path:
        fig.write_image(save_path)

    # Display the plot
    fig.show()

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


def apply_trace_style(style_dict):
    pass










# def apply_custom_styles(fig, style_config):
#     # Verify that the style_config is properly structured
#     if not isinstance(style_config, dict):
#         raise ValueError("style_config must be a dictionary")

#     for dataset_type in ['Train', 'Test']:
#         # Check if dataset type exists in the config
#         if dataset_type not in style_config:
#             raise KeyError(f"Missing style configuration for dataset type: '{dataset_type}'")
        
#         # Validate opacity value
#         opacity = style_config[dataset_type].get('opacity', 1.0)  # Default to full opacity if not set
#         if not (isinstance(opacity, (int, float)) and 0 <= opacity <= 1):
#             raise ValueError(f"Opacity for dataset type '{dataset_type}' must be a float between 0 and 1")

#     # Apply styles to the figure
#     for trace in fig.data:
#         dataset_type = trace.name  # The name will be 'Train' or 'Test'
        
#         if dataset_type in style_config:
#             # Extract colors for each emotion in trace.x
#             colors = []
#             for emotion in trace.x:
#                 if emotion in style_config[dataset_type]['color']:
#                     colors.append(style_config[dataset_type]['color'][emotion])
#                 else:
#                     raise KeyError(f"Missing color configuration for emotion: '{emotion}' in dataset type: '{dataset_type}'")
            
#             # Set optional style elements with default values
#             line_color = style_config[dataset_type].get('line', {}).get('color', 'black')  # Default to black
#             line_width = style_config[dataset_type].get('line', {}).get('width', 1)       # Default to width 1
#             pattern_shape = style_config[dataset_type].get('pattern', {}).get('shape', None)  # Default to None (no pattern)
#             hoverinfo = style_config[dataset_type].get('hoverinfo', 'x+y')  # Default to show x and y values
#             bar_width = style_config[dataset_type].get('width', None)       # Default to None (use Plotly default)

#             # Update the trace with custom colors and optional styles
#             trace.update(
#                 marker=dict(
#                     color=colors,
#                     opacity=opacity,
#                     line=dict(
#                         color=line_color,
#                         width=line_width
#                     ),
#                     pattern=dict(
#                         shape=pattern_shape
#                     ) if pattern_shape else {}  # Only add pattern if it's defined
#                 ),
#                 hoverinfo=hoverinfo,
#                 width=bar_width
#             )
