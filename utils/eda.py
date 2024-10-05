import plotly.express as px

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