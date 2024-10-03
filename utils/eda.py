import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Function to create a seaborn stacked bar chart for emotions
def plot_emotion_counts_seaborn(
        df, 
        x='emotion', 
        hue='usage', 
        palette='viridis', 
        title='Emotion Counts by Usage', 
        xlabel='Emotion', 
        ylabel='Count', 
        save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=df, x=x, hue=hue, palette=palette, ax=ax)
    apply_standard_formatting(fig, ax, title=title, xlabel=xlabel, ylabel=ylabel)
    # Capitalize the legend title
    ax.legend(title=hue.capitalize())

    # Save the plot if save_path is provided
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    # Display the plot
    plt.show()

def apply_standard_formatting(
        fig, 
        ax, 
        title, 
        xlabel, 
        ylabel, 
        xtick_rotation=0, 
        ytick_rotation=0):
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='x', rotation=xtick_rotation, labelsize=12)
    ax.tick_params(axis='y', rotation=ytick_rotation, labelsize=12)
    fig.tight_layout()

# Function to create a plotly bar chart with facet columns
def plot_emotion_counts_plotly(
        df, 
        x='emotion', 
        y='count',
        color='usage', 
        title='Emotion Counts Split by Usage (Train/Test)', 
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

def apply_plotly_formatting(fig, title, xlabel, ylabel):
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
