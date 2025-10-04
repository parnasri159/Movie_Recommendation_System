import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_revenue_trends(trends_data):
    df = pd.DataFrame(trends_data)
    fig = px.line(df, x='year', y='revenue', title='Average Revenue Over Years', template='plotly_dark')
    fig.update_layout(showlegend=False)
    return fig

def plot_correlation_heatmap(corr_data):
    df = pd.DataFrame(corr_data)
    fig = px.imshow(df, text_auto=True, color_continuous_scale='RdBu', title='Feature Correlations')
    return fig

def plot_genre_popularity(genre_data):
    df = pd.DataFrame(genre_data)
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig.update_layout(title='Genre Popularity Over Time', template='plotly_dark', showlegend=True)
    return fig

def plot_runtime_impact(impact_data):
    df = pd.DataFrame(impact_data)
    fig = px.bar(df, x='runtime_bin', y=['popularity', 'revenue'], barmode='group', title='Runtime Impact on Popularity and Revenue', template='plotly_dark')
    return fig