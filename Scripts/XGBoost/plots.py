import os 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_3d_mesh(df: pd.DataFrame, x_col: str, y_col: str,
                 z_col: str) -> go.Figure: 
    
    '''This function creates a 3D mesh plot using Plotly,
    with the x_col, y_col and z_col columns of the df
    DataFrame as the x, y and z values, respectively. The 
    plot has a title and axis labels that match the column 
    names, and teh intensity of the mesh is proportional to
    the values in the z_col column. This function returns a 
    Plotly Figure object that can be displayed or saved as 
    desired. '''

    fig = go.Figure(data = [go.Mesh3d(x=df[x_col],
                                      y=df[y_col],
                                      z=df[z_col],
                                      intensity = df[z_col]/df[z_col].min(),
                                      hovertemplate=f"{z_col}: %{{z}}<br>{x_col}: %{{x}}<br>{y_col}:"
                                      "%{{y}}<extra></extra>")],
    )
    
    fig.update_layout(title = dict(text = f'{y_col} vs {x_col}'),
                      scene = dict(xaxis_title = x_col,
                                   yaxis_title = y_col,
                                   zaxis_title = z_col),
                      width = 700,
                      margin = dict(r=20, b=10, l=10, t=50))
    
    return fig