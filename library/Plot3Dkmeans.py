 
# Initialize plotting library and functions for 3D scatter plots 
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_classification, make_regression
#from sklearn.externals import six
import pandas as pd
import numpy as np
import argparse
import json
import re
import os
import sys
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()

def rename_columns(df, prefix='x'):
    """
    Rename the columns of a dataframe to have X in front of them

    :param df: data frame we're operating on
    :param prefix: the prefix string
    """
    df = df.copy()
    df.columns = [prefix + str(i) for i in df.columns]
    return df

# Create an artificial dataset with 3 clusters for 3 feature columns
X, Y = make_classification(n_samples=100, n_classes=3, n_features=3, n_redundant=0, n_informative=3,
                             scale=1000, n_clusters_per_class=1)
df = pd.DataFrame(X)
# rename X columns
df = rename_columns(df)
# and add the Y
df['y'] = Y
#df.head(3)

def Plot3Dk_means():
    # Visualize cluster shapes in 3d.

    cluster1=df.loc[df['y'] == 0]
    cluster2=df.loc[df['y'] == 1]
    cluster3=df.loc[df['y'] == 2]

    scatter1 = dict(
        mode = "markers",
        name = "Cluster 1",
        type = "scatter3d",    
        x = cluster1.values[:,0], y = cluster1.values[:,1], z = cluster1.values[:,2],
        marker = dict( size=2, color='green')
    )
    scatter2 = dict(
        mode = "markers",
        name = "Cluster 2",
        type = "scatter3d",    
        x = cluster2.values[:,0], y = cluster2.values[:,1], z = cluster2.values[:,2],
        marker = dict( size=2, color='blue')
    )
    scatter3 = dict(
        mode = "markers",
        name = "Cluster 3",
        type = "scatter3d",    
        x = cluster3.values[:,0], y = cluster3.values[:,1], z = cluster3.values[:,2],
        marker = dict( size=2, color='red')
    )
    cluster1 = dict(
        alphahull = 5,
        name = "Cluster 1",
        opacity = .1,
        type = "mesh3d",    
        x = cluster1.values[:,0], y = cluster1.values[:,1], z = cluster1.values[:,2],
        color='green', showscale = True
    )
    cluster2 = dict(
        alphahull = 5,
        name = "Cluster 2",
        opacity = .1,
        type = "mesh3d",    
        x = cluster2.values[:,0], y = cluster2.values[:,1], z = cluster2.values[:,2],
        color='blue', showscale = True
    )
    cluster3 = dict(
        alphahull = 5,
        name = "Cluster 3",
        opacity = .1,
        type = "mesh3d",    
        x = cluster3.values[:,0], y = cluster3.values[:,1], z = cluster3.values[:,2],
        color='red', showscale = True
    )
    layout = dict(
        title = 'Interactive Cluster Shapes in 3D',
        scene = dict(
            xaxis = dict( zeroline=True ),
            yaxis = dict( zeroline=True ),
            zaxis = dict( zeroline=True ),
        )
    )
    fig = dict( data=[scatter1, scatter2, scatter3, cluster1, cluster2, cluster3], layout=layout )
    # Use py.iplot() for IPython notebook
    plotly.offline.iplot(fig, filename='mesh3d_sample')