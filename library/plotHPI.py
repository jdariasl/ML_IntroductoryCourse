#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

def plot_hpi(df):
    
    samples, columns = df.shape
    features = columns -1
    data = df.iloc[:,0: features-1]
    output = df.iloc[:, -1:]
    zdata = output[0:100].values
    xdata = data['AGE'].iloc[0:100,].values
    ydata = data['TAX'].iloc[0:100,].values
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens');
    ax.set_xlabel("Edad")
    ax.set_ylabel("Tasa de impuesto")
    ax.set_zlabel("HPI")
    return plt


