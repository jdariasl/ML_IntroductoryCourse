#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow, contour
import sompy
get_ipython().run_line_magic('matplotlib', 'inline')


def show(umat, som, distance2=1, row_normalized=False, show_data=True,
         contooor=True, blob=False, labels=False, colors=None):
    umat = umat
    msz = som.codebook.mapsize
    proj = som.project_data(som.data_raw)
    coord = som.bmu_ind_to_xy(proj)

    fig, ax = plt.subplots(1, 1)
    imshow(umat, cmap=plt.cm.get_cmap('RdYlBu_r'), alpha=0.9)
    plt.axis('off')
    

    if show_data:
        plt.scatter(coord[:, 1], coord[:, 0], s=30, alpha=1., c=colors,
                    marker='o', cmap='jet')
        

    if labels:
        if labels is True:
            labels = som.build_data_labels()
        for label, x, y in zip(labels, coord[:, 1], coord[:, 0]):
            plt.annotate(str(label), xy=(x, y),
                         horizontalalignment='center',
                         verticalalignment='center')

    ratio = float(msz[0])/(msz[0]+msz[1])
    fig.set_size_inches((1-ratio)*15, ratio*15)
    plt.tight_layout()
    plt.subplots_adjust(hspace=.00, wspace=.000)
    sel_points = list()
    
    plt.show()
    return umat

