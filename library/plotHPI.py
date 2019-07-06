#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import math
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("DataFiles/housing.data",delim_whitespace=True, header=None, names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])

samples, columns = df.shape
features = columns -1
data = df.iloc[:,0: features-1]
output = df.iloc[:, -1:]
    
def plot_hpi():
    
    
    zdata = output[0:100].values
    xdata = data['AGE'].iloc[0:100,].values
    ydata = data['TAX'].iloc[0:100,].values
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens');
    ax.set_xlabel("Edad")
    ax.set_ylabel("Tasa de impuesto")
    ax.set_zlabel("HPI x10^3")
    return plt
def distance_vectors(x, X):
    return np.linalg.norm(x - X)


def U(x, X, h):
    return (distance_vectors(x, X)/h)


def kernel_gauss(u):
    return math.exp((-0.5)*math.pow(u, 2))


def parzenW(X, x, h):
    N = len(X)
    accumulator = 0
    for i in range(N):
        accumulator = accumulator + kernel_gauss(U(x, X[i], h))
    return accumulator/N


def nadaraya_watson(X, x, y, h):
    numerator = 0
    denominator = 0
    N = len(X)
    for i in range(N):
        #print(U(x, X[i, :], h))
        numerator = numerator + kernel_gauss(U(x, X[i, :], h)) * y[i]
        #print(numerator)
        denominator = denominator + kernel_gauss(U(x, X[i, :], h))
        #print(denominator)
        if denominator == 0:
            z = 0
        else:
            z = numerator/denominator
    return z


def silverman_bandwidth(X):
    """https://stats.stackexchange.com/questions/6670/which-is-the-formula-from-silverman-to-calculate-the-bandwidth-in-a-kernel-densi"""
    iqr = scipy.stats.iqr(X)
    subIqr = (iqr/1.34)
    std = np.std(X)
    A = np.min([subIqr, std])
    silvermanBandwidth = ((0.9) * A) / math.pow(len(X),(1/5))
    return silvermanBandwidth

def sigmoide(u):
    g = np.exp(u)/(1 + np.exp(u))
    return g
def Gradient(X,y):
    #Aprendizaje
    MaxIter = 100000
    N = X.shape[0]
    d = X.shape[1]
    w = np.ones(d+1).reshape(d+1, 1)
    eta = 0.001
    Error =np.zeros(MaxIter)
    Xent = np.c_[X,np.ones((N,1))]

    for i in range(MaxIter):
        tem = np.dot(Xent,w)
        tem2 = tem-np.array(y)
        Error[i] = np.sum(tem2**2)/N
        tem = np.dot(Xent.T,tem2)
        wsig = w - eta*tem/N
        w = wsig
    #print(w)
    #print('Error=',Error[-1])
    return w

def Poli1():
    zdata = output[0:100].values
    xdata = data['AGE'].iloc[0:100,].values
    ydata = data['TAX'].iloc[0:100,].values*1000000
    X = np.c_[xdata.reshape(100,1),ydata.reshape(100,1)]
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    w=Gradient(Xn,zdata.reshape(100,1))
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    x1 = np.linspace(np.min(xdata), np.max(xdata), num=20)
    x2 = np.linspace(np.min(ydata), np.max(ydata), num=20)

    x1, x2 = np.meshgrid(x1, x2)
    Z = w[0]*(x1-scaler.mean_[0])/np.sqrt(scaler.var_[0]) + w[1]*(x2-scaler.mean_[1])/np.sqrt(scaler.var_[1]) + w[2]

    # Plot the surface.
    surf = ax.plot_surface(x1, x2, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens');
    ax.set_title("Función de regresión polinomial grado 1")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Tasa de impuesto")
    ax.set_zlabel("HPI x10^3")
    
    return ax
    
def Poli2():
    zdata = output[0:100].values
    xdata = data['AGE'].iloc[0:100,].values
    ydata = data['TAX'].iloc[0:100,].values*1000000
    X = np.c_[xdata.reshape(100,1),ydata.reshape(100,1)]
    X = np.c_[X,X**2]
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    w=Gradient(Xn,zdata.reshape(100,1))
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    x1 = np.linspace(np.min(xdata), np.max(xdata), num=20)
    x2 = np.linspace(np.min(ydata), np.max(ydata), num=20)

    x1, x2 = np.meshgrid(x1, x2)
    x3 = x1**2
    x4 = x2**2
    Z = w[0]*(x1-scaler.mean_[0])/np.sqrt(scaler.var_[0]) + w[1]*(x2-scaler.mean_[1])/np.sqrt(scaler.var_[1]) + w[2]*(x3-scaler.mean_[2])/np.sqrt(scaler.var_[2]) + w[3]*(x4-scaler.mean_[3])/np.sqrt(scaler.var_[3]) + w[4]

    # Plot the surface.
    surf = ax.plot_surface(x1, x2, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens');
    ax.set_title("Función de regresión polinomial grado 2")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Tasa de impuesto")
    ax.set_zlabel("HPI x10^3")
    return ax

def HistogramReg(X,Y,x1,x2):
    N = X.shape[0]
    CountMatrix = np.zeros((11,11,11))
    Output = np.zeros((11,11))
    _,bin_edgesX1 = np.histogram(X[:,0],bins=10)
    _,bin_edgesX2 = np.histogram(X[:,1],bins=10)
    _,bin_edgesY = np.histogram(Y,bins=10)
    inc = bin_edgesY[1] - bin_edgesY[0]
    Yrep = bin_edgesY - inc/2
    for i in range(N):
        posx1 = np.nonzero(bin_edgesX1>=X[i,0])
        posx2 = np.nonzero(bin_edgesX2>=X[i,1])
        posY = np.nonzero(bin_edgesY>=Y[i])
        CountMatrix[posx1[0][0]-1,posx2[0][0]-1,posY[0][0]-1] +=1
    for i in range(11):
        for j in range(11):
            Yhist = CountMatrix[i,j,:]
            YTotal = np.sum(Yhist)
            if YTotal == 0:
                Output[i,j] = 0
            else:
                prob = Yhist/YTotal
                Output[i,j] = np.sum(Yrep*prob)
    x1, x2 = np.meshgrid(x1, x2)
    n1,n2 = x1.shape
    Z = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            posx1 = np.nonzero(bin_edgesX1>=x1[i,j])
            posx2 = np.nonzero(bin_edgesX2>=x2[i,j])
            Z[i,j] = Output[posx1[0][0]-1,posx2[0][0]-1]
    return Z


def HisPlot():
    zdata = output[0:100].values
    xdata = data['AGE'].iloc[0:100,].values
    ydata = data['TAX'].iloc[0:100,].values*1000000
    X = np.c_[xdata.reshape(100,1),ydata.reshape(100,1)]
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    x1 = np.linspace(np.min(xdata), np.max(xdata), num=20)
    x2 = np.linspace(np.min(ydata), np.max(ydata), num=20)

    Z = HistogramReg(X,zdata.reshape(100,1),x1,x2)
    x1, x2 = np.meshgrid(x1, x2)
    # Plot the surface.
    surf = ax.plot_surface(x1, x2, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens');
    ax.set_title("Histogram Regression function")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Tasa de impuesto")
    ax.set_zlabel("HPI x10^3")
    
def knn_un(n_neighbors):
    zdata = output[0:100].values
    xdata = data['AGE'].iloc[0:100,].values
    ydata = data['TAX'].iloc[0:100,].values*1000000
    X = np.c_[xdata.reshape(100,1),ydata.reshape(100,1)]
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors, algorithm = 'brute')
    neigh.fit(X, zdata.reshape(100,1))
    x1 = np.linspace(np.min(xdata), np.max(xdata), num=20)
    x2 = np.linspace(np.min(ydata), np.max(ydata), num=20)
    x1, x2 = np.meshgrid(x1, x2)
    n1,n2 = x1.shape
    Z = np.zeros([n1,n2])
    for i in range(n1):
        for j in range(n2):
            Z[i,j] = neigh.predict(np.array([x1[i,j],x2[i,j]]).reshape(1,2))
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x1, x2, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens');
    ax.set_title('K-NN Regression function, k = {}'.format(n_neighbors))
    ax.set_xlabel("Edad")
    ax.set_ylabel("Tasa de impuesto")
    ax.set_zlabel("HPI x10^3")
    
    return ax
   
def knn_n(n_neighbors):
    zdata = output[0:100].values
    xdata = data['AGE'].iloc[0:100,].values
    ydata = data['TAX'].iloc[0:100,].values*1000000
    X = np.c_[xdata.reshape(100,1),ydata.reshape(100,1)]
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    neigh = KNeighborsRegressor(n_neighbors=2, algorithm = 'brute')
    neigh.fit(Xn, zdata.reshape(100,1))
    x1 = np.linspace(np.min(xdata), np.max(xdata), num=20)
    x2 = np.linspace(np.min(ydata), np.max(ydata), num=20)
    x1, x2 = np.meshgrid(x1, x2)
    n1,n2 = x1.shape
    Z = np.zeros([n1,n2])
    for i in range(n1):
        for j in range(n2):
            xval = np.array([x1[i,j],x2[i,j]]).reshape(1,2)
            xvaln = scaler.transform(xval)
            Z[i,j] = neigh.predict(xvaln)
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x1, x2, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens');
    ax.set_title('K-NN Regression function, k = {}'.format(n_neighbors))
    ax.set_xlabel("Edad")
    ax.set_ylabel("Tasa de impuesto")
    ax.set_zlabel("HPI x10^3")
    
    return ax

def ParzenPlot_un(h):
    zdata = output[0:100].values
    xdata = data['AGE'].iloc[0:100,].values
    ydata = data['TAX'].iloc[0:100,].values*1000000
    X = np.c_[xdata.reshape(100,1),ydata.reshape(100,1)]
    x1 = np.linspace(np.min(xdata), np.max(xdata), num=100)
    x2 = np.linspace(np.min(ydata), np.max(ydata), num=100)
    x1, x2 = np.meshgrid(x1, x2)
    n1,n2 = x1.shape
    Z = np.zeros([n1,n2])
    for i in range(n1):
        for j in range(n2):
            xval = np.array([x1[i,j],x2[i,j]]).reshape(1,2)
            Z[i,j] = nadaraya_watson(X,xval,zdata.reshape(100,1),h)
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x1, x2, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens');
    ax.set_title('Parzen Window Regression function, h = {}'.format(h))
    ax.set_xlabel("Edad")
    ax.set_ylabel("Tasa de impuesto")
    ax.set_zlabel("HPI x10^3")
    return ax

def ParzenPlot_n(h):
    zdata = output[0:100].values
    xdata = data['AGE'].iloc[0:100,].values
    ydata = data['TAX'].iloc[0:100,].values*1000000
    X = np.c_[xdata.reshape(100,1),ydata.reshape(100,1)]
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    x1 = np.linspace(np.min(xdata), np.max(xdata), num=100)
    x2 = np.linspace(np.min(ydata), np.max(ydata), num=100)
    x1, x2 = np.meshgrid(x1, x2)
    n1,n2 = x1.shape
    Z = np.zeros([n1,n2])
    for i in range(n1):
        for j in range(n2):
            xval = np.array([x1[i,j],x2[i,j]]).reshape(1,2)
            xvaln = scaler.transform(xval)
            Z[i,j] = nadaraya_watson(Xn,xvaln,zdata.reshape(100,1),h)
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x1, x2, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens');
    ax.set_title('Parzen Window Regression function, h = {}'.format(h))
    ax.set_xlabel("Edad")
    ax.set_ylabel("Tasa de impuesto")
    ax.set_zlabel("HPI x10^3")
    return ax
