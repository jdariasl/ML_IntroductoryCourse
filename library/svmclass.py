from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# Create color maps for 3-class classification problem, as with iris
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    
    try:
        X, y = X.values, y.values
    except AttributeError:
        pass
    
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
    return plt

def DecisionBoundaryPlot():

	#-------------------- Generacion de Datos Artificiales --------------------------
	x1 = np.random.rand(50,2)
	x2 = np.random.rand(50,2) + 2
	Y = np.concatenate((np.ones((50,1)),np.zeros((50,1))),axis=0)
	X1 = np.concatenate((x1,x2),axis=0)
	X = np.concatenate((X1,np.ones((100,1))),axis=1)
	#-------------------- Declaracion de variables del Algoritmo de Gradiente
	#Error = np.zeros(MaxIter)
	w = np.ones(3).reshape(3, 1)
	MaxIter = 1000
	eta = 1
	N = X.shape[0]
	#------------------ Minimizacion --------------------------------------------------
	for i in range(MaxIter):
		tem = 1/(1 + np.exp(-1*np.dot(X,w)))
    	tem2 = tem-np.array(Y)
    	tem = np.dot(X.T,tem2)
    	w = w - eta*(1/N)*tem
	#-------------- Grafica de la Frontera --------------------------------------------
	x = np.linspace(-1,5,100).reshape(100, 1)
	print(w)
	y = -(w[0]/w[1])*x - (w[2]/w[1])
	#------------------------- Grafica ------------------------------------------------
	plt.title('Espacio de caracteristicas', fontsize=14)
	plt.xlabel('Caracteristica 1')
	plt.ylabel('Caracteristica 2')
	plt.scatter(x1[:,0], x1[:,1])
	plt.scatter(x2[:,0], x2[:,1],color='red')
	plt.plot(x,y,'k')

def PlotEjemploSVR(X,y,y_rbf,y_lin,y_poly):
	plt.scatter(X, y, c='k', label='data')
	plt.plot(X, y_rbf, c='g', label='RBF model')
	plt.plot(X, y_lin, c='r', label='Linear model')
	plt.plot(X, y_poly, c='b', label='Polynomial model')
	plt.xlabel('data')
	plt.ylabel('target')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()