from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# coding=utf-8
def mapFeature(X1, X2, degree = 6):

    Nm = X1.shape[0]       
    out = np.ones(Nm).reshape(Nm, 1)
    for i in range(1,degree+1):
        for j in range(i+1):
            tem = (X1**(i-j))*(X2**j)
            out = np.hstack([out,tem.reshape(Nm,1)])
    return out
def sigmoide(u):
    g = np.exp(u)/(1 + np.exp(u))
    return g

def plotDecisionBoundary(w, X, Y):
    plt.figure()
    plt.plot(X[Y.flat==1,1],X[Y.flat==1,2],'r+',label='Clase 1')
    plt.plot(X[Y.flat==0,1],X[Y.flat==0,2],'bo', label='Clase 0')

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [np.min(X[:,1])-2,  np.max(X[:,1])+2];

        # Calculate the decision boundary line
        plot_y = (-1/w[3])*(w[2]*plot_x + w[1]);

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y,color = 'black',label = 'Frontera de decision')
    
        # Legend, specific for the exercise
        plt.legend()
        #axis([30, 100, 30, 100])
    else:
        #Here is the grid range
        u = np.linspace(-1, 1.5, num=50);
        v = np.linspace(-1, 1.5, num =50);

        z = np.zeros([len(u), len(v)]);
        #Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = np.dot(mapFeature(u[i].reshape(1,1), v[j].reshape(1,1)),w)
        
        xv, yv = np.meshgrid(u, v)
        plt.contour(xv, yv, z.T, 0)

def StandardLogisticRegression(X,Y,lam=0):
    #Aprendizaje
    MaxIter = 100000
    eta = 10
    N = len(Y)
    Error =np.zeros(MaxIter)
    Xent = mapFeature(X[:,0], X[:,1])
    w = np.ones(Xent.shape[1]).reshape(Xent.shape[1], 1)

    for i in range(MaxIter):
        tem = np.dot(Xent,w)
        tem2 = sigmoide(tem.T)-Y
        Error[i] = np.sum(np.abs(tem2))/N
        tem = np.dot(Xent.T,tem2.T) + lam*np.concatenate((np.array([0]).reshape(1,1),w[1:]),axis=0)
        w = w - eta*(1/N)*tem
    print('Error=',Error[-1])
    plt.plot(Error)
    plt.title("Error de entrenamiento")
    plt.xlabel("Iteraciones")
    plotDecisionBoundary(w, Xent, Y);

def PrintOverfitting():
    print(__doc__)

    def true_fun(X):
        return np.cos(1.5 * np.pi * X)

    np.random.seed(0)

    n_samples = 30
    degrees = [1, 4, 15]

    X = np.sort(np.random.rand(n_samples))
    y = true_fun(X) + np.random.randn(n_samples) * 0.1

    plt.figure(figsize=(14, 5))
    for i in range(len(degrees)):
        ax = plt.subplot(1, len(degrees), i + 1)
        plt.setp(ax, xticks=(), yticks=())

        polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
        pipeline.fit(X[:, np.newaxis], y)

        # Evaluate the models using crossvalidation
        scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
        plt.plot(X_test, true_fun(X_test), label="True function")
        plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best")
        plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
            degrees[i], -scores.mean(), scores.std()))
    plt.show()

def main():
    print("Module Loaded")
if __name__ == '__main__':
    main()