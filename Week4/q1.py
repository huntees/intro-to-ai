import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.mixture import GaussianMixture

def main():
    #load the iris dataset and load it into a DataFrame
    iris = datasets.load_iris()

    #build a principal component analysis, with two components
    pca = PCA(2)    #pca = PCA(n_components=2)

    #fit this model and transform the iris data, projecting it onto the two principal components
    projected = pca.fit_transform(iris.data)

    print("original shape:   ", iris.data.shape)
    print("transformed shape:", projected.shape)

    #plot the transformed, labelled, data as a scatter plot and observe the separation of the three species
    # plt.scatter(projected[:, 0], projected[:, 1]);
    # plt.xlabel('component 1')
    # plt.ylabel('component 2')
    # plt.colorbar();


    #### kMeans ######
    
    #build a kMeans model with 3 clusters
    kmeans = KMeans(n_clusters=3)

    #fit this model to the iris data and use this model to make prediction for each data (i.e. which cluster it is in)
    clusters = kmeans.fit_predict(iris.data)

    #use a mask to produce a set of labels for the prediction to match the dataset
    kmeansLabels = np.zeros_like(clusters)
    for i in range(3):
        mask = (clusters == i)
        kmeansLabels[mask] = mode(iris.target[mask])[0]

    #use this analyse the accuracy of the clusters
    print('kMeans Accuracy: %f' % accuracy_score(iris.target, kmeansLabels))

    #### Gaussian ######

    #build a Gaussian mixture model with 3 clusters and fit this model to the iris data
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0).fit(iris.data)

    #use this model to make prediction for each data (i.e. which cluster it is in)
    #use a mask to produce a set of labels for the prediction to match the dataset
    gmmLabels = gmm.predict(iris.data)

    #use this analyse the accuracy of the clusters
    print('GMM Accuracy: %f' % accuracy_score(iris.target, gmmLabels))

if __name__ == '__main__':
    main()