import numpy as np
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def main():
    faces = fetch_lfw_people(min_faces_per_person=60)
    print(faces.target_names)
    print(faces.images.shape)

    fig, ax = plt.subplots(3, 5)
    for i, axi in enumerate(ax.flat):
        axi.imshow(faces.images[i], cmap='bone')
        axi.set(xticks=[], yticks=[],
                xlabel=faces.target_names[faces.target[i]])
    plt.show()
    plt.close()

    #Build randomized PCA with 150 components
    pca = PCA(n_components=150, svd_solver='randomized')
    pca.fit(faces.data)

    ##Also, produce a graph plotting how explained variance changes with the number of components
    # print(pca.explained_variance_)
    #
    # def draw_vector(v0, v1, ax=None):
    #     ax = ax or plt.gca()
    #     arrowprops = dict(arrowstyle='->',
    #                       linewidth=2,
    #                       shrinkA=0, shrinkB=0, color='k')
    #     ax.annotate('', v1, v0, arrowprops=arrowprops)
    #
    # # plot data
    # plt.scatter(faces.data[:, 0], faces.data[:, 1], alpha=0.2)
    # for length, vector in zip(pca.explained_variance_, pca.components_):
    #     v = vector * 3 * np.sqrt(length)
    #     draw_vector(pca.mean_, pca.mean_ + v)
    # plt.axis('equal');


    #Then split your data into training and testing.
    X_train, X_test= train_test_split(faces.data, test_size=0.25, random_state=42)

    #Transform your training and testing data using your PCA
    X_train_pro = pca.fit_transform(X_train)
    X_test_pro = pca.fit_transform(X_test)

    # print("original shape:   ", X_train.shape)
    # print("transformed shape:", X_train_pro.shape)
    # print("original shape:   ", X_test.shape)
    # print("transformed shape:", X_test_pro.shape)

    # Set up an SVM (it is suggest you use the 'rbf' kernal, gamma=0.001, C=5. Again, investigate changes to these.
    svm_model = SVC(kernel='rbf', gamma='0.001', C=5)

    # Fit the model to the transformed data
    svm_model.fit(X_train_pro)

    # Predict the values of the training set
    X_pred = svm_model.predict(X_train_pro)
    print(X_test_pro)
    print(X_pred)

    # Calculate accuracy.
    print('Accuracy: %.2f' % accuracy_score(X_test_pro, X_pred))


if __name__ == '__main__':
    main()