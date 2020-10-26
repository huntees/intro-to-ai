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
    pca = PCA(n_components=150, whiten=True, random_state=42, svd_solver='randomized')
    pca.fit(faces['data'])

    ##Also, produce a graph plotting how explained variance changes with the number of components
    plt.figure
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.show()
    plt.close()


    #Then split your data into training and testing.
    X_train, X_test, y_train, y_test = train_test_split(faces['data'], faces['target'], random_state=42)

    #Transform your training and testing data using your PCA
    projection = pca.fit_transform(X_train)
    projectionTest = pca.fit_transform(X_test)

    # print("original shape:   ", X_train.shape)
    # print("transformed shape:", projection.shape)

    # Set up an SVM (it is suggest you use the 'rbf' kernal, gamma=0.001, C=5. Again, investigate changes to these.
    svm_model = SVC(kernel='rbf', class_weight='balanced', gamma='0.001', C=5)

    # Fit the model to the transformed data
    svm_model.fit(projection, y_train)

    # Predict the values of the training set
    y_pred = svm_model.predict(projectionTest)
    print(y_test)
    print(y_pred)

    # Calculate accuracy.
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    # From SOLUTION
    # use the classification report to get a more details summary of successes
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, target_names=faces.target_names))

    # show the PCA representations of the faces, and colour
    # code labels for success or failure.
    fig, ax = plt.subplots(3, 5)
    for i, axi in enumerate(ax.flat):
        axi.imshow(pca.components_[i].reshape(62, 47), cmap='bone')
        axi.set(xticks=[], yticks=[])
        axi.set_ylabel(faces.target_names[y_pred[i]].split()[-1],
                       color='black' if y_pred[i] == y_test[i] else 'red')
    fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)


if __name__ == '__main__':
    main()