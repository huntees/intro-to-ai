import os
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    #load the iris dataset and load it into a DataFrame
    iris = datasets.load_iris()

    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=np.append(iris['feature_names'], ['target']))

    #shuffle data
    df = df.reindex(np.random.permutation(df.index))

    # print first five
    print(df.head())

    #select columns for X
    result = []
    for x in df.columns:
        if x != 'target':
            result.append(x)

    #assign X and y
    X = df[result].values
    y = df['target'].values
    variety = iris.target_names

    #split the dataset into X (the features) and y (the target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # plot a 3 x 2 figure comparing the feature against each other
    plt.figure()

    plt.subplot(321)
    plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=y, s=50)

    plt.subplot(322)
    plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], c=y, s=50)

    plt.subplot(323)
    plt.scatter(df['sepal length (cm)'], df['petal width (cm)'], c=y, s=50)

    plt.subplot(324)
    plt.scatter(df['sepal width (cm)'], df['petal length (cm)'], c=y, s=50)

    plt.subplot(325)
    plt.scatter(df['sepal width (cm)'], df['petal width (cm)'], c=y, s=50)

    plt.subplot(326)
    plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c=y, s=50)

    #build svm model and fit to the training data
    svm_model = SVC(gamma='scale', decision_function_shape='ovo')
    svm_model.fit(X_train, y_train)

    #predict values for the testing data
    y_pred = svm_model.predict(X_test)
    #print(y_test)
    #print(y_pred)

    # put the results into a DataFrame and print side-by-side
    output = pd.DataFrame(data=np.c_[y_test, y_pred])
    print(output)

    #print accuracy
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    # find the confusion matrix, normalise and print
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)

    # confusion matric as a figure
    plt.figure()
    plot_confusion_matrix(cm_normalized, variety, title='Normalized confusion matrix')
    plt.show()

if __name__ == '__main__':
    main()
    #plot_confusion_matrix()