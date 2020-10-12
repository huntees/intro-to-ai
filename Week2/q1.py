import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#a function to plot confusion matrices
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

#load the dataset
wine = datasets.load_wine()

#into a DataFrame
df = pd.DataFrame(data=np.c_[wine['data'],wine['target']],
                  columns=np.append(wine['feature_names'],['target']))

#shuffle the data
df = df.reindex(np.random.permutation(df.index))

#print the first five entries
print(df.head())

#select columns for X
result = []
for x in df.columns:
    if x != 'target':
        result.append(x)

#Assign X and y        
X = df[result].values
y = df['target'].values
bottles = wine.target_names

#perform a single split, with 25% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=42) 

#build a decision tree and fit to the training data
tree = DecisionTreeClassifier(criterion = 'entropy').fit(X_train,y_train)

#predict values for the testing data
y_pred = tree.predict(X_test)
print(y_test)
print(y_pred)

#print accuracy (other metrics are available)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#use the confusion matrix function to build a confusion matrix
#(twice) once with numbers, once with proportions
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)

#give a grpahical representation
plt.figure()
plot_confusion_matrix(cm, bottles, title='')

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, bottles, title='Normalized confusion matrix')
plt.show()

