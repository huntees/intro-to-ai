import pandas as pd
import io
import requests
import numpy as np
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import base64
import matplotlib.pyplot as plt
import requests
from sklearn import preprocessing
from tensorflow.keras.models import load_model


def main() :
    digits = datasets.load_digits()

    # set features and target
    X = digits.data
    y = digits.target

    # use the keras built in to ensure the targets are categories
    y = keras.utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # set up a keras model.
    model = Sequential()

    model.add(Dense(128, input_dim=X.shape[1], activation='sigmoid'))

    #model.add(Dropout(0.04))

    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    monitor = EarlyStopping(monitor='loss', min_delta=1e-4, patience=5, verbose=1, mode='auto')

    model.fit(X_train, y_train, verbose=2, epochs=256)

    training_trace = model.fit(X_train, y_train, callbacks=[monitor], validation_split=0.25, verbose=0, epochs=1000)

    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)
    y_compare = np.argmax(y_test, axis=1)
    score = metrics.accuracy_score(y_compare, pred)

    print("Accuracy score model : {}".format(score))

    ## plot the loss on the training data, and also the validation data
    plt.figure(figsize=(10, 10))

    plt.plot(training_trace.history['loss'])
    plt.plot(training_trace.history['val_loss'])
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

if __name__ == '__main__':
    main()