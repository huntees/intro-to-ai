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
from tensorflow.keras.layers import Dense
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

    # You might try varying the activation function, and/or the number of hidden units
    model.add(Dense(128, input_dim=X.shape[1], activation='sigmoid'))

    # you might experiment with a second hidden layer
    # model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(y.shape[1], activation='softmax'))

    # compile the model setting the loss (error) measure and the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Fir the model, you might change the number of epochs training is run for.
    model.fit(X_train, y_train, verbose=2, epochs=256)

    # model = Sequential()
    # model.add(Dense(64, input_dim=X.shape[1], activation='relu'))  # layer 1
    # model.add(Dense(64, activation='relu'))  # layer 2
    # model.add(Dense(10))
    # model.add(Dense(y.shape[1], activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    # model.fit(X_train, y_train, verbose=0, epochs=128)

    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)
    y_compare = np.argmax(y_test, axis=1)
    score = metrics.accuracy_score(y_compare, pred)

    print("Accuracy score model 1: {}".format(score))

    # # path to where the file will be saved
    # save_path = "."
    #
    # # # save neural network structure to JSON (no weights)
    # # model_json = model.to_json()
    # # with open(os.path.join(save_path, "network.json"), "w") as json_file:
    # #     json_file.write(model_json)
    #
    # # save entire network to HDF5 (save everything, suggested)
    # model.save(os.path.join(save_path, "network.h5"))
    #
    # model2 = load_model(os.path.join(save_path, "network.h5"))
    # pred = model2.predict(X_test)
    # pred = np.argmax(pred, axis=1)
    # y_compare = np.argmax(y_test, axis=1)
    # score = metrics.accuracy_score(y_compare, pred)
    # print("Accuracy score model 2: {}".format(score))

    # plt.show()

if __name__ == '__main__':
    main()