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


# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

# Encode text values to a single dummy variable.  The new columns (which do not replace the old) will have a 1
# at every location where the original column (name) matches each of the target_values.  One column is added for
# each target value.
def encode_text_single_dummy(df, name, target_values):
    for tv in target_values:
        l = list(df[name].astype(str))
        l = [1 if str(x) == str(tv) else 0 for x in l]
        name2 = f"{name}-{tv}"
        df[name2] = l

# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()
    if sd is None:
        sd = df[name].std()
    df[name] = (df[name] - mean) / sd

# Convert all missing values in the specified column to the median
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)

# Convert all missing values in the specified column to the default
def missing_default(df, name, default_value):
    df[name] = df[name].fillna(default_value)

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    # Regression
    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"

# Regression chart.
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

# Remove all rows where the specified column is +/- sd standard deviations
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean())
                          >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)

# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])
    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
        * (normalized_high - normalized_low) + normalized_low




def main() :

    digits = datasets.load_digits()

    df = pd.DataFrame(data=np.c_[digits['data'], digits['target']],
                      columns=np.append(digits['feature_names'], ['target']))

    np.random.seed(42)
    df = df.reindex(np.random.permutation(df.index))

    #print(df)

    target = encode_text_index(df, 'target')
    X, y = to_xy(df, 'target')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu')) # layer 1
    model.add(Dense(64, activation='relu')) #layer 2
    model.add(Dense(10))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=128)
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