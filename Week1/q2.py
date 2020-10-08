import os
import pandas as pd

path = "./../../Intro to AI/ex1/"


# Load the dataset
def loadData():
    filename_read = os.path.join(path, "iris.csv")
    df = pd.read_csv(filename_read)

    return df


# Print the dataset
def printData(df):
    print(df)


# Print the rows in the dataset where petal_w < 1.0
def printSmallPetals(df):
    for i in df.index:
        if (df.iloc[i, 3] < 1.0):  # 3 indicates column 3, note 3 is 4th row because 0 index
            print(df.iloc[i].values)


# Sort the dataset by sepal_l and print this
def printSortedBySepalLength(df):
    dfs = df.sort_values(by='sepal_l', ascending=True)
    print(dfs)
    return dfs


# Save the sorted dataset to a new file
def saveFile(df):
    filename_write = os.path.join(path, "sorted_iris.csv")
    df.to_csv(filename_write, index=False)
    print("Filed Saved")


# Calculate the variance of sepal_l
def sepalLengthVariance(df):
    # calculate variance
    # first mean
    mean = df['sepal_l'].mean()

    # then sum of square of difference to mean
    sum = 0
    for val in df['sepal_l']:
        sum += (mean - val) ** 2

    # divide by the number of values
    variance = sum / len(df.index)
    print("variance: " + str(variance))

if __name__ == '__main__':
    df = loadData()
    # printData(df)
    # printSmallPetals(df)
    dfs = printSortedBySepalLength(df)
    saveFile(dfs)
    sepalLengthVariance(dfs)

