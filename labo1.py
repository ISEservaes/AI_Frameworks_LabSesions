import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

import pandas as pd

import os

file = r'ex1data1.txt'

def getNum():
    Invalid_Num = True
    while (Invalid_Num):
        print("with train test split yes[1] no[2]?\t:")
        keuze = int(input())
        if (keuze == 1 or keuze == 2):
            Invalid_Num = False

    return keuze


def getData(file):
    if(os.path.exists(file)):
        df = pd.read_csv(file, names=['x', 'y'])
    else:
        print("file not found")

    x = df[['x']]  # dubbele haakjes: 2D
    y = df['y']
    return df,x, y



def rum_model(x,y):
    model = LinearRegression()
    model.fit(x, y)

    y_voorspelling = model.predict(x)

    return model,y_voorspelling


df,x,y = getData(file)


antwoord = getNum()

if(antwoord == 1):
    model,y_voorspelling = rum_model(x, y)
    plt.scatter(x, y)
    plt.plot(x, y_voorspelling, color='red')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()
else:
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    model, y_voorspelling = rum_model(x_train, y_train)


    plt.scatter(x_train, y_train, color='green')
    plt.scatter(x_test, y_test, color='blue')
    plt.plot(x_train, y_voorspelling, color='red')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()















