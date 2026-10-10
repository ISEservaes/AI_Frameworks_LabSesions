import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

import pandas as pd

import os

file = r'ex1data2.txt'

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
        df = pd.read_csv(file, names=['size', 'bedrooms',"price"])
    else:
        print("file not found")

    x = df[['size', 'bedrooms']]  # dubbele haakjes: 2D
    y = df["price"]
    return df,x, y

#model,y_voorspelling = rum_model(x, y)
def rum_model(x,y):
    model = LinearRegression()
    model.fit(x, y)


    y_voorspelling = model.predict(x)
    print('cost:', mean_squared_error(y, y_voorspelling) / 2)
    print('voorspelling voor 35.000 inwoners:', model.predict([[1650,3]])[0])

    return model,y_voorspelling


df,x,y = getData(file)


antwoord = getNum()

if(antwoord == 1):
    model,y_voorspelling = rum_model(x, y)
    plt.scatter(y, y_voorspelling)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
    plt.xlabel('real price')
    plt.ylabel('price prediction')
    plt.show()
else:
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    model, y_voorspelling = rum_model(x_train, y_train)

    y_test_voorspelling = model.predict(x_test)

    plt.scatter(y_train, y_voorspelling, color='green')
    plt.scatter(y_test, y_test_voorspelling, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
    plt.xlabel('real price')
    plt.ylabel('price prediction')
    plt.show()















