import numpy as np
import matplotlib.pyplot as plt

"""
Final project for Stats 251

Authors:    Andrew Carpenter
            David Desrochers
            Cole Rogers
            Connor Coval
            Christopher Arras
"""

xvals = np.linspace(-1, 1, 20)
fn0 = lambda x: 0.1*x
fn1 = lambda x: 3*np.ones(len(xvals)) #sort of like a function
fn2 = lambda x: -2+3.*x
fn3 = lambda x:3*x**2
fnTrue = lambda x: 2+10*x-2*x**2

def fn0funct(x):
    return 0.1*x

def bestFit(xdata, ydata):
    """return the y values representing the line of best fit"""
    coefficients=np.polyfit(xdata, ydata, 1)
    polynomial=np.poly1d(coefficients)
    return polynomial(xdata)

def makeFakeData(fn, data):
    return fn(data)+np.random.normal(size=len(data))

def main():
    with open("classB7.dat", "r") as ins:
        data = []
        for line in ins:
            tempArr = []
            for num in line.split('\t'):
                tempArr.append(float(num))
            data.append(tempArr)

    xdata = data[0]
    ydata = data[1]
    print(xdata)
    print(ydata)

    # print(np.linspace(-1, 1, 20))
    # print(fn0(ydata)+np.random.normal(size=len(ydata)))
    # print(fn0funct(ydata)+np.random.normal(size=len(ydata)))

    plt.plot(xdata, ydata, 'ko')

    yfit=bestFit(xdata,ydata)
    plt.plot(xdata,yfit)

    plt.show()



main()