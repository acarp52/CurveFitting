import numpy as np
import matplotlib.pyplot as plt
import math
"""
Final project for Stats 251

Authors:    Andrew Carpenter
            David Desrochers
            Cole Rogers
            Connor Coval
            Christopher Arras
"""

xvals = np.linspace(-1, 1, 20)
fn0 = lambda x: x*5
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

def makeFakeData(fn, m,b ):
    return fn(data)+np.random.normal(size=len(data))

def cofInt(xdata,m,b):
    estx=[x*m+b for x in xdata]
    #need tocalculate variance

    upper=[x + 1.644853*math.sqrt(var) for x in estx]
    lower=[x - 1.644853*math.sqrt(var) for x in estx]
    plt.plot(xdata,lower)
    plt.plot(xdata,upper)

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

    #m,b=coefficients=np.polyfit(xdata, ydata, 1)
    #cofInt(xdata,m,b)

    plt.plot(xdata, ydata, 'ko')

    yfit=bestFit(xdata,ydata)
    plt.plot(xdata,yfit)

    plt.show()



main()