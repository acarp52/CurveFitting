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

    plt.plot(xdata, ydata, 'ko')
    plt.show()

main()