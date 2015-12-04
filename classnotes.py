import numpy as np
import matplotlib.pyplot as plt


"""

Polyfit:
x, y --> a_hat, b_hat, c_hat

z = np.polyfit(x, y, z) --> z[0]x^2 = z[1]x = z[0]

minimize distance between data: sum((yk - [a_hat + b_hat*xk + c_hat*xk^2])/sigma)

ipython = very handy

"""

xvals = np.linspace(-1, 1, 20)
fn1 = lambda x: 3*np.ones(len(xvals)) #sort of like a function
fn2 = lambda x: -2+3.*x
fn3 = lambda x:3*x**2
fnTrue = lambda x: 2+10*x-2*x**2


def fn0(x):
    return 0.1*x


def makeFakeData(fn):
    return fn(xvals)+np.random.normal(size=len(xvals))

"""
fake data: f(xk) + epsilon_k : N(0,1)

Fit quadratic:
"""

yExamples = makeFakeData(fnTrue)

plt.plot(xvals, yExamples, 1)
plt.show()

z = np.polyfit(xvals, yExamples, 2)
fn = np.poly1d(z)

xvals_show = np.linspace(-1, 1, 200)
plt.plot(xvals, yExamples, 'o')
plt.plot(xvals_show, fn(xvals_show))
plt.show


def get_random_coefs(deg):
    yvals = makeFakeData(fnTrue)
    return np.polyfit(xvals, yvals, deg)

print(get_random_coefs(2))
