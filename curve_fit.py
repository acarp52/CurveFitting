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
fn1 = lambda x: 3*np.ones(len(xvals))  #sort of like a function
fn2 = lambda x: -2+3.*x
fn3 = lambda x: 3*x**2
fnTrue = lambda x: 2+10*x-2*x**2

def fn0funct(x):
    return 0.1*x

def fn1func(x):
    ret = []
    for num in x:
        ret.append(3)
    return ret

def fn2func(x):
    ret = []
    for num in x:
        ret.append(-2.+3.*num)
    return ret

def fn3func(x):
    ret = []
    for num in x:
        ret.append(3.*num**2.)
    return ret

def bestFit(xdata, ydata):
    """return the y values representing the line of best fit"""
    coefficients=np.polyfit(xdata, ydata, 1)
    polynomial=np.poly1d(coefficients)
    return polynomial(xdata)

def makeFakeData(fn, data):
    return fn(data)+np.random.normal(size=len(data))
    # return np.random.normal(size=len(data))

def cdfQuadFit(fn, fnName):
    """plots CDFSs for the 3 coefficients of a quadratic for given function (fn)"""
    #lists of coefficients
    c0=[]
    c1=[]
    c2=[]
    xrd=np.linspace(-1,1,20)

    #generates 1000 synthetic sample sets of size 20 and quadratic coefficients for them
    for _ in range(1000):
        yrd=makeFakeData(fn,xrd)
        c2x, c1x, c0x=np.polyfit(xrd,yrd,2)
        c2.append(c2x)
        c1.append(c1x)
        c0.append(c0x)

    #cdf for c0
    counts,bins=np.histogram(c0,100)
    c0cum=np.cumsum(counts)
    plt.plot(bins[1:],c0cum/1000.0)
    plt.suptitle("CDF of c0 for function " +fnName)
    plt.show()

    #cdf for c1
    counts,bins=np.histogram(c1,100)
    c1cum=np.cumsum(counts)
    plt.plot(bins[1:],c1cum/1000.0)
    plt.suptitle("CDF of c1 for function " + fnName)
    plt.show()

    #cdf for c2
    counts,bins=np.histogram(c2,100)
    c2cum=np.cumsum(counts)
    plt.suptitle("CDF of c2 for function " + fnName)
    plt.plot(bins[1:],c2cum/1000.0)
    plt.show()

def scatLinFit(fn,fnName):
    """creates a scatterplot of linear regression coefficients for a given function (fn)"""

    m=[] #c1
    b=[] #c0
    xrd=np.linspace(-1,1,20)
    #gets linear regression coefficients of 1000 synthetic samples, each of size 20
    for _ in range(1000):
        yrd=makeFakeData(fn,xrd)
        mm,bb=np.polyfit(xrd,yrd,1)
        m.append(mm)
        b.append(bb)

    plt.scatter(b,m)
    plt.xlabel('y-intercept', fontsize=16)
    plt.ylabel('slope', fontsize=16)
    plt.suptitle("Linear regression coefficients of 1000 samples for function " + fnName)

    #trouble getting correlation
    #x,y=np.asarray(bb),np.asarray(mm)
    # R2 = np.corrcoef(x, y)[0, 1]**2  # coefficient of determination between x and y
   # print(R2)
    plt.show()


def linearfit(x, y, yerr):
    """Linear fit of x and y with uncertainty and plots results."""

    import numpy as np
    import scipy.stats as stats

    x, y = np.asarray(x), np.asarray(y)
    n = y.size
    p, cov = np.polyfit(x, y, 1, w=1/yerr, cov=True)  # coefficients and covariance matrix
    yfit = np.polyval(p, x)                           # evaluate the polynomial at x
    perr = np.sqrt(np.diag(cov))     # standard-deviation estimates for each coefficient
    R2 = np.corrcoef(x, y)[0, 1]**2  # coefficient of determination between x and y
    resid = y - yfit
    chi2red = np.sum((resid/yerr)**2)/(n - 2)  # Chi-square reduced
    s_err = np.sqrt(np.sum(resid**2)/(n - 2))  # stamdard deviation of the error (residuals)

    x2 = np.linspace(np.min(x), np.max(x), 100)
    y2 = np.linspace(np.min(yfit), np.max(yfit), 100)
    if p[0]<0:
        y2=np.linspace(np.max(yfit),np.min(yfit),100)

    # Confidence interval for the linear fit, with n-2 degrees of freedom:
    t = stats.t.ppf(0.90, n - 2)
    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2/np.sum((x-np.mean(x))**2))
    # Prediction interval for the linear fit:
    pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2/np.sum((x-np.mean(x))**2))

    # Plot
    #plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'ko')
    #plt.fill_between(x2, y2+pi, y2-pi, color=[1, 0, 0, 0.1], edgecolor='')
    #plt.fill_between(x2, y2+ci, y2-ci, color=[1, 0, 0, 0.15], edgecolor='')
    plt.plot(x2, y2+ci, 'r--')
    plt.plot(x2, y2-ci, 'r--')
    #plt.errorbar(x, y, yerr=yerr, fmt = 'bo', ecolor='b', capsize=0)
    plt.plot(x, yfit, 'b')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    #plt.title('$y = %.2f \pm %.2f + (%.2f \pm %.2f)x \; [R^2=%.2f,\, \chi^2_{red}=%.1f]$'
    #          %(p[1], perr[1], p[0], perr[0], R2, chi2red), fontsize=20, color=[0, 0, 0])
    plt.xlim((-1, 1))
    #plt.axis([-1, 1, -20, 20])
    plt.show()

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

    """
    cdfQuadFit(fn1, "a")
    cdfQuadFit(fn2, "b")
    cdfQuadFit(fn3, "c")
    """

    """
    scatLinFit(fn2,"b")
    """


    """
    yerr = np.abs(np.random.randn(len(ydata))) + 2
    linearfit(xdata, ydata, yerr)
    linearfit(xdata, makeFakeData(fn1func, ydata), yerr)
    linearfit(xdata, makeFakeData(fn2func, ydata), yerr)
    linearfit(xdata, makeFakeData(fn3func, ydata), yerr)
"""


"""
    print(makeFakeData(fn2func, ydata))
    print(np.var(ydata))
    print(np.cov(ydata))
    plt.plot(xdata, ydata, 'ko')
    yfit=bestFit(xdata,ydata)
    plt.axis([-1, 1, -20, 20])
    plt.plot(xdata,yfit)
    plt.show()


    print(makeFakeData(fn1func, ydata))
    plt.plot(xdata, makeFakeData(fn1func, ydata), 'ro')
    yfit=bestFit(xdata,makeFakeData(fn1func, ydata))
    plt.axis([-1, 1, -20, 20])
    plt.plot(xdata,yfit)
    plt.show()

"""

main()