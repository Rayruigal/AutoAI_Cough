import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score
import matplotlib

# matplotlib.use('Agg')

def extra_stats(ytest, preds):

    R2 = r2_score(ytest, preds)
    print("R2 score : %.6f" % R2)

    mae = mean_absolute_error(ytest,preds)

    """
    print("Explained variance score: %.6f" % explained_variance_score(ytest,preds))
    print("Mean squared error: %.6f" % mean_squared_error(ytest,preds))
    print("Mean absolute error: %.6f" % mean_absolute_error(ytest,preds))
    print("Median absolute error: %.6f" % median_absolute_error(ytest,preds))
    """

    er = []
    g = 0
    for i in range(len(ytest)):
        # print( "actual=", ytest[i], " observed=", preds[i])
        x = (ytest[i] - preds[i]) **2
        er.append(x)
        g = g + x

    x = 0
    for i in range(len(er)):
        x = x + er[i]

    mse = x / len(er)

    v = np.var(er)

    m = np.mean(ytest)

    y = 0
    for i in range(len(ytest)):
        y = y + ((ytest[i] - m) ** 2)

    r2_calc = 1-(g/y)

    """
    print ("MSE: %.6f" % mse)
    print ("variance: %.6f" %  v)
    print ("average of errors: %.6f" % np.mean(er))
    print ("average of observed values: %.6f" % m)
    print ("total sum of squares: %.6f" % y)
    print ("total sum of residuals: %.6f" % g)
    print ("r2 calculated: %.6f" % r2_calc)
    """

    return R2, mae



def analyze_results_class(testY, preds_f, output_id, output_name, verbose=True, savefile=False, verbose2=False, train=False):

    both = np.empty((len(preds_f),2))
    if len(testY.shape) == 1:
        y = testY
        p = preds_f
    else:
        y = testY[:,output_id]
        p = preds_f[:, output_id]

    both[:,0] = y
    both[:,1] = p
    np.set_printoptions(precision=6, suppress=True)

    """
    if verbose or True:
        print("real vs predicted")
        print(both)
    """

    if savefile:
        if train:
            np.savetxt(output_name + ".train.csv", both, fmt="%.6f", delimiter=" ")
        else:
            np.savetxt(output_name + ".csv", both, fmt="%.6f", delimiter=" ")



def analyze_results(testY, preds_f, output_id, output_name, verbose=True, savefile=False, verbose2=False, train=False):

    both = np.empty((len(preds_f),2))
    if len(testY.shape) == 1:
        y = testY
        p = preds_f
    else:
        y = testY[:,output_id]
        p = preds_f[:, output_id]

    both[:,0] = y
    both[:,1] = p
    np.set_printoptions(precision=6, suppress=True)

    """
    if verbose or True:
        print("real vs predicted")
        print(both)
    """

    if savefile:
        if train:
            np.savetxt(output_name + ".train.csv", both, fmt="%.6f", delimiter=" ")
        else:
            np.savetxt(output_name + ".csv", both, fmt="%.6f", delimiter=" ")

    diff = p - y
    if verbose and False:
        plt.plot(y, 'ro')
        plt.plot(p, 'bx')
        plt.ylim((np.min([y.min(),p.min()]), np.max([y.max(),p.max()])))
        plt.title(output_name + ': Predicted (x) & Real (o)')
        plt.show()

    if verbose and False:
        colors = itertools.cycle(["r", "b", "g"])
        plt.title(output_name + ': Predicted & Real v2')
        for i in range(len(preds_f)):
            plt.plot([i, i], [both[i,0], both[i,1]], color=next(colors), marker='o', linestyle='dashed')
        plt.ylim((np.min([y.min(),p.min()]), np.max([y.max(),p.max()])))
        plt.show()

    if verbose2:
        fig, ax = plt.subplots()
        ax.scatter(y, p)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1)
        plt.xlim((np.min([y.min(),p.min()]), np.max([y.max(),p.max()])))
        plt.ylim((np.min([y.min(),p.min()]), np.max([y.max(),p.max()])))
        ax.margins(0.05)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.title(output_name)
        plt.show()

    if verbose and False:
        plt.plot(diff)
        plt.title(output_name + ': Predicted - Real')
        plt.show()

    R2, mae = extra_stats(y, p)

    idx = np.where(np.abs(y) < 1e-7)
    # print(idx[0])

    # print("{}: deleting {} elements".format(output_name, len(idx[0])))
    if len(idx[0]) > 0:
        for i in idx[0]:
           pass
           # print(i, y[i], p[i], p[i]-y[i])

    y = np.delete(y, idx)
    p = np.delete(p, idx)

    diff = p -  y
    percentDiff = (diff / y) * 100

    # plt.plot(percentDiff)
    # plt.title(output_name + ': (Predicted - Real)/Real (%)')
    # plt.show()

    absPercentDiff = np.abs(percentDiff)

    if verbose and False:
        plt.plot(absPercentDiff)
        plt.title(output_name + ': abs[(Predicted - Real)/Real] (%)')
        plt.show()

    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(absPercentDiff)
    std = np.std(absPercentDiff)

    print("[INFO] {} mean: {:.2f}%, std: {:.2f}%".format(output_name, mean, std))

    return R2, mae


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage: {} <filename>".format(sys.argv[0]))
        sys.exit(1)

    filename = sys.argv[1]

    dataframe = pd.read_csv(filename, delim_whitespace=True, header=None, comment='#')
    dataset = dataframe.values

    R2, mae = analyze_results(dataset[:,0], dataset[:,1], 0, filename, verbose=False, verbose2=True)

    print("R2=", R2)
    print("mae=", mae)
