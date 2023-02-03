import scipy
from scipy import linalg
from scipy import sparse
import dask.array as da
import pandas as pd
import numpy as np
import argparse
import os
import scipy.sparse
import matplotlib.pyplot as plt

np.set_printoptions(threshold = np.inf)

def rmse(util_mtx, p, q):
    e = 0.
    m = 0.
    r,c = util_mtx.shape
    for i in range(r):
        for j in util_mtx[i].indices:
            e += (util_mtx[i,j]-np.dot(q[j], p[i]))**2
            m+=1
    return np.sqrt(e/m)

def sgd_uv(util_mtx, f=5, lr=0.001, reg=0.1):
    err_arr = []
    r,c = util_mtx.shape
    #item matrix
    q = np.random.normal(0,16,(c,f))
    #user matrix
    p = np.random.normal(0,16,(r,f))
    #fit the matrix
    for t in range(5):
        print("cycle " + str(t))
        for i in range(r):
            for j in range(c):
                err = util_mtx[i,j] - np.dot(q[j], p[i])
                q[j] = q[j] + lr*(err*p[i]-reg*q[j])
                p[i] = p[i] + lr*(err*q[j]-reg*p[i])
                
            print("row " + str(i))
        err_arr.append(rmse(util_mtx,p,q))
    return p,q,err_arr

def main(args):
    path_folder = args.d

    # checking input
    if type(path_folder) != type(""):
        raise TypeError("The argument --d is not a string")
    elif not os.path.exists(path_folder):
        print("Missing folder")
        exit(1)

    # changing working directory
    os.chdir(path_folder)

    # reading file
    print("Reading utility matrix")
    df = pd.read_csv("utility_matrix.csv")
    pd.set_option("display.precision", 10)

    row = df.shape[0]
    column = df.shape[1]

    # normalization of each matrix column
    print("Normalizing utility matrix")
    user_ratings_mean = df.mean(axis = 0)
    df = df.sub(user_ratings_mean, axis = 1)
    df = df.fillna(0)
    #user_ratings_mean = user_ratings_mean.fillna(0)

    p,q,err_arr = sgd_uv(scipy.sparse.dok_matrix(df.values), f=5, lr=1, reg=0.1)
    plt.plot(err_arr, linewidth=2)
    plt.xlabel('Iteration, i', fontsize=20)
    plt.ylabel('RMSE', fontsize=20)
    plt.title('Components = '+str(5), fontsize=20)
    plt.show()

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)

    main(args = parser.parse_args())