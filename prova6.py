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
import tensorflow as tf
import math
import multiprocessing as mp

np.set_printoptions(threshold = np.inf)

def early_stopping(losses, patience = 3):
     
    if len(losses) <= patience + 1:
        return False
     
    avg_loss = np.mean(losses[-1 - patience:-1])

    if avg_loss - losses[-1] < 0.01 * avg_loss or losses[-1] < 0.001 :
        return True
     
    return False

def isNotNan(x):
    if(pd.isna(x)):
        return 0.
    else:
        return 1.

def WALS(R, M, K, weights=None, iterations=10, lambda_reg=0.1):
    """
    Weighted Alternating Least Squares (WALS) algorithm for matrix factorization.
    
    Parameters:
    R (array-like): The input matrix to factorize.
    M (int): Number of rows in R.
    K (int): Number of columns in R.
    weights (array-like, optional): Weights for each element in R. Defaults to None.
    iterations (int, optional): Number of iterations for WALS. Defaults to 10.
    lambda_reg (float, optional): Regularization factor for WALS. Defaults to 0.1.
    
    Returns:
    tuple: Two matrices U and V such that R ≈ UV'
    """
    if weights is None:
        weights = np.ones((M, K))
    
    U = np.random.rand(M, K)
    V = np.random.rand(K, K)
    
    for i in range(iterations):
        for j in range(M):
            W = np.diag(weights[j, :])
            rows = np.nonzero(R[j, :])[0]
            if len(rows) == 0:
                continue
            U[j, :] = np.linalg.solve(np.dot(V[:, rows].T, np.dot(W, V[:, rows])) + lambda_reg * np.eye(K),
                                      np.dot(V[:, rows].T, np.dot(W, R[j, rows].T))).T
        for j in range(K):
            W = np.diag(np.array([weights[i, j] for i in np.nonzero(R[:, j])[0]]))
            cols = np.nonzero(R[:, j])[0]
            if len(cols) == 0:
                continue
            V[:, j] = np.linalg.solve(np.dot(U[cols, :].T, np.dot(W, U[cols, :])) + lambda_reg * np.eye(K),
                                      np.dot(U[cols, :].T, np.dot(W, R[cols, j])))
    
    return U, V

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

    n_rows = df.shape[0]
    n_columns = df.shape[1]
    #12000 * 8001 -> 600
    uv_dimension = math.ceil((n_rows*n_columns) ** (1/3))
    print("uv dimension: " + str(uv_dimension))

    # normalization of each matrix column
    print("Normalizing utility matrix")
    user_ratings_mean = df.mean(axis = 0)
    df = df.sub(user_ratings_mean, axis = 1)
    df_copy = df.copy()

    df = df.fillna(0)
    #user_ratings_mean = user_ratings_mean.fillna(0)

    #matplotlib inline
 
    #U = tf.random.normal((500,500), mean=0, stddev=16, dtype = 'float32')
    #V = tf.random.normal((500,500), mean=0, stddev=16, dtype = 'float32')
    
    #print(df.head())

    M = tf.convert_to_tensor(df, dtype=tf.float32)

    #harm = np.array([1/(i + 1) for i in range(500) ])
    #plt.plot(harm)
    #plt.show()

    #print(M[2])
    #print(type(M[2][0]))
    #print(tf.not_equal(M, np.nan)[2])

    df_copy = df_copy.applymap(isNotNan)
    #print(df_copy.head())

    sparsity_mat = tf.convert_to_tensor(df_copy, dtype=tf.float32)
    #sparsity_mat = tf.cast(tf.map_fn(isNotNan, M), dtype = 'float32')
    masked_entries = tf.cast(tf.not_equal(sparsity_mat, 1), dtype = 'float32')

    df_copy = None
    #print(sparsity_mat[2])
    #print(masked_entries[2])

    w_matrix = sparsity_mat + 0.2*masked_entries
    Uwals, Vwals = WALS(M, n_rows, n_columns, w_matrix, 10, 0.1)
    print((Uwals @ Vwals)[0])

    exit(1)

    U_d = tf.Variable(tf.random.normal((n_rows, uv_dimension), mean=0, stddev=1))
    V_d = tf.Variable(tf.random.normal((uv_dimension, n_columns), mean=0, stddev=1))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=.4,
    decay_steps=100.,
    decay_rate=0.96,
    staircase=False
    )

    adam_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    from datetime import datetime
    ep = 0
    start_time = datetime.now()
    
    losses = []
    val_losses = []
    weighted_losses = []
    weight = 0.2
    
    train_norm = tf.reduce_sum(sparsity_mat)
    #print("train_norm ", train_norm)
    val_norm = tf.reduce_sum(masked_entries)
    #print("val_norm", val_norm)
    
    #print((U_d @ V_d)[0])

    while True:
        
        with tf.GradientTape() as tape:
            M_app = U_d @ V_d
            
            pred_errors_squared = tf.square(M - M_app)
            #print("pred_errors_squared", pred_errors_squared[0])
            loss = tf.reduce_sum((sparsity_mat * pred_errors_squared)/train_norm)
            
            val_loss = tf.reduce_sum((masked_entries * pred_errors_squared)/val_norm)

            weighted_loss = loss + weight * val_loss
    
        if ep%10 == 0:
            print(datetime.now() - start_time, loss, val_loss, weighted_loss, ep)
            losses.append(loss.numpy())
            val_losses.append(val_loss.numpy())
            weighted_losses.append(weighted_loss.numpy())
            print((U_d @ V_d)[0][3])
            #print(losses)
            #print(val_losses)
        if early_stopping(weighted_losses): #val_losses
            break
        
        grads = tape.gradient(weighted_loss, [U_d, V_d])
        adam_opt.apply_gradients(zip(grads, [U_d, V_d]))
    
        ep += 1
    
    print('total time: ', datetime.now() - start_time)
    print('epochs: ', ep)

    final_matrix = tf.cast(U_d @ V_d, dtype=tf.int32)
    print(df.head())
    print(final_matrix[0])


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)

    main(args = parser.parse_args())