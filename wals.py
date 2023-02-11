import scipy
from scipy import linalg
from scipy import sparse
import dask.array as da
import pandas as pd
import numpy as np
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import scipy.sparse
#import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np

def WALS(ratings, num_factors, weights, num_iters=20, lambda_reg=0.1):
    """
    Perform Weighted Alternating Least Squares to factorize the ratings matrix into user and item latent factor matrices.
    
    Parameters:
    - ratings (2D np.array): The matrix with the ratings (rows represent users and columns represent items)
    - num_factors (int): The number of latent factors
    - weights (2D np.array): The weight matrix (same shape as ratings matrix)
    - num_iters (int, optional): The number of iterations (default: 20)
    - lambda_reg (float, optional): The regularization parameter (default: 0.1)
    
    Returns:
    - user_matrix (2D np.array): The user latent factor matrix
    - item_matrix (2D np.array): The item latent factor matrix
    """
    num_users, num_items = ratings.shape
    user_matrix = np.random.normal(size=(num_users, num_factors))
    item_matrix = np.random.normal(size=(num_items, num_factors))
    
    for i in range(num_iters):
        # Update user matrix
        for u in range(num_users):
            # Get the indices of the items rated by user u
            item_indices = np.where(ratings[u, :] > 0)[0]
            # Get the ratings given by user u and the corresponding weight
            Ru = ratings[u, item_indices]
            Wu = weights[u, item_indices]
            # Calculate the weights-weighted sum of the item latent factors
            weighted_sum = np.dot(item_matrix[item_indices, :] * Wu[:, np.newaxis], 
                                  item_matrix[item_indices, :].T)
            # Add the regularization term
            weighted_sum += lambda_reg * num_items * np.eye(num_factors)
            # Solve the weighted least squares problem to update the user latent factors
            user_matrix[u, :] = np.linalg.solve(weighted_sum, np.dot(item_matrix[item_indices, :].T, Ru * Wu))
        
        # Update item matrix
        for j in range(num_items):
            # Get the indices of the users who rated item j
            user_indices = np.where(ratings[:, j] > 0)[0]
            # Get the ratings given to item j and the corresponding weight
            Rj = ratings[user_indices, j]
            Wj = weights[user_indices, j]
            # Calculate the weights-weighted sum of the user latent factors
            weighted_sum = np.dot(user_matrix[user_indices, :].T * Wj[:, np.newaxis], 
                                  user_matrix[user_indices, :])
            # Add the regularization term
            weighted_sum += lambda_reg * num_users * np.eye(num_factors)
            # Solve the weighted least squares problem to update the item latent factors
            item_matrix[j, :] = np.linalg.solve(weighted_sum, np.dot(user_matrix[user_indices, :].T, Rj * Wj))
    
    return user_matrix, item_matrix



def isNotNan(x):
    if(pd.isna(x)):
        return 0.
    else:
        return 1.

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
    df_copy = df.copy()

    df = df.fillna(0)

    M = tf.convert_to_tensor(df, dtype=tf.float32)

    df_copy = df_copy.applymap(isNotNan)

    sparsity_mat = tf.convert_to_tensor(df_copy, dtype=tf.float32)
    masked_entries = tf.cast(tf.not_equal(sparsity_mat, 1), dtype = 'float32')

    weights = sparsity_mat + 0.25*masked_entries

    df_copy = None

    R = df.to_numpy()

    U,V = WALS(R, 600, weights.numpy())

    print(np.dot(U,V)[0])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)

    main(args = parser.parse_args())
