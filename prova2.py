from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import dask.dataframe as dd
import scipy
from scipy import linalg
from scipy import sparse
import dask.array as da
import pandas as pd
import numpy as np
import fileinput
import argparse
import time
import math
import os

np.set_printoptions(threshold = np.inf)

# calculate cosine similarity
def GetNearestQuery(indices, distances, i):

    sim_queries = indices[i].tolist()
    sim_queries = sim_queries[1:]
    
    query_distances = distances[i].tolist()
    query_distances = query_distances[1:]

    return (sim_queries, query_distances)

# compute the prediction for the user query
def ComputePredictedValue(sim_queries, query_distances, user_rating_matrix, j, i):

    number_similar_query = 0

    if user_rating_matrix.iat[j, i] == 0:         
                
        predicted_rating = 0
        distance_sum = 0

        for k in range(len(sim_queries)):
            if number_similar_query > 10:
                break

            if user_rating_matrix.iat[j, k] != 0:
                predicted_rating += query_distances[k] * user_rating_matrix.iat[j, k]
                distance_sum += query_distances[k]

                number_similar_query += 1

        predicted_rating = int(predicted_rating / distance_sum)

        user_rating_matrix.iat[j, i] =  predicted_rating

    return

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
    print("reading utility matrix")
    df = dd.read_csv("utility_matrix.csv", blocksize="128MB")

    print(df.head())

    # normalize dataframe
    print("normalizing dataframe")
    scaler = StandardScaler(with_std = False)
    scaler.fit(df)
    print(scaler.mean_)
    df = dd.from_array(scaler.transform(df), columns=df.columns)
    df = df.fillna(0)

    print(df.head())

    print("converting to pandas df")
    df = df.compute()
    m = df.shape[0]
    n = df.shape[1]

    print(df.iloc[0,:])
    print(df.iloc[1,:])

    print("SVD")
    U, s, Vh = scipy.linalg.svd(df, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')

    print("s = " + s)

    print("reconstructing")
    sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        sigma[i, i] = s[i]
    a1 = np.dot(U, np.dot(sigma, Vh))


    print(sigma[0,:])

    #sdf = df.astype(pd.SparseDtype("float", 0.0))
    #print('dense : {:0.2f} bytes'.format(df.memory_usage().sum() / 1e3))
    #print('dense : {:0.2f} bytes'.format(sdf.memory_usage().sum() / 1e3))
    #print(pandas_df.iloc[0])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)

    main(args = parser.parse_args())