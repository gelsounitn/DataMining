from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import dask.dataframe as dd
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

    # normalize dataframe
    print("normalizing dataframe")
    scaler = StandardScaler(with_std = False)
    scaler.fit(df)
    #print(scaler.mean_)
    df = dd.from_array(scaler.transform(df))
    df = df.fillna(0)

    #print(df.head())

    # test number 1 (cosine similarity) knn
    print("using knn")
    knn = NearestNeighbors(metric = 'cosine')
    knn.fit(df)
    distances, indices = knn.kneighbors(df.values, n_neighbors = 300)

    print("converting to pandas df")
    pandas_df = df.compute()

    #st = time.time()
    print("computing missing values")
    for i in range(pandas_df.shape[1]):
        sim_queries, query_distances = GetNearestQuery(distances, indices, i)

        for j in range(pandas_df.shape[0]):
            ComputePredictedValue(sim_queries, query_distances, pandas_df, j, i)
        
        if(i % 10 == 0):
            print("calculating for column " + str(i))

    #et = time.time()
    #print(str(et - st) + " Seconds")

    print(pandas_df.iloc[0])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)

    main(args = parser.parse_args())