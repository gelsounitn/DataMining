from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import fileinput
import argparse
import scipy
import math
import os

np.set_printoptions(threshold = np.inf)

# transfrom each read line into a list
def GetList(line):

    line_list = line.split(",")

    last_element = line_list.pop()
    last_element = last_element.replace('\n', '')

    line_list.append(last_element)

    return line_list

# transform ratings from strings to integers
def ModifyRatings(tmp):

    line_list = []

    for rating in tmp:
        if rating == '':
            line_list.append(-1)
        else:
            line_list.append(int(rating))

    return line_list

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

    if user_rating_matrix.iat[j, i] == -1:         
                
        predicted_rating = 0
        distance_sum = 0

        for k in range(len(sim_queries)):
            if number_similar_query > 10:
                break

            if user_rating_matrix.iat[j, k] != -1:
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

    matrix_query_ids = []
    matrix_user_ids = []
    user_rating_matrix = []

    index = 0
    # reading utility matrix
    for line in fileinput.input(['utility_matrix.csv']):

        line_list = GetList(line)
        if index == 0:
            matrix_query_ids = line_list
        else:
            matrix_user_ids.append(line_list[0])
            line_list = ModifyRatings(line_list[1:])
            user_rating_matrix.append(line_list)

        index += 1

    user_rating_matrix = pd.DataFrame(user_rating_matrix, columns = matrix_query_ids)
    user_rating_matrix = np.vstack([user_rating_matrix])

    import time

    # test number 1 (cosine similarity)
    knn = NearestNeighbors(metric = 'cosine')
    knn.fit(user_rating_matrix.values)
    distances, indices = knn.kneighbors(user_rating_matrix.values, n_neighbors = int(math.sqrt(len(matrix_query_ids))))
    print("Finished KNN")

    st = time.time()
    for i in range(len(matrix_query_ids)):
        sim_queries, query_distances = GetNearestQuery(distances, indices, i)

        mean = 0

        for j in range(len(matrix_user_ids)):
            if user_rating_matrix.iat[j, i] != -1:
                mean += user_rating_matrix.iat[j, i]
        
        for j in range(len(matrix_user_ids)):
            if user_rating_matrix.iat[j, i] != -1:
                user_rating_matrix.iat[j, i] - mean

        for j in range(1):
            ComputePredictedValue(sim_queries, query_distances, user_rating_matrix, j, i)

    et = time.time()
    print(str(et - st) + " Seconds")

    print(user_rating_matrix.iloc[0])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)

    main(args = parser.parse_args())