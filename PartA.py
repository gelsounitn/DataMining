from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import fileinput
import argparse
import os

np.set_printoptions(threshold = np.inf)

def GetList(line):

    line_list = line.split(",")

    last_element = line_list.pop()
    last_element = last_element.replace('\n', '')

    line_list.append(last_element)

    return line_list

def ModifyRatings(tmp):

    line_list = []

    for rating in tmp:
        if rating == '':
            line_list.append(-1)
        else:
            line_list.append(int(rating))

    return line_list

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

    #user_rating_matrix = np.array(user_rating_matrix)
    user_rating_matrix = pd.DataFrame(user_rating_matrix, columns = matrix_query_ids)

    # test n. 1 (cosine similarity)
    knn = NearestNeighbors(metric = 'cosine')
    knn.fit(user_rating_matrix.values)
    distances, indices = knn.kneighbors(user_rating_matrix.values, n_neighbors = 10)

    for i in range(5):

        sim_queries = indices[i].tolist()
        query_distances = distances[i].tolist()

        id_query = sim_queries.index(i)
        #sim_queries.remove(id_query)
        #query_distances.pop(id_query)

        print(sim_queries)
        print(query_distances)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)

    main(args = parser.parse_args())