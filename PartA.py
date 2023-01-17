import numpy as np
import argparse
import os

np.set_printoptions(threshold = np.inf)

def main(args):
    path_folder = args.d

    if type(path_folder) != type(""):
        raise TypeError("The argument --d is not a string")
    elif not os.path.exists(path_folder):
        print("Missing folder")
        exit(1)

    os.chdir(path_folder)

    try:
        database = open("database.csv", "r")
        query_set = open("query_set.csv", "r")
        user_set = open("user_set.txt", "r")
        utility_matrix = open("utility_matrix.csv", "r")
    except Exception as e:
        print(e)

    utility_matrix_lines = utility_matrix.readlines()

    matrix_query_ids = utility_matrix_lines[0].split(",")
    matrix_query_ids[-1] = matrix_query_ids[-1].replace("\n", "")

    # utility matrix
    rating_matrix = np.empty((len(utility_matrix_lines) - 1, len(matrix_query_ids)))

    matrix_user_ids = []
    for i in range(1,len(utility_matrix_lines)):
        line = utility_matrix_lines[i]
        line_list = line.split(",")
        line_list[-1] = line_list[-1].replace("\n", "")

        # da sistemare
        # line_list = ["-1" if x == '' else x for x in line_list]
        # line_list = [int(x) for x in line_list[1:]]

        matrix_user_ids.append(line_list[0])

        #np_rating_line = np.array(line_list[1:])
        rating_matrix[i-1,:] = line_list[1:]

    database.close()
    query_set.close()
    user_set.close()
    utility_matrix.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)

    main(args = parser.parse_args())