import pandas as pd
import numpy as np
import argparse
import random
import math
import os

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

    # sampling elements to compute RMSE
    samples_array = []
    num_sample = 100
    iterations = 10
    
    for i in range(iterations):

        samples = []

        for _ in range(num_sample):
            random_column = random.randint(1,8000)
            random_row = random.randint(1,11999)

            while np.isnan(df.iat[random_row, random_column]) or (df.iat[random_row, random_column], random_row, random_column) in samples:
                random_column = random.randint(1,8000)
                random_row = random.randint(1,11999)

            samples.append((df.iat[random_row, random_column], random_row, random_column))
            df.iat[random_row, random_column] = np.nan

        samples_array.append(samples)

    # normalization of each matrix column
    print("Normalizing utility matrix")
    user_ratings_mean = df.mean(axis = 0)
    df = df.sub(user_ratings_mean, axis = 1)
    df = df.fillna(0)
    user_ratings_mean = user_ratings_mean.fillna(0)

    # SVD
    print("Computing SVD decomposition")
    U, eigenvalues, V = np.linalg.svd(df, full_matrices = False)

    # removing eigenvalues and computing RMSE
    for i in range(1,6):
        print("Reconstructing matrix with {}% of eigen values removed".format(i))

        k = int(len(eigenvalues)*(i/100))
        selected_eigenvalues = np.sort(eigenvalues)

        for j in range(k):
            selected_eigenvalues[j] = 0

        final_df = pd.DataFrame(np.dot(np.dot(U, np.diag(selected_eigenvalues)), V), columns = df.columns)
        final_df = final_df.add(user_ratings_mean, axis = 1)

        AVERAGE_RMSE = 0

        for samples in samples_array:

            RMSE = 0

            for sample in samples:
                initial_value = sample[0]
                row = sample[1]
                column = sample[2]

                estimated_value = final_df.iat[row, column]

                RMSE += math.sqrt(pow(initial_value - estimated_value,2))

            RMSE = RMSE / num_sample
            AVERAGE_RMSE += RMSE

        AVERAGE_RMSE = AVERAGE_RMSE / iterations

        print("Average RMSE calculated for {} random samples: {}".format(num_sample, AVERAGE_RMSE))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)

    main(args = parser.parse_args())