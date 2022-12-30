import pandas as pd
import argparse
import re

datasets = ['movies']

def d_movies(file):

    print("Creating user set")
    user_set = set(list(file["User_Id"]))

    with open("data/movies_UserSet.txt", "w") as user_set_file:
        for user in user_set:
            user_set_file.write(str(user))
            user_set_file.write("\n")

    print("Creating movie set")
    tmp_movie_name = list(file["Movie_Name"])
    
    year = []
    movie_name = []
    movie_year_genre = set()
    for movie in tmp_movie_name:
        year += re.findall(r'(\d\d\d\d)', movie)
        
        m = movie[:len(movie) - 7]
        if "," in movie:
            m = '"' + m + '"'

        movie_name.append(m)
    
    tmp_genres = list(file["Genre"])

    genres = []
    for genre in tmp_genres:
        g = genre.split("|")
        genres.append(g[0])

    for i in range(len(movie_name)):
        movie_year_genre.add((str(movie_name[i]), year[i], genres[i]))

    print("Creating Database")
    with open("data/movies_Database.csv", "w") as database:
        database.write(",movie,year,genre\n")

        counter = 1
        for myg in movie_year_genre:
            database.write(str(counter) + "," + myg[0] + "," + myg[1] + "," + myg[2] + "\n")

            counter += 1
        
def main(args):
    d = args.d

    if type(d) != type(""):
        raise TypeError("The argument --d is not a string")
    elif d not in datasets:
        print("Select one dataset among {}".format(datasets))
        exit(1)

    dataset_path = "data/" + d + ".csv"

    print("Reading file .csv")
    file = pd.read_csv(dataset_path, engine = "pyarrow")

    if d == 'movies':
        d_movies(file)


    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type = str, required = True)

    main(args = parser.parse_args())