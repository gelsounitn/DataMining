import pandas as pd
import argparse
import random
import ast
import os

datasets = ['movies', 'music']

def movies():

    print("Creating dataset directory")
    if not os.path.exists("data/movies"):
        os.mkdir("data/movies")

    print("Opening .csv file")
    u = pd.read_csv("data/movies/user_data.csv", engine = "pyarrow")
    m = pd.read_csv("data/movies/movie_data.csv", engine = "pyarrow")

    id = m.movie_id
    title = m.movie_title
    year = m.movie_release_year
    popularity = m.movie_popularity
    dir_name = m.director_name

    print("Creating user set")
    user_set = set(u.user_id)
    with open("data/movies/user_set.txt", "w") as file:
        for user in user_set:
            file.write(str(user))
            file.write("\n")

    print("Creating database")
    movie_info = set()

    for i in range(len(id)):
        movie_info.add((id[i], title[i], round(year[i]), popularity[i], dir_name[i]))

    with open("data/movies/database.csv", "w") as database:
        database.write("id,title,year,popularity,director_name\n")

        for movie in movie_info:
            directors_name = list(movie[4].split(","))

            d_name_str = "["
            for d_name in directors_name:
                d_name_str += d_name + ";"

            d_name_str = d_name_str[:-1]
            d_name_str += "]"

            t = movie[1]
            if "," in t:
                pass
            else:
                database.write(str(movie[0]) + "," + movie[1] + "," + str(movie[2]) + "," + str(movie[3]) + "," + d_name_str + "\n")

def music():

    print("Creating dataset directory")
    if not os.path.exists("data/music"):
        os.mkdir("data/music")

    print("Opening .csv file")
    m = pd.read_csv("data/music/tracks.csv", engine = "pyarrow")

    id = m.id
    name = m.name
    duration = m.duration_ms
    explicit = m.explicit
    artists = m.artists
    date = m.release_date
    tempo = m.tempo

    print("Creating user set")
    user_set = []

    for i in range(len(id)):
        user_set.append(i)
    random.shuffle(user_set)

    with open("data/music/user_set.txt", "w") as file:
        for user in user_set:
            file.write(str(user))
            file.write("\n")

    print("Creating database")
    track_info = set()

    for i in range(len(id)):
        artist_list = tuple(ast.literal_eval(artists[i]))
        track_info.add((name[i], round(duration[i]/1000), explicit[i], artist_list, date[i], tempo[i]))

    with open("data/music/database.csv", "w") as database:
        database.write("id,name,duration_min,explicit,artists,date,tempo\n")

        counter = 1
        for track in track_info:
            artist = list(track[3])

            artist_str = "["
            for a in artist:
                if "," in a:
                    pass
                else:
                    artist_str += a
                    artist_str += ";"

            artist_str = artist_str[:-1]
            artist_str += "]"

            name = track[0]
            if "," in name:
                pass
            else:
                database.write(str(counter) + "," + track[0] + "," + str(track[1]) + "," + str(track[2]) + "," + artist_str + "," + track[4] + "," + str(track[5]) + "\n")
                counter += 1

def main(args):
    d = args.d

    if type(d) != type(""):
        raise TypeError("The argument --d is not a string")
    elif d not in datasets:
        print("Select one dataset among {}".format(datasets))
        exit(1)

    if d == 'movies':
        movies()
    elif d == 'music':
        music()

    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type = str, required = True)

    main(args = parser.parse_args())