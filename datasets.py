from faker import Faker
import pandas as pd
import argparse
import random
import ast
import os

datasets = ['movies', 'music', 'soccer', 'social']
userNumber = 10000

def movies():

    print("Creating dataset directory (if does not exist)")
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
    user_set = list(set(u.user_id))
    with open("data/movies/user_set.txt", "w") as file:
        for i in range(userNumber):
            file.write("u_" + str(user_set[i]))
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
    
    check_csv_file("data/movies/database.csv")

def music():

    print("Creating dataset directory (if does not exist)")
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

    for i in range(userNumber):
        user_set.append(i)
    random.shuffle(user_set)

    with open("data/music/user_set.txt", "w") as file:
        for user in user_set:
            file.write("u_" + str(user))
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
    
    check_csv_file("data/music/database.csv")

def soccer():

    print("Creating dataset directory (if does not exist)")
    if not os.path.exists("data/soccer"):
        os.mkdir("data/soccer")

    print("Opening .csv file")
    print("Creating user set")
    user_set = []

    for i in range(userNumber):
        user_set.append("u_" + str(i))

    with open("data/soccer/user_set.txt", "w") as file:
        for user in user_set:
            file.write(str(user))
            file.write("\n")

    check_csv_file("data/soccer/database.csv")

def social_media():

    print("Creating dataset directory (if does not exist)")
    if not os.path.exists("data/social"):
        os.mkdir("data/social")

    #columns = ["id", "user_name", "email" ,"age", "gender", "location", "n_followers", "n_following", "n_post", "language", "occupation"]

    print("Creating user set")
    with open("data/social/user_set.txt", "w") as file:
        for i in range(userNumber):
            file.write("u_" + str(i))
            file.write("\n")

    fake = Faker()

    print("Generating data")
    profile_info = set()

    for i in range(100000):
        id = i
        gender = random.choices(population = ["M", "F", "O"], weights = [0.45, 0.45, 0.10])[0]

        first_name = fake.first_name_male() if gender == "M" else fake.first_name_female() if gender == "F" else fake.first_name_nonbinary()
        last_name = fake.last_name_male() if gender == "M" else fake.last_name_female() if gender == "F" else fake.last_name_nonbinary()
        user_name = first_name + " " + last_name

        email = first_name + "." + last_name + "@" + fake.domain_name()
        age = random.choice(range(18, 65))
        location = fake.city()
        n_follower = max(round(random.normalvariate(1000, 3000)), 0)
        n_following = max(round(random.normalvariate(2000, 3000)), 0)
        n_post = random.randint(0,1000)

        language = fake.language_name()
        while "," in language:
            language = fake.language_name()
        
        occupation = fake.job()
        while "," in occupation:
            occupation = fake.job()

        profile_info.add((id, user_name, email, age, gender, location, n_follower, n_following, n_post, language, occupation))

    print("Creating database")
    with open("data/social/database.csv", "w") as database:
        database.write("id,user_name,email,age,gender,location,n_followers,n_following,n_post,language,occupation\n")

        for profile in profile_info:
            database.write(str(profile[0]) + "," + profile[1] + "," + profile[2] + "," + str(profile[3]) + "," + profile[4] + "," + profile[5] + "," + str(profile[6]) + "," + str(profile[7]) + "," + str(profile[8]) + "," + profile[9] + "," + profile[10] + "\n")

    check_csv_file("data/social/database.csv")

def check_csv_file(path):
    try:
        pd.read_csv(path, engine = "pyarrow")
    except:
        print("Malformed csv file")
        exit(1)

    print("Csv file OK")

def main(args):
    d = args.d
    global userNumber
    userNumber = args.u

    if type(d) != type(""):
        raise TypeError("The argument --d is not a string")
    elif d not in datasets:
        print("Select one dataset among {}".format(datasets))
        exit(1)

    if d == "movies":
        movies()
    elif d == "music":
        music()
    elif d == "soccer":
        soccer()
    elif d == "social":
        social_media()

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)
    parser.add_argument("--u", type = int, required = False)

    main(args = parser.parse_args())