import pandas as pd
import argparse
import random

# use --d to specify path/file.csv of the database
# use --n to specify number of queries you want in return (default = 2000)

def main(args):

    q = args.q
    u = args.u
    n = 2000
 
    if type(q) != type(""):
        raise TypeError("The argument --d is not a string")

    if type(u) != type(""):
        raise TypeError("The argument --u is not a string")

    ###

    database = pd.read_csv(q, engine = "pyarrow")
    users = pd.read_csv(u, engine = "pyarrow")

    print(users)
    print("done")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type = str, required = True)     #the path/file with the queries
    parser.add_argument('--u', type = str, required = True)     #the path/file with the user list
    main(args = parser.parse_args())