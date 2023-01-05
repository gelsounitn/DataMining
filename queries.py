import pandas as pd
import argparse
import random

from datasets import datasets

# use --d to specify path/file.csv of the database
# use --n to specify number of queries you want in return (default = 2000)
# note that different queries might be equevalent and have attributes in different orders

def main(args):

    d = args.d
    n = 2000

    if args.n != 0:
        n = args.n
 
    if type(d) != type(""):
        raise TypeError("The argument --d is not a string")
    elif d not in datasets:
        print("Select one dataset among {}".format(datasets))
        exit(1)

    ###
    print("Creating query set")
    q_path = "data/" + d + "/query_set.csv"
    with open(q_path, "w") as file:

        database_path = "data/" + d + "/database.csv"
        database = pd.read_csv(database_path, engine = "pyarrow")

        attributes = list(database)
        print("- Database rows: " + str(len(database.index)))
        print("- Attributes: " + str(attributes))

        extracted_list = []
        i = 0
        for i in range(n):

            r = random.randint(1, len(database.index)-1)
            extracted_list.append(r)
            na = random.randint(1, len(attributes)-1)

            attribute_set = set()
            for a in range(na):
                attribute_set.add(attributes[random.randint(1, len(attributes)-1)])
            
            query = "Q" + str(i)
            for a in attribute_set:
                query += "," + a + "=" + str(database.at[r, a])
            
            query += "\n"
            file.write(query)

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)     #remember to write the name as argument
    parser.add_argument("--n", type = int, required = False)    #the number of queries you want to generate

    main(args = parser.parse_args())