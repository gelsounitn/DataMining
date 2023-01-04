import pandas as pd
import argparse
import random

# use --d to specify path/file.csv of the database
# use --n to specify number of queries you want in return (default = 2000)

def main(args):

    q = args.q
    u = args.u
    mu = 30
 
    if type(q) != type(""):
        raise TypeError("The argument --d is not a string")

    if type(u) != type(""):
        raise TypeError("The argument --u is not a string")

    ###
    print("Opening user set and query set")
    with open(u, "r") as user_set, open(q, "r") as query_set, open("utility_matrix.csv", "w") as u_matrix:
        users = user_set.readlines()
        queries = query_set.readlines()

        print("Creating utility matrix")
        query_ids = []
        for i in range(len(queries)):
            attributes = queries[i].split(",")

            if i == len(queries) - 1:
                u_matrix.write(str(attributes[0]) + "\n")
            else:
                u_matrix.write(str(attributes[0]) + ",")

            query_ids.append(attributes[0])

        sigma = round(mu/5)

        for _ in users:
            randomnumber = min(round(abs(random.normalvariate(mu, sigma))), len(queries))
            
            extracted_list = []
            for _ in range(randomnumber):
                extracted_list.append(random.randint(0, len(queries)-1))
            
            extracted_list.sort()
            
            k = 0
            for j in range(len(query_ids)):
                if j == extracted_list[k]:
                    u_matrix.write("," + str(query_ids[j]))

                    if k < len(extracted_list) - 1:
                        k += 1

                else:
                    u_matrix.write(",")
            
            u_matrix.write("\n")

    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type = str, required = True)     #the path/file with the queries
    parser.add_argument('--u', type = str, required = True)     #the path/file with the user list
    main(args = parser.parse_args())