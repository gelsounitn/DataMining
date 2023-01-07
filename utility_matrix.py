import argparse
import random

# use --d to specify path/file.csv of the database
# use --n to specify number of queries you want in return (default = 2000)

def main(args):

    q = args.q
    u = args.u
    mu = 500 #200
 
    if type(q) != type(""):
        raise TypeError("The argument --d is not a string")

    if type(u) != type(""):
        raise TypeError("The argument --u is not a string")

    ###
    print("Opening user set and query set")
    path_list = u.split("/")
    d = path_list[1]
    u_path = "data/" + d + "/utility_matrix.csv"

    with open(u, "r") as user_set, open(q, "r") as query_set, open(u_path, "w") as u_matrix:
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

        sigma = round(mu/50) #/10

        for user in users:
            user = user[:-1]
            s = user + ","

            randomnumber = min(round(abs(random.normalvariate(mu, sigma))), len(queries))
            
            extracted_list = []
            extracted_set = set()
            while len(extracted_set) != randomnumber:
                extracted_set.add(random.randint(0, len(queries)-1))
            
            extracted_list = list(extracted_set)
            extracted_list.sort()
            

            k = 0
            for j in range(len(query_ids)):
                if j == extracted_list[k]:
                    s += "," + str(random.randint(1, 100))

                    if k < len(extracted_list) - 1:
                        k += 1
                else:
                    s += ","
            
            u_matrix.write(s + "\n")

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type = str, required = True)     #the path/file with the queries
    parser.add_argument("--u", type = str, required = True)     #the path/file with the user list
    main(args = parser.parse_args())