import pandas as pd
import argparse
import random

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

    ###

    with open("query_set.txt", "w") as file:

        database = pd.read_csv(d, engine = "pyarrow")

        attributes = list(database)
        print("Database rows: " + str(len(database.index)))
        print("Attributes: " + str(attributes))
        #print( "n = " + str(n))

        extracted_set = set()
        i = 0
        extracted_set_size = 1
        for i in range(n):
            #print("i = " + str(i))
            while len(extracted_set) < extracted_set_size:
                r = random.randint(1, len(database.index))
                #print("r = " + str(r))
                if r not in extracted_set:
                    extracted_set.add(r)
                    na = random.randint(1, len(attributes)-1)
                    attribute_set = set()
                    j = 0
                    for a in range(na):
                        attribute_set.add(attributes[random.randint(1, len(attributes)-1)])
                    
                    query = "Q" + str(i)
                    for a in attribute_set:
                        query += "," + a + "=" + str(database.at[r, a])
                    
                    query += "\n"
                    file.write(query)

            extracted_set_size = extracted_set_size + 1 

        print("Done")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type = str, required = True)     #remember to write a path/file as argument
    parser.add_argument('--n', type = int, required = False)    #the number of queries you want to generate

    main(args = parser.parse_args())