import pandas as pd
import numpy as np
import argparse
import os
import tensorflow as tf
import math

np.set_printoptions(threshold = np.inf)


# es query: db.query('year == 2004 and popularity == 0')
def rowToQuery(row):
    row = row.replace('\n', '')
    row = row.replace('=', '=="')
    row = row.replace(',', '",')
    row = row.split(',')
    query = str(row[1])
    row = row[2:]
    for u in row:
        query += ' and '
        query += str(u)
    query += '"'
    return query



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
    print("Reading database")
    db = pd.read_csv("database.csv", dtype="string", index_col="id")
    print("Reading query list")
    ql = []
    with open("query_set.csv", "r") as queryFile:
        ql = queryFile.readlines()

    #print(db.head())

    signaturesTf = tf.zeros([db.shape[0],len(ql)], tf.dtypes.int8)
    signatures = pd.DataFrame(signaturesTf)
    signatures["id"] = db.index
    signatures.set_index("id")



    # for each query in query_list
    for i in range(len(ql)):
        query = rowToQuery(ql[i])
        elements = db.query(query)
        
        # foreach element returned by the query
        for index, row in elements.iterrows():
            id = row.name

            # put ones where elements are found in the column
            #db.loc[id, i] = 1

        if i%10 == 0:
            print(i)
            

    


    exit(1)

    n_rows = df.shape[0]
    n_columns = df.shape[1]

    uv_dimension = math.ceil((n_rows*n_columns) ** (1/3)) * 2
    print("uv dimension: " + str(uv_dimension))

    # normalization of each matrix column
    print("Normalizing utility matrix")
    user_ratings_mean = df.mean(axis = 0)
    df = df.sub(user_ratings_mean, axis = 1)
    df_copy = df.copy()

    df = df.fillna(0)

    M = tf.convert_to_tensor(df, dtype=tf.float32)

    #harm = np.array([1/(i + 1) for i in range(500) ])
    #plt.plot(harm)
    #plt.show()

    #print(M[2])
    #print(type(M[2][0]))
    #print(tf.not_equal(M, np.nan)[2])

    df_copy = df_copy.applymap(isNotNan)
    #print(df_copy.head())

    sparsity_mat = tf.convert_to_tensor(df_copy, dtype=tf.float32)
    masked_entries = tf.cast(tf.not_equal(sparsity_mat, 1), dtype = 'float32')

    df_copy = None

    U_d = tf.Variable(tf.random.normal((n_rows, uv_dimension), mean=0, stddev=1))
    V_d = tf.Variable(tf.random.normal((uv_dimension, n_columns), mean=0, stddev=1))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=50.,
    decay_steps=100.,
    decay_rate=0.96,
    staircase=False
    )

    adam_opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    
    from datetime import datetime
    ep = 0
    start_time = datetime.now()
    
    losses = []
    val_losses = []
    weighted_losses = []
    weight = 0.25
    
    train_norm = tf.reduce_sum(sparsity_mat)
    #print("train_norm ", train_norm)
    val_norm = tf.reduce_sum(masked_entries)
    #print("val_norm", val_norm)

    while True:
        
        with tf.GradientTape() as tape:

            M_app = U_d @ V_d
            
            pred_errors_squared = tf.square(M - M_app)
            #print("pred_errors_squared", pred_errors_squared[0])
            loss = tf.reduce_sum((sparsity_mat * pred_errors_squared)/train_norm)
            
            val_loss = tf.reduce_sum((masked_entries * pred_errors_squared)/val_norm)

            weighted_loss = loss + weight * val_loss
    
        if ep%10 == 0:
            print(datetime.now() - start_time, loss, val_loss, weighted_loss, ep)
            losses.append(loss.numpy())
            val_losses.append(val_loss.numpy())
            weighted_losses.append(weighted_loss.numpy())
            print((U_d @ V_d)[0][3])
            #print(losses)
            #print(val_losses)

        if early_stopping(weighted_losses): #val_losses
            break
        
        grads = tape.gradient(weighted_loss, [U_d, V_d])
        adam_opt.apply_gradients(zip(grads, [U_d, V_d]))
    
        ep += 1
    
    print('total time: ', datetime.now() - start_time)
    print('epochs: ', ep)

    final_matrix = tf.cast(U_d @ V_d, dtype=tf.int32)
    print(df.head())
    print(final_matrix[0])


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)

    main(args = parser.parse_args())