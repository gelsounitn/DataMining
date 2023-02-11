from silence_tensorflow import silence_tensorflow
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import math
import os

# disabling all messages from tensorflow package
silence_tensorflow()

def early_stopping(losses, patience = 3):
     
    if len(losses) <= patience + 1:
        return False
     
    avg_loss = np.mean(losses[-1 - patience:-1])

    if avg_loss - losses[-1] < 0.01 * avg_loss or losses[-1] < 0.001 :
        return True
     
    return False

def isNotNan(x):
    if(pd.isna(x)):
        return 0.
    else:
        return 1.

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

    # saving matrix dimensions
    n_rows = df.shape[0]
    n_columns = df.shape[1]

    # choosing u,v dimensions (latent_factors)
    # 12000 * 8001 -> 600
    latent_factors = math.ceil((n_rows*n_columns) ** (1/3)) * 2
    #print("uv dimension: " + str(latent_factors))

    # normalization of each matrix column
    # 1 = rows, 0 = columns
    print("Normalizing utility matrix")
    user_ratings_mean = df.mean(axis = 1)
    user_ratings_mean /= 2
    query_average = df.mean(axis = 0)
    query_average /= 2

    df = df.sub(user_ratings_mean, axis = 0)
    df = df.sub(query_average, axis = 1)

    # filling the nans in the means vectors with 0s
    user_ratings_mean = user_ratings_mean.fillna(0)
    query_average = query_average.fillna(0)

    # creating a dataframe with ones where there are nans 
    df_copy = df.copy()
    df_copy = df_copy.applymap(isNotNan)

    # filling the dataframe nans with 0s
    df = df.fillna(0)

    # making the dataframe a tensor
    M = tf.convert_to_tensor(df, dtype = tf.float32)

    # making tensors with infos about filled and empty values
    sparsity_mat = tf.convert_to_tensor(df_copy, dtype = tf.float32)
    masked_entries = tf.cast(tf.not_equal(sparsity_mat, 1), dtype = 'float32')

    # deleting df copy
    df_copy = None

    # prova con 5 set (2 distribuzioni normali + 1 con solo uni + 1 con solo zeri + 1 con valori presi casualmente tra -c e c che decido io (deve essere vicino a 0, per esempio -1, 1))
    # initializing U and V matrices
    U_d = tf.Variable(tf.random.normal((n_rows, latent_factors), mean = 0, stddev = 1))
    V_d = tf.Variable(tf.random.normal((latent_factors, n_columns), mean = 0, stddev = 1))

    # defining a varying learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = .4,
    decay_steps = 100.,
    decay_rate = 0.96,
    staircase = False
    )

    # choosing optimizer
    adam_opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    
    from datetime import datetime
    ep = 0
    start_time = datetime.now()
    
    losses = []
    val_losses = []
    weighted_losses = []

    # peso da assegnare ad ogni cella vuota
    weight = 0.25
    
    # get the number of filled and missing cells
    train_norm = tf.reduce_sum(sparsity_mat)
    val_norm = tf.reduce_sum(masked_entries)

    # performing optimization
    while True:
        
        with tf.GradientTape() as tape:

            #print(U_d.shape, V_d.shape)
            M_app = U_d @ V_d
            
            pred_errors_squared = tf.square(M - M_app)
            loss = tf.reduce_sum((sparsity_mat * pred_errors_squared) / train_norm)
            val_loss = tf.reduce_sum((masked_entries * pred_errors_squared) / val_norm)
            weighted_loss = loss + weight * val_loss
    
        if ep % 10 == 0:
            print(datetime.now() - start_time, loss, val_loss, weighted_loss, ep)
            losses.append(loss.numpy())
            val_losses.append(val_loss.numpy())
            weighted_losses.append(weighted_loss.numpy())
            print((U_d @ V_d)[0][3])

        if early_stopping(weighted_losses):
            break
        
        grads = tape.gradient(weighted_loss, [U_d, V_d])
        adam_opt.apply_gradients(zip(grads, [U_d, V_d]))
    
        ep += 1
    
    print('total time: ', datetime.now() - start_time)
    print('epochs: ', ep)

    final_matrix = tf.cast(U_d @ V_d, dtype = tf.int32)
    final_df = pd.DataFrame(final_matrix)
    #final_df = final_df.add(user_ratings_mean, axis = 0)
    #user_ratings_mean = tf.convert_to_tensor(user_ratings_mean, dtype = tf.int32)
    #query_average = tf.convert_to_tensor(query_average, dtype = tf.int32)
    #final_matrix += user_ratings_mean
    #final_matrix = tf.compat.v1.assign_add(final_matrix, query_average)
    #print(df.head())
    #print(final_matrix[0])

    # for column in final_df.columns:
    #     final_df[column] += query_average

    # print(final_df.index)

    #final_df = final_df.add(user_ratings_mean, axis = 1)
    #final_df = final_df.add(query_average, axis = 0)
    #final_df = final_df.add(user_ratings_mean, axis = 1)
    #final_df = final_df.add(query_average, axis = 1)
    print(final_df.head())

# learning rate (anche come decade), latent_factors, peso delle celle vuote, 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)

    main(args = parser.parse_args())