import pandas as pd
import numpy as np
import argparse
import os
import tensorflow as tf
import math

np.set_printoptions(threshold = np.inf)

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

    # choosing u,v dimensions
    # 12000 * 8001 -> 600
    uv_dimension = math.ceil((n_rows*n_columns) ** (1/3)) * 2
    print("uv dimension: " + str(uv_dimension))

    # normalization of each matrix column
    print("Normalizing utility matrix")
    user_ratings_mean = df.mean(axis = 0)
    df = df.sub(user_ratings_mean, axis = 1)

    # creatin a dataframe with ones where there are nans 
    df_copy = df.copy()
    df_copy = df_copy.applymap(isNotNan)

    # filling df nans with zeros (mean) and making it a tensor
    df = df.fillna(0)
    M = tf.convert_to_tensor(df, dtype=tf.float32)

    # making tensors with infos about filled and empty values
    sparsity_mat = tf.convert_to_tensor(df_copy, dtype=tf.float32)
    masked_entries = tf.cast(tf.not_equal(sparsity_mat, 1), dtype = 'float32')

    # deleting df copy
    df_copy = None

    # initializing U and V matrices
    U_d = tf.Variable(tf.random.normal((n_rows, uv_dimension), mean=0, stddev=1))
    V_d = tf.Variable(tf.random.normal((uv_dimension, n_columns), mean=0, stddev=1))

    # defining a varying learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=.4,
    decay_steps=100.,
    decay_rate=0.96,
    staircase=False
    )

    # choosing optimizer
    adam_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    from datetime import datetime
    ep = 0
    start_time = datetime.now()
    
    losses = []
    val_losses = []
    weighted_losses = []
    weight = 0.25
    
    # get the number of filled and missing cells
    train_norm = tf.reduce_sum(sparsity_mat)
    val_norm = tf.reduce_sum(masked_entries)

    # performing optimization
    while True:
        
        with tf.GradientTape() as tape:

            M_app = U_d @ V_d
            
            pred_errors_squared = tf.square(M - M_app)
            loss = tf.reduce_sum((sparsity_mat * pred_errors_squared)/train_norm)
            val_loss = tf.reduce_sum((masked_entries * pred_errors_squared)/val_norm)
            weighted_loss = loss + weight * val_loss
    
        if ep%10 == 0:
            print(datetime.now() - start_time, loss, val_loss, weighted_loss, ep)
            losses.append(loss.numpy())
            val_losses.append(val_loss.numpy())
            weighted_losses.append(weighted_loss.numpy())
            print((U_d @ V_d)[0][3])

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