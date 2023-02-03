import scipy
from scipy import linalg
from scipy import sparse
import dask.array as da
import pandas as pd
import numpy as np
import argparse
import os
import scipy.sparse
import matplotlib.pyplot as plt
import tensorflow as tf

np.set_printoptions(threshold = np.inf)

def early_stopping(losses, patience = 5):
     
    if len(losses) <= patience + 1:
        return False
     
    avg_loss = np.mean(losses[-1 - patience:-1])
     
    if avg_loss - losses[-1] < 0.01*avg_loss:
        return True
     
    return False

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

    row = df.shape[0]
    column = df.shape[1]

    # normalization of each matrix column
    print("Normalizing utility matrix")
    user_ratings_mean = df.mean(axis = 0)
    df = df.sub(user_ratings_mean, axis = 1)
    df = df.fillna(0)
    #user_ratings_mean = user_ratings_mean.fillna(0)

    #matplotlib inline
 
    #U = tf.random.normal((500,500), mean=0, stddev=16, dtype = 'float32')
    #V = tf.random.normal((500,500), mean=0, stddev=16, dtype = 'float32')
    
    M = tf.convert_to_tensor(df, dtype=tf.float32)

    #harm = np.array([1/(i + 1) for i in range(500) ])
    #plt.plot(harm)
    #plt.show()

    sparsity_mat = tf.cast(tf.not_equal(M, np.nan), dtype = 'float32')
    masked_entries = tf.cast(tf.not_equal(sparsity_mat, 1), dtype = 'float32')

    U_d = tf.Variable(tf.random.normal((12000, 50), mean=0, stddev=16))
    V_d = tf.Variable(tf.random.normal((50, 8001), mean=0, stddev=16))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=10.,
    decay_steps=100.,
    decay_rate=0.9,
    staircase=False
    )

    adam_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    from datetime import datetime
    ep = 0
    start_time = datetime.now()
    
    losses = []
    val_losses = []
    
    train_norm = tf.reduce_sum(sparsity_mat)
    val_norm = tf.reduce_sum(masked_entries)
    
    while True:
        
        with tf.GradientTape() as tape:
            M_app = U_d @ V_d
            
            pred_errors_squared = tf.square(M - M_app)
            loss = tf.reduce_sum((sparsity_mat * pred_errors_squared)/train_norm)
            
        val_loss = tf.reduce_sum((masked_entries * pred_errors_squared)/val_norm)
    
        if ep%100 == 0:
            print(datetime.now() - start_time, loss, val_loss, ep)
            losses.append(loss.numpy())
            val_losses.append(val_loss.numpy())
        if early_stopping(val_losses):
            break
        
        grads = tape.gradient(loss, [U_d, V_d])
        adam_opt.apply_gradients(zip(grads, [U_d, V_d]))
    
        ep += 1
    
    print('total time: ', datetime.now() - start_time)
    print('epochs: ', ep)
    print('learning rate now: ', lr_schedule)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = str, required = True)

    main(args = parser.parse_args())