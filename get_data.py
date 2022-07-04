import numpy as np
import pandas as pd

def get_data(path):
    try:
        df_raw = pd.read_csv(path)
        return df_raw
    except:
        raise

def preprocess_data(raw_train, raw_test):
    df_train_x = raw_train.copy()
    df_test_x = raw_test.copy()
    mu = df_train_x.mean(axis = 0)
    std = df_train_x.std(axis = 0)
    df_train_x = (df_train_x - mu) / std
    df_test_x = (df_test_x - mu) / std
    return df_train_x, df_test_x


def separate_target(df, index, shuffle, split_ratio):
    
    data = df.to_numpy().astype(np.float32)
    print(f"Shape of data is {data.shape}\nLenght of data is {len(data)}")
    if shuffle:
        np.random.seed(100)
        np.random.shuffle(data)
        
    target = data[:,index]
    train_data = np.delete(data, index, 1)

    M = round(split_ratio*len(target))
    print(f"Split will occur after index of {M}...")
    
    Y_train = target[:M]
    Y_test = target[M:]
    
    X_train = train_data[:M]
    X_test = train_data[M:]
    
    return X_train, Y_train, X_test, Y_test


def prepare_data_to_next_step(path, index, split_ratio = 0.7, shuffle = 1):
    data = get_data(path)
    raw_x_train, Y_train, raw_x_test, Y_test = separate_target(data, index, shuffle, split_ratio)
    X_train, X_test = preprocess_data(raw_x_train, raw_x_test)
    return X_train, Y_train, X_test, Y_test
