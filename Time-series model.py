# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:38:50 2019

@author: Ranak Roy Chowdhury
"""
import numpy as np


# read the files
def readFile():
    train_filename = "nasdaq00.txt"
    train_list = []
    with open(train_filename) as f:
        for line in f:
            train_list.append(float(line))
            
    test_filename = "nasdaq01.txt"
    test_list = []
    with open(test_filename) as f:
        for line in f:
            test_list.append(float(line))
     
    return train_list, test_list


# convert the lists into 
# train_input, train_label, test_input, test_label
def preprocess(train_list, test_list, past_days):
    train_input = []
    train_label = []
    for i in range(len(train_list) - past_days):
        train_input.append(train_list[i : i + past_days])
        train_label.append(train_list[i + past_days])
        
    test_input = []
    test_label = []
    for i in range(len(test_list) - past_days):
        test_input.append(test_list[i : i + past_days])
        test_label.append(test_list[i + past_days])
    
    # convert all lists to numpy arrays
    train_data_input = np.array(train_input)
    train_data_output = np.array(train_label)
    test_data_input = np.array(test_input)
    test_data_output = np.array(test_label)
    
    return train_data_input, train_data_output, test_data_input, test_data_output
    

# perform training to learn weights
def training(train_data_input, train_data_output, past_days):
    # perform the inverse of the input matrix, A^(-1)
    train_data_input_inv = np.linalg.inv(np.matmul(train_data_input.transpose(), train_data_input))
    
    # prepare the matrix, b
    b = []
    for i in range(past_days):
        b.append(np.dot(train_data_input[:, i], train_data_output))
    b = np.array(b)
    
    # learn the weights
    weights = np.matmul(train_data_input_inv, b.transpose())
    return weights
    

# evaluate the performance of learned model on training and test data
def evaluation(train_data_input, train_data_output, test_data_input, test_data_output, weights):
    
    train_pred = np.matmul(train_data_input, weights)
    mse_train = ((train_pred - train_data_output)**2).mean(axis=None)
    
    test_pred = np.matmul(test_data_input, weights)
    mse_test = ((test_pred - test_data_output)**2).mean(axis=None)
    
    return mse_train, mse_test
    

if __name__ == "__main__":
    train_list, test_list = readFile()
    past_days = 3
    train_data_input, train_data_output, test_data_input, test_data_output = preprocess(train_list, test_list, past_days)
    
    weights = training(train_data_input, train_data_output, past_days)
    mse_train, mse_test = evaluation(train_data_input, train_data_output, test_data_input, test_data_output, weights)    
    
    print(weights)
    print("Mean Square Error on Training data (Year 2000): ", str(mse_train))
    print("Mean Square Error on Test data (Year 2001): ", str(mse_test))