# -*- coding: utf-8 -*-
"""
File:   hw0.py
Author: Mohit Patil
Date:   09/11/19
Desc:   ECE 
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
plt.close('all') #close any open plots

"""
===============================================================================
===============================================================================
============================ Question 1 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Function definitions ========================== """


def plotData(x1,t1,x2=None,t2=None,x3=None,t3=None,legend=[], outFile = None, title = None, xRange = None, yRange = None, showImage = True, xlabel = None,
             ylabel = None):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function
       edited the function- now takes in different labes, title, option to save the image'''
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    if(x2 is not None):
        p2 = plt.plot(x2, t2, 'g') #plot true value
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r') #plot training data

    #add title, legend and axes labels
    if ylabel:
        plt.ylabel(ylabel) #label x and y axes
    if xlabel:
        plt.xlabel(xlabel)
    
    if(x2 is None):
        plt.legend((p1[0]),legend)
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0]),legend)

    if xRange is not None:
            plt.xlim(xRange)

    if yRange is not None:
            plt.ylim(yRange)

    if title:
        plt.title(title)

    if outFile:
        plt.savefig(outFile)

    if showImage:
        plt.show()

    plt.close()
      
def fitdataLS(x,t,M):
    '''fitdataLS(x,t,M): Fit a polynomial of order M to the data (x,t) using LS'''
    X = np.array([x ** m for m in range(M + 1)]).T
    w = np.linalg.inv(X.T @ X) @ X.T @ t
    absolute_error = sum(abs(t - X @ w).T)
    # print(absolute_error)
    return w

def fitdataIRLS(x,t,M,k, tolerance_limit = 0.001):
    '''fitdataIRLS(x,t,M,k): Fit a polynomial of order M to the data (x,t) using IRLS'''

    w0 = fitdataLS(x,t,M)  #getting initial LS weights
    X = np.array([x ** m for m in range(M + 1)]).T
    (n,p) = np.shape(X)
    iop = np.zeros(n)
    # print(iop.shape)
    B = np.diag(iop) #initialize diagonal matrix B with ls weights

    # print(np.shape(t))
    # print(np.shape(X))
    # print(np.shape(w0))

    error_vec = abs(t - X @ w0).T #first error vector using LS weights

    for i in range(len(error_vec)):
        if error_vec[i] <= k:
            B[i, i] = 1
        elif error_vec[i] > k:
            # print(error_vec[i])
            B[i, i] = k / error_vec[i]
    W = np.linalg.inv(X.T @ B @ X) @ X.T @ B @ t #calculating B for the first time accordingly and getting W

    counter = 0
    while True:
        counter += 1
        B_old = B.copy() #save current B as B_old for comparison
        W_old = W
        err_vec = abs(t - X @ W).T  # get errors using current B

        '''the next loop is where B is updated in each iteration'''
        for i in range(len(err_vec)):
            if err_vec[i] <= k:
                B[i,i] = 1
            else:
                B[i,i] = k / err_vec[i]
        # print(B)

        W = np.linalg.inv(X.T @ B @ X) @ X.T @ B @ t

        B_diff = B - B_old

        tolerance_B = np.sum(abs(B_diff))
        # print("Tolerance_B = {}" .format(tolerance_B))

        tolerance_W = np.sum(abs(W - W_old))
        # print("Tolerance_W = %s".format(tolerance_W))

        if (tolerance_W < tolerance_limit):
            break

        if (tolerance_B < tolerance_limit):
            break

    return W

def optimize_k_and_m(kRange, MRange, k_step):
    'for different values of M and K, runs and minimized absolute error'
    'nested loops over K and M create multiple permutations of k,m and extract absolute error values for each of them' \
    'the minimum of such matrix is the optimized combination of k and M which is then used for subsequent analysis'

    error_stored_LS_train = []
    error_stored_IRLS_train = []
    error_stored_LS_test = []
    error_stored_IRLS_test = []

    data_uniform_train = np.load('TrainData.npy')
    x_train = data_uniform_train[:, 0]
    t_train = data_uniform_train[:, 1]

    data_uniform_test = np.load('TestData.npy')
    x_test = data_uniform_test[:, 0]
    t_test = data_uniform_test[:, 1]

    for k in np.arange(kRange[0], kRange[1], k_step):
        LS_train_list = []
        IRLS_train_list = []
        LS_test_list = []
        IRLS_test_list = []

        for M in np.arange(MRange[0], MRange[1], 1):
            wLS = fitdataLS(x_train, t_train, M)
            wIRLS = fitdataIRLS(x_train, t_train, M, k, tolerance_limit=0.001)

            X_train = np.array([x_train ** m for m in range(M + 1)]).T
            t_train_esty_LS = X_train @ wLS
            error_vec_LS_train = np.sum(abs(t_train - t_train_esty_LS).T)
            t_train_esty_IRLS = X_train @ wIRLS
            error_vec_IRLS_train = np.sum(abs(t_train - t_train_esty_IRLS).T)

            x_train_plot = np.linspace(-4.5, 4.5, 100)
            X_train_plot = np.array([x_train_plot ** m for m in range(M + 1)]).T
            t_train_plot_LS = X_train_plot @ wLS
            t_train_plot_IRLS = X_train_plot @ wIRLS

            X_test = np.array([x_test ** m for m in range(M + 1)]).T
            t_test_esty_LS = X_test @ wLS
            error_vec_LS_test = np.sum(abs(t_test - t_test_esty_LS).T)
            # print(error_vec_LS)
            t_test_esty_IRLS = X_test @ wIRLS
            error_vec_IRLS_test = np.sum(abs(t_test - t_test_esty_IRLS).T)

            LS_train_list.append(error_vec_LS_train)
            IRLS_train_list.append(error_vec_IRLS_train)
            LS_test_list.append(error_vec_LS_test)
            IRLS_test_list.append(error_vec_IRLS_test)

            # plotData(x_train, t_train, x_train_plot, t_train_plot_LS, x_train_plot ,t_train_plot_IRLS,
            #          legend=['training data', 'LS Estimation', 'IRLS Estimation'], title='M={},k={}'.format(M,k),
            #          outFile=r'D:\UF\ECE\Fundamentals of ML\Github\assignment-01-Mohit-U-Patil\plots\M={},k={}.png'.format(M,k), showImage= None, yRange=(-2, 2))

            # plotData(x_test, t_test, x_train_plot, t_train_plot_LS, x_train_plot ,t_train_plot_IRLS,
            #          legend=['training data', 'LS Estimation', 'IRLS Estimation'], title='M={},k={}'.format(M,k), xlabel='x', ylabel='t',
            #          outFile=None, showImage= True, yRange=(-2, 2)) #r'D:\UF\ECE\Fundamentals of ML\Github\assignment-01-Mohit-U-Patil\plots\M={},k={}.png'.format(M,k), put this as outfile to save the .png

        error_stored_LS_train.append(LS_train_list)
        error_stored_IRLS_train.append(IRLS_train_list)
        error_stored_LS_test.append(LS_test_list)
        error_stored_IRLS_test.append(IRLS_test_list)

    error_stored_LS_train = np.asmatrix(np.array(error_stored_LS_train))
    error_stored_IRLS_train = np.asmatrix(np.array(error_stored_IRLS_train))
    error_stored_LS_test = np.asmatrix(np.array(error_stored_LS_test))
    error_stored_IRLS_test = np.asmatrix(np.array(error_stored_IRLS_test))

    # print(error_stored_LS_train)
    # print(error_stored_IRLS_train)
    # print(error_stored_LS_test)
    # print(error_stored_IRLS_test)

    ind = np.unravel_index(np.argmin(error_stored_IRLS_test, axis=None), error_stored_IRLS_test.shape)
    print('minimum error is {}'.format(np.matrix.min(error_stored_IRLS_test)))
    print('optimal k= {} at row = {}'.format(np.arange(kRange[0], kRange[1], k_step)[ind[0]],ind[0]))
    print('optimal M= {} at column = {}'.format(np.arange(MRange[0], MRange[1], 1)[ind[1]],ind[1]))
    print(error_stored_IRLS_test[28:35,0:5])

def M_comparison_plots(MRange,k):
    '''for optimized K value'''

    data_uniform_train = np.load('TrainData.npy')
    x_train = data_uniform_train[:, 0]
    t_train = data_uniform_train[:, 1]

    data_uniform_test = np.load('TestData.npy')
    x_test = data_uniform_test[:, 0]
    t_test = data_uniform_test[:, 1]

    LS_error_list = []
    IRLS_error_list = []

    for M in np.arange(MRange[0], MRange[1], 1):
        wLS = fitdataLS(x_train, t_train, M)
        wIRLS = fitdataIRLS(x_train, t_train, M, k, tolerance_limit=0.001)

        x_train_plot = np.linspace(-4.5, 4.5, 100)
        X_train_plot = np.array([x_train_plot ** m for m in range(M + 1)]).T
        t_train_plot_LS = X_train_plot @ wLS
        t_train_plot_IRLS = X_train_plot @ wIRLS

        X_test = np.array([x_test ** m for m in range(M + 1)]).T
        t_test_esty_LS = X_test @ wLS
        error_LS_test = np.sum(abs(t_test - t_test_esty_LS).T)
        # print(error_vec_LS)
        t_test_esty_IRLS = X_test @ wIRLS
        error_IRLS_test = np.sum(abs(t_test - t_test_esty_IRLS).T)

        LS_error_list.append(error_LS_test)
        IRLS_error_list.append(error_IRLS_test)

    plotData(MRange[0],LS_error_list[0], np.arange(MRange[0], MRange[1], 1), LS_error_list, np.arange(MRange[0], MRange[1], 1),
             IRLS_error_list,
             legend=['.', 'LS: M vs error', 'IRLS: M vs error'], title='Absolute Error vs M plot for constant K', xlabel='M values', ylabel='Absolute errors',
             outFile=r'D:\UF\ECE\Fundamentals of ML\Github\assignment-01-Mohit-U-Patil\plots\0M plot for constant K.png', showImage=True)

def k_comparison_plots(kRange, M):
    '''for optimized M value'''

    data_uniform_train = np.load('TrainData.npy')
    x_train = data_uniform_train[:, 0]
    t_train = data_uniform_train[:, 1]

    data_uniform_test = np.load('TestData.npy')
    x_test = data_uniform_test[:, 0]
    t_test = data_uniform_test[:, 1]

    LS_error_list = []
    IRLS_error_list = []

    for k in np.arange(kRange[0], kRange[1], 0.01):
        wLS = fitdataLS(x_train, t_train, M)
        wIRLS = fitdataIRLS(x_train, t_train, M, k, tolerance_limit=0.001)

        x_train_plot = np.linspace(-4.5, 4.5, 100)
        X_train_plot = np.array([x_train_plot ** m for m in range(M + 1)]).T
        t_train_plot_LS = X_train_plot @ wLS
        t_train_plot_IRLS = X_train_plot @ wIRLS

        X_test = np.array([x_test ** m for m in range(M + 1)]).T
        t_test_esty_LS = X_test @ wLS
        error_LS_test = np.sum(abs(t_test - t_test_esty_LS).T)
        # print(error_vec_LS)
        t_test_esty_IRLS = X_test @ wIRLS
        error_IRLS_test = np.sum(abs(t_test - t_test_esty_IRLS).T)

        LS_error_list.append(error_LS_test)
        IRLS_error_list.append(error_IRLS_test)

    plotData(kRange[0],LS_error_list[0], np.arange(kRange[0], kRange[1], 0.01), LS_error_list, np.arange(kRange[0], kRange[1], 0.01),
             IRLS_error_list,
             legend=['.', 'LS: k vs error','IRLS: k vs erro'], title='Absolute Error vs k plot for constant M', xlabel='k values', ylabel='Absolute errors',
             outFile=r'D:\UF\ECE\Fundamentals of ML\Github\assignment-01-Mohit-U-Patil\plots\0k plot for constant M.png',
             showImage=True)

def training_estimation_plots(k, M):
    '''for optimized K,M values'''

    data_uniform_train = np.load('TrainData.npy')
    x_train = data_uniform_train[:, 0]
    t_train = data_uniform_train[:, 1]

    wLS = fitdataLS(x_train, t_train, M)
    wIRLS = fitdataIRLS(x_train, t_train, M, k, tolerance_limit=0.001)

    x_train_plot = np.linspace(-4.5, 4.5, 100)
    X_train_plot = np.array([x_train_plot ** m for m in range(M + 1)]).T
    t_train_plot_LS = X_train_plot @ wLS
    t_train_plot_IRLS = X_train_plot @ wIRLS

    plotData(x_train, t_train, x_train_plot, t_train_plot_LS, x_train_plot, t_train_plot_IRLS,
                      legend=['training data', 'LS Estimation', 'IRLS Estimation'], title='Estimations for M={},k={}'.format(M,k),
                      outFile=r'D:\UF\ECE\Fundamentals of ML\Github\assignment-01-Mohit-U-Patil\plots\M={},k={}.png'.format(M,k), showImage= True, yRange=(-2, 2))


             # """ ======================  Variable Declaration ========================== """
# M =  8 #regression model order
# k = 0.1 #Huber M-estimator tuning parameter
# iterations = 1000
# tolerance_limit = 0.001
# """ =======================  Load Training Data ======================= """
# data_uniform = np.load('TrainData.npy')
# np.shape(data_uniform)
# x_train = data_uniform[:,0]
# t_train = data_uniform[:,1]
#
#
# """ ========================  Train the Model ============================= """
# wLS = fitdataLS(x_train, t_train,M)
# wIRLS = fitdataIRLS(x_train, t_train, M,k, iterations, tolerance_limit)
#
# """ ======================== Load Test Data  and Test the Model =========================== """
#
# """This is where you should load the testing data set. You shoud NOT re-train the model   """
# X_train = np.array([x_train ** m for m in range(M + 1)]).T
# t_train_esty_LS = X_train @ wLS
# error_vec_LS_train = abs(t_train - t_train_esty_LS).T
# t_train_esty_IRLS = X_train @ wIRLS
# error_vec_IRLS_train = abs(t_train - t_train_esty_IRLS).T
#
# x_train_plot = np.linspace(-4.5, 4.5 , 50)
# X_train_plot = np.array([x_train_plot ** m for m in range(M + 1)]).T
# t_train_plot_LS = X_train_plot @ wLS
# t_train_plot_IRLS = X_train_plot @ wIRLS
#
#
# data_uniform_test = np.load('TestData.npy')
# x_test = data_uniform[:,0]
# t_test = data_uniform[:,1]
#
# X_test = np.array([x_test ** m for m in range(M + 1)]).T
#
# t_test_esty_LS = X_test @ wLS
# error_vec_LS_test = abs(t_test - t_test_esty_LS).T
# # print(error_vec_LS)
#
# t_test_esty_IRLS = X_test @ wIRLS
# error_vec_IRLS_test = abs(t_test - t_test_esty_IRLS).T
# # print(error_vec_IRLS)
""" ========================  Plot Results ============================== """

""" This is where you should create the plots requested """
# plotData(x_train, t_train, x_train_plot, t_train_plot_LS, x_train_plot ,t_train_plot_IRLS, legend=['training data','LS Estimation','IRLS Estimation'], title='trial',
#          outFile=r'D:\UF\ECE\Fundamentals of ML\Github\assignment-01-Mohit-U-Patil\plots\trial.png', yRange=(-2,2))

optimize_k_and_m([0.001, 0.3],[8,14],0.001)
M_comparison_plots((8,14),0.033)
k_comparison_plots((0.001,1),10)
training_estimation_plots(0.033, 10)