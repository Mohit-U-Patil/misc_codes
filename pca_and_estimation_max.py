# -*- coding: utf-8 -*-
"""
File:   hw03B and C.py
Author: Mohit Patil
"""

"""
====================================================
================ Import Packages ===================
====================================================
"""
import sys

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import skimage.filters as filt
from sklearn.preprocessing import StandardScaler

"""
====================================================
=========  Part B =============
====================================================
"""
"""PCA dimentionality reduction"""

def calculator():
    print(" \n Q1.2 calulations to support hand calculations. Answers differ from manualy calculated answer after covariance matrix.")
    x = np.array([2, 3, 3, 4, 5, 7])
    y = np.array([2, 4,  5, 5, 6, 8])

    data_raw = np.vstack((x, y))
    print("data without standardization \n{}".format(data_raw))
    cov_mat_raw = np.cov(data_raw)
    print("Covariance matrix for raw data \n{}".format(cov_mat_raw))
    eigen_vals_raw, eigen_vecs_raw = np.linalg.eig(cov_mat_raw)
    print(r'Eigenvalues %s' % eigen_vals_raw)
    print('Eigenvectors \n %s' % eigen_vecs_raw)

    #perform dimensionality reduction
    eigen_pairs_raw = [(np.abs(eigen_vals_raw[i]), eigen_vecs_raw[:,i]) for i in range(len(eigen_vals_raw))]
    eigen_pairs_raw.sort(reverse=True)
    w_raw = np.hstack((eigen_pairs_raw[0][1][:, np.newaxis]))
    # print(r'Matrix W:', w_raw)
    # X_pca_raw = (data_raw.T).dot(w_raw)
    # print("dimension reduced data: {}".format(X_pca_raw))


    mu_x = 4
    mu_y = 5
    std_dev_x = np.sqrt(sum((x-mu_x)**2)/6)
    print('Standard dev x', std_dev_x)
    x_standardized = (x-mu_x)/std_dev_x
    std_dev_y = np.sqrt(sum((y - mu_y) ** 2) / 6)
    y_standardized = (y-mu_y) / std_dev_y
    print('Standard dev y', std_dev_y)
    data_standardized = np.vstack((x_standardized,y_standardized))
    print("data after standardization \n {}".format(data_standardized))
    cov_mat = np.cov(data_standardized)
    print("Covariance matrix for standardized data \n {}".format(cov_mat))
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print(r'Eigenvalues %s' % eigen_vals)
    print('Eigenvectors \n %s' % eigen_vecs)

    #perform dimensionality reduction
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    # print(eigen_pairs)
    eigen_pairs.sort(reverse=True)
    # w = np.hstack((eigen_pairs[0][1][:, np.newaxis]))
    # print('Matrix W:', w)
    # X_pca = (data_standardized.T).dot(w)
    # print("dimension reduced data: {}".format(X_pca))

def PCA_reduction():
    print("\n Q1.4 calulations PCA dimensionality reduction ")
    X = [[2,2], [3,4], [3,5], [4,5], [5,6], [7,8]]
    X = np.array(X)
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    print("data standardized \n {}".format(np.round(X_std,3)))
    # print(X_std)
    # print(X.shape)

    #compute covariance, eigenvals and eigenvecs
    cov_mat = np.cov(X_std.T)
    print("Covariance matrix for standardized data \n {}".format(cov_mat))

    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print('\nEigenvalues \n%s' % eigen_vals)
    print('\nEigenvectors \n%s' % eigen_vecs)


    #perform dimensionality reduction
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(reverse=True)

    w = np.hstack((eigen_pairs[0][1][:, np.newaxis]))
    print('Matrix W:\n', w)

    X_pca = X_std.dot(w)
    print("dimension reduced data: {}".format(X_pca))

    # cov_mat = np.cov(X_pca.T)
    # print(cov_mat)
    # print(X_pca.shape)
    colors = ['r','b','g']
    markers = ['s', 'x', 'o']
    plt.scatter(X_pca,np.ones(np.shape(X_pca)))
    plt.xlabel('PC 1')
    plt.show()
    # print(X_pca)

"""Estimation Maximization"""

def EM(iterations, initialization):
    print(" \n Q2 EM.")

    y = np.array([1,1,0,1,0,0,1,0,1,1])
    mu = np.zeros(10)
    MaximumNumberOfIterations = iterations

    # Initialize Parameters
    (pi,p,q) = initialization

    for i in range(len(y)):
        mu[i] = pi * p**y[i] * (1-p)**(1-y[i]) / (pi * p**y[i] * (1-p)**(1-y[i]) + (1-pi) * q**y[i] * (1-q)**(1-y[i]))

    print("\ntheta(0) is {}".format((round(pi,2),round(p,2),round(q,2))))
    print("mu(1) is {}".format(np.round(mu,2)))
    Diff = np.inf
    NumberIterations = 1

    while NumberIterations <= MaximumNumberOfIterations:
        # Update Means, Sigs, Ps

        pi = 0.1 * sum(mu)
        p = sum(mu * y)/ sum(mu)
        q = sum((1-mu) * y)/ sum(1-mu)

        for i in range(len(y)):
            mu[i] = pi * p ** y[i] * (1 - p) ** (1 - y[i]) / (
                        pi * p ** y[i] * (1 - p) ** (1 - y[i]) + (1 - pi) * q ** y[i] * (1 - q) ** (1 - y[i]))

        print("\ntheta({}) is {}".format(NumberIterations, (round(pi,2),round(p,2),round(q,2))))
        print("mu({}) is {}".format(NumberIterations+1, np.round(mu,2)))
        NumberIterations = NumberIterations + 1

"""
====================================================
================ Part C ==================
====================================================
"""

def process_image(in_fname,out_fname,debug=False):

    # load image
    x_in = np.array(Image.open(in_fname))

    # convert to grayscale
    x_gray = 1.0-rgb2gray(x_in)

    if debug:
        plt.figure(1)
        plt.imshow(x_gray)
        plt.title('original grayscale image')
        plt.show()

    # threshold to convert to binary
    thresh = filt.threshold_minimum(x_gray)
    fg = x_gray > thresh

    if debug:
        plt.figure(2)
        plt.imshow(fg)
        plt.title('binarized image')
        plt.show()

    # find bounds
    nz_r,nz_c = fg.nonzero()
    n_r,n_c = fg.shape
    l,r = max(0,min(nz_c)-1),min(n_c-1,max(nz_c)+1)+1
    t,b = max(0,min(nz_r)-1),min(n_r-1,max(nz_r)+1)+1

    # extract window
    win = fg[t:b,l:r]

    if debug:
        plt.figure(3)
        plt.imshow(win)
        plt.title('windowed image')
        plt.show()

    # resize so largest dim is 48 pixels 
    max_dim = max(win.shape)
    new_r = int(round(win.shape[0]/max_dim*48))
    new_c = int(round(win.shape[1]/max_dim*48))

    win_img = Image.fromarray(win.astype(np.uint8)*255)
    resize_img = win_img.resize((new_c,new_r))
    resize_win = np.array(resize_img).astype(bool)

    # embed into output array with 1 pixel border
    out_win = np.zeros((resize_win.shape[0]+2,resize_win.shape[1]+2),dtype=bool)
    out_win[1:-1,1:-1] = resize_win

    if debug:
        plt.figure(4)
        plt.imshow(out_win,cmap='Greys')
        plt.title('resized windowed image')
        plt.show()

    #save out result as numpy array
    np.save(out_fname,out_win)
    return out_win

def mass_process():
    """the function needs filepaths of local directory where all raw images are stored and the path to the directory where all new binary images are supposed to be saved"""
    binary_image_collection = []
    label_collection = []
    for i in np.arange(1,81,1):
        if i<10:
            in_fname =  r'D:\UF\ECE\Fundamentals of ML\Github\assignment-03-Mohit-U-Patil\alphabet_images_scanned\image_part_00{}.jpg'.format(i)
            out_fname = r'D:\UF\ECE\Fundamentals of ML\Github\assignment-03-Mohit-U-Patil\binary_images\image_part_00{}_binary.npy'.format(i)
            out_win = process_image(in_fname,out_fname, debug=None)
        elif i >= 10:
            in_fname = r'D:\UF\ECE\Fundamentals of ML\Github\assignment-03-Mohit-U-Patil\alphabet_images_scanned\image_part_0{}.jpg'.format(i)
            out_fname = r'D:\UF\ECE\Fundamentals of ML\Github\assignment-03-Mohit-U-Patil\binary_images\image_part_0{}_binary.npy'.format(i)
            out_win = process_image(in_fname, out_fname)
            # print(type(out_win))
            # print(np.shape(out_win))
        # plt.figure()
        # plt.imshow(out_win)
        # plt.show()

        if i<=10:
            label = 1
        elif 10<i<=20:
            label =2
        elif 20<i<=30:
            label =3
        elif 30<i<=40:
            label =4
        elif 40<i<=50:
            label =5
        elif 50<i<=60:
            label =6
        elif 60<i<=70:
            label =7
        elif 70<i<=80:
            label =8
        else:
            label = np.NaN
        # binary_image_collection[i,0] = out_win
        binary_image_collection.append(out_win)
        # print(np.shape(out_win))
        # label_collection[i,0] = label
        label_collection.append(label)
    np.save(r'D:\UF\ECE\Fundamentals of ML\Github\assignment-03-Mohit-U-Patil\data.npy',binary_image_collection)
    print("\n\n shape of saved image collection {}".format(np.shape(binary_image_collection)))
    np.save(r'D:\UF\ECE\Fundamentals of ML\Github\assignment-03-Mohit-U-Patil\labels.npy',label_collection)
    print("shape of the label collection {}".format(np.shape(label_collection)))

"""
====================================================
========= Generate Features and Labels =============
====================================================
"""

if __name__ == '__main__':

    # To not call from command line, comment the following code block and use example below 
    # to use command line, call: python hw03.py K.jpg output

    # if len(sys.argv) != 3 and len(sys.argv) != 4:
    #     print('usage: {} <in_filename> <out_filename> (--debug)'.format(sys.argv[0]))
    #     sys.exit(0)
    #
    # in_fname = sys.argv[1]
    # out_fname = sys.argv[2]
    #
    # if len(sys.argv) == 4:
    #     debug = sys.argv[3] == '--debug'
    # else:
    #     debug = False
    #
    # #e.g. use
    # process_image('C:/Desktop/K.jpg','C:/Desktop/output.npy',debug=True)
    # process_image(in_fname,out_fname)

    calculator()
    PCA_reduction()
    EM(20, (0.5,0.5,0.5))
    EM(20, (0.4,0.6,0.7))

    #"""to run the image processing part, run the following function"""
    #"""if there is an error due to local file path, please go to the mass_process() function and put appropriate paths"""
    # mass_process()


    # a = np.load(r'D:\UF\ECE\Fundamentals of ML\Github\assignment-03-Mohit-U-Patil\binary_image_collection.npy',
    #                allow_pickle=True)
    # b = np.load(r'D:\UF\ECE\Fundamentals of ML\Github\assignment-03-Mohit-U-Patil\label_collection.npy',
    #             allow_pickle=True)
    # print(np.shape(a))
    # print(type(a))
    # print(b)
    # print(np.shape(b))
    # plt.figure()
    # plt.imshow(a[1])
    # plt.show()