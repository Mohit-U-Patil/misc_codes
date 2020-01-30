"""
File:   hw02.py
Author: Mohit Patil
Date:   09/30/2019
Desc:   Assignment 02B 
    
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.colors import ListedColormap
import sklearn.metrics as sklmet
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn import neighbors
from matplotlib.colors import ListedColormap

def plotData(x1, t1, x2=None, t2=None, x3=None, t3=None, legend=[], outFile=None, title=None, xRange=None, yRange=None,
             showImage=True, xlabel=None,
             ylabel=None, type = 'line'):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the
       training data, the true function, and the estimated function
       edited the function- now takes in different labes, title, option to save the image'''
    if type == 'point':
        p1 = plt.plot(x1, t1, 'bo')  # plot training data
        if (x2 is not None):
            p2 = plt.plot(x2, t2, 'go')  # plot true value
        if (x3 is not None):
            p3 = plt.plot(x3, t3, 'r0')  # plot training data
    elif type =='line':
        p1 = plt.plot(x1, t1, 'b')  # plot training data
        if (x2 is not None):
            p2 = plt.plot(x2, t2, 'g')  # plot true value
        if (x3 is not None):
            p3 = plt.plot(x3, t3, 'r')  # plot training data

    # add title, legend and axes labels
    if ylabel:
        plt.ylabel(ylabel)  # label x and y axes
    if xlabel:
        plt.xlabel(xlabel)

    if legend:
        if (x2 is None):
            plt.legend((p1[0]), legend)
        if (x3 is None):
            plt.legend((p1[0], p2[0]), legend)
        else:
            plt.legend((p1[0], p2[0], p3[0]), legend)

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

def load_Data(filepath, label_axis = 'last', output = 'array', comments="#", delimiter=" ", unpack=False):
    from numpy import loadtxt
    loaded_array = loadtxt(filepath, comments=comments, delimiter=delimiter, unpack=unpack)
    loaded_array = np.array(loaded_array)
    print('number of classes {}'.format(np.unique(loaded_array[:,np.shape(loaded_array)[1]-1])))
    # print(type(loaded_array))
    return(loaded_array)

def split_classes(input_data):
    class0 = []
    class1 = []
    for i in np.arange(np.shape(input_data)[0]):
        if input_data[i, np.shape(input_data)[1]-1] == 0:
            class0.append(input_data[i,:])
        if input_data[i, np.shape(input_data)[1]-1] == 1:
            class1.append(input_data[i,:])
    class0 = np.array(class0)
    class1 = np.array(class1)
    # print('input data shape {}'.format(np.shape(input_data)))
    return class0, class1

def prob_gen_mod_trainer(train_data):
    '''from the lectures'''

    class0, class1 = split_classes(train_data)
    label_axis = np.shape(class0)[1]
    class0_labeless = class0[:, 0:(label_axis - 1)]
    class1_labeless = class1[:, 0:(label_axis - 1)]
    # print('class0 shape {}'.format(np.shape(class0)))
    # print('class1 shape {}'.format(np.shape(class1)))
    # print('class0_labeless {}'.format(np.shape(class0_labeless)))
    # print('class1_labeless {}'.format(np.shape(class1_labeless)))

    #Estimate the mean and covariance for each class from the training data
    mu1 = np.mean(class0_labeless, axis=0)
    # print(mu1)
    cov1 = np.cov(class0_labeless.T)
    # print(cov1)
    mu2 = np.mean(class1_labeless, axis=0)
    # print(mu2)
    cov2 = np.cov(class1_labeless.T)
    # print(cov2)
    # Estimate the prior for each class
    pC1 = class0_labeless.shape[0]/(class0_labeless.shape[0] + class1_labeless.shape[0])
    # print(pC1)
    pC2 = class1_labeless.shape[0]/(class0_labeless.shape[0] + class1_labeless.shape[0])
    # print(pC2)

    return mu1,mu2,cov1,cov2,pC1,pC2

def prob_gen_mod_classifier(test_data , train_data = None, mu1 = None, mu2 = None, cov1= None, cov2=None, pC1=None ,pC2=None):

    if train_data is not None:
        mu1, mu2, cov1, cov2, pC1, pC2 = prob_gen_mod_trainer(train_data)
    else:
        mu1 = mu1
        mu2 = mu2
        cov1 = cov1
        cov2 = cov2
        pC1 = pC1
        pC2 = pC2

    test_data_classified = test_data.copy()
    # print(np.shape(test_data_classified)[1] - 2)
    # print(np.shape(test_data_classified)[1] - 2)
    probabilities = np.zeros(shape=(np.shape(test_data_classified)[0],2))
    for i in np.arange(np.shape(test_data_classified)[0]):
        #look at the pdf for class 1
        y1 = multivariate_normal.pdf(test_data_classified[i,0:np.shape(test_data_classified)[1] - 1], mean=mu1, cov=cov1, allow_singular= False)
        # plt.imshow(y1)
        #look at the pdf for class 2
        y2 = multivariate_normal.pdf(test_data_classified[i,0:np.shape(test_data_classified)[1] - 1], mean=mu2, cov=cov2, allow_singular= False);
        # plt.imshow(y2)
        #Look at the posterior for class 1
        pos1 = (y1*pC1)/(y1*pC1 + y2*pC2 )
        # plt.imshow(pos1)
        #Look at the posterior for class 2
        pos2 = (y2*pC2)/(y1*pC1 + y2*pC2 )

        probabilities[i,0] = pos1
        probabilities[i,1] = pos2

        # plt.imshow(pos2)
        #Look at the decision boundary
        # plt.imshow(pos1>pos2)
        if (pos1>pos2):
            test_data_classified[i, np.shape(test_data)[1]-1] = 0
            # print('{} no. datapoint classified = {}'.format(i, 0))
        elif (pos1<pos2):
            test_data_classified[i, np.shape(test_data)[1]-1] = 1
            # print('{} no. datapoint classified = {}'.format(i, 1))
        else:
            test_data_classified[i, np.shape(test_data)[1]-1] = np.NaN
            # print('{} no. datapoint classified = {}'.format(i, np.NaN))


    confusion_matrix = sklmet.confusion_matrix(test_data[:,-1], test_data_classified[:,-1])
    # print('confusion_matrix= {}'.format(confusion_matrix))
    # print("Accuracy for Prob gen model Classifier: " + str(accuracy_score(test_data[:,-1], test_data_classified[:,-1]) * 100) + "%")
    return test_data_classified, probabilities

def cross_validation_probgen(training_data,cross_validation_list):
    result_matrix = np.zeros(shape=(2,len(cross_validation_list)))
    # result_matrix = np.array(result_matrix)
    result_matrix[0, :] = cross_validation_list
    for j in np.arange(len(cross_validation_list)):
        x = training_data[:, 0:np.shape(training_data)[1] - 1]
        y = training_data[:, - 1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=cross_validation_list[j])
        # print('x_train shape {}'.format(np.shape(x_train)))
        # print('y_train shape {}'.format(np.shape(y_train)))
        # print('x test shape {}'.format(np.shape(x_test)))
        # print('y test shape {}'.format(np.shape(y_test)))

        y_train = np.transpose(y_train)
        y_train = y_train.reshape((np.shape(y_train)[0],1))
        y_test = np.transpose(y_test)
        y_test = y_test.reshape((np.shape(y_test)[0], 1))

        train_data = np.concatenate((x_train, y_train),1)
        test_data = np.concatenate((x_test, y_test),1)
        test_data_classified, probabilities = prob_gen_mod_classifier(test_data, train_data)
        # print(test_data_classified)
        # print(test_data_classified[:,-1])
        # print(np.transpose(y_test))
        # print(np.transpose(test_data_classified[:, -1]))
        accuracy = accuracy_score(y_test, test_data_classified[:,-1]);
        result_matrix[1,j] = accuracy

    # print('the accuracy vs cross validation result is: with first row split ratio and second row accuracy \n {}'.format(result_matrix))
    return result_matrix

def knn_trainer(training_data, num_neighbors, weights, cross_validation_split = None):

    x = training_data[:,0:np.shape(training_data)[1] - 1]
    y = training_data[:, np.shape(training_data)[1] - 1]

    if cross_validation_split is not None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=cross_validation_split)
        # print('x_train shape {}'.format(np.shape(x_train)))
        # print('y_train shape {}'.format(np.shape(y_train)))
        # print('x test shape {}'.format(np.shape(x_test)))
        # print('y test shape {}'.format(np.shape(y_test)))
    else:
        x_train = x
        y_train = y
        # print('x_train shape {}'.format(np.shape(x_train)))
        # print('y_train shape {}'.format(np.shape(y_train)))

    clf_knn = neighbors.KNeighborsClassifier(num_neighbors, weights)
    clf_train = clf_knn.fit(x_train, y_train)

    if cross_validation_split is not None:
        outKN = clf_train.predict(x_test)
        # print("Accuracy for KNeighbors Classifier: " + str(accuracy_score(y_test, outKN) * 100) + "%")
        accuracy = accuracy_score(y_test, outKN);
        return accuracy

    if cross_validation_split is None:
        return clf_train

def knn_classifier(testing_data, clf_train):

    test_data_classified = testing_data.copy()
    x_test = test_data_classified[:, 0:np.shape(test_data_classified)[1] - 1]
    y_test = test_data_classified[:, np.shape(test_data_classified)[1] - 1]
    outKN = clf_train.predict(x_test)
    # print("Accuracy for KNeighbors Classifier: " + str(accuracy_score(y_test, outKN) * 100) + "%")
    accuracy = accuracy_score(y_test, outKN)
    test_data_classified[:, np.shape(testing_data)[1] - 1] = outKN
    # print(accuracy)
    confusion_matrix = sklmet.confusion_matrix(testing_data[:, -1], test_data_classified[:, -1])
    # print('confusion_matrix= {}'.format(confusion_matrix))
    return test_data_classified, accuracy

def cross_validation_knn(training_data, k_list,weights, cross_validation_list, plot_split= None):
    result_matrix = np.zeros(shape=(len(k_list) + 1,len(cross_validation_list)+1))
    # result_matrix = np.array(result_matrix)
    result_matrix[0, 1:len(cross_validation_list)+1] = cross_validation_list
    result_matrix[1:len(k_list) + 1,0] = k_list
    for i in np.arange(len(k_list)):
        for j in np.arange(len(cross_validation_list)):
            result_matrix[i+1,j+1] = knn_trainer(training_data, k_list[i], weights, cross_validation_list[j])
    print('the k vs cross validation result is: with first column k and first row validation split \n {}'.format(result_matrix))
    if plot_split is not None:
        plotData(result_matrix[1:len(k_list) + 1,0],result_matrix[1:len(k_list) + 1,cross_validation_list.index(plot_split)+1], title='k vs accuracy', xlabel='k values', ylabel='accuracy')
    return result_matrix

def k_plot_knn(training_data, testing_data, k_list,weights):
    result_matrix = np.zeros(shape=(len(k_list),(len(weights)*2)+1))
    # result_matrix = np.array(result_matrix)
    result_matrix[:, 0] = k_list
    for j in np.arange(len(weights)):
        for i in np.arange(len(k_list)):
            clf_train = knn_trainer(training_data, k_list[i], weights[j], None)
            testing_data_classified, result_matrix[i,j+1] = knn_classifier(testing_data, clf_train)
            result_matrix[i,j+3]= np.mean(testing_data_classified[:,-1] != testing_data[:,-1]) #error
    # print(result_matrix)
    plotData(result_matrix[:,0],result_matrix[:,1],result_matrix[:,0],result_matrix[:,2], legend=weights, title='k vs accuracy', xlabel='k values', ylabel='accuracy')
    plotData(result_matrix[:,0],result_matrix[:,3],result_matrix[:,0],result_matrix[:,4], legend=weights, title='k vs error', xlabel='k values', ylabel='error')
    plotData(result_matrix[:,0],result_matrix[:,1]/result_matrix[:,3],result_matrix[:,0],result_matrix[:,2]/result_matrix[:,4], legend=weights, title='k vs performance(accuracy/error)', xlabel='k values', ylabel='performance(accuracy/error)')

def ROC_plots(test_data, train_data):
    test_data_classified, probabilities = prob_gen_mod_classifier(test_data, train_data)
    fp, tp, threshold = sklmet.roc_curve(test_data[:,-1], test_data_classified[:,-1])
    # plotData(fp,tp, title = 'In built ROC plot for Prob Gen Classifier on 10dDataSet', xlabel='False Positive Rate', ylabel='True Positive Rate')

    # threshold_manual = np.arange(0,1, 0.05)
    # tpr_A = np.empty([np.shape(threshold_manual)[0], 1])
    # fpr_A = np.empty([np.shape(threshold_manual)[0], 1])
    # tpr_B = np.empty([np.shape(threshold_manual)[0], 1])
    # fpr_B = np.empty([np.shape(threshold_manual)[0], 1])
    # print(np.shape(threshold_manual))
    # for j in np.arange(0,np.shape(threshold_manual)[0]):
    #     ROC_A = np.empty([np.shape(test_data_classified)[0], 1])
    #     ROC_B = np.empty([np.shape(test_data_classified)[0], 1])
    #     for i in np.arange(0,np.shape(test_data_classified)[0]):
    #         if probabilities[i,0]>threshold_manual[j,]:
    #             ROC_A[i,0] = 0
    #         else:
    #             ROC_A[i,0] = 1
    #
    #         if probabilities[i,1]>threshold_manual[j]:
    #             ROC_B[i,0]= 1
    #         else:
    #             ROC_B[i,0]= 0
    #     fpr_A[j,0],tpr_A[j,0], thresholdA = sklmet.roc_curve(test_data[:,-1], ROC_A[:,0])
    #     fpr_B[j,0],tpr_B[j,0], thresholdB = sklmet.roc_curve(test_data[:,-1], ROC_B[:,0])
    # plotData(fpr_A,tpr_A, title = 'Manual 0class ROC plot for Prob Gen Classifier on 10dDataSet', xlabel='False Positive Rate', ylabel='True Positive Rate')
    # plotData(fpr_B,tpr_B, title = 'Manual 1class ROC plot for Prob Gen Classifier on 10dDataSet', xlabel='False Positive Rate', ylabel='True Positive Rate')

    fpr_A, tpr_A, thresholdA = sklmet.roc_curve(test_data[:, -1], probabilities[:, 1])
    plotData(fpr_A,tpr_A, title = 'InBuilt ROC plot for Prob Gen Classifier on 10dDataSet', xlabel='False Positive Rate', ylabel='True Positive Rate')

def plot_10dData(data1, data2, labels, xlabel=None, ylabel = None, title = None):
    # figure params
    h = .02  # step size in the mesh
    figsize = (17, 9)
    figure = plt.figure()
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(1,1,1)
    # Plot the training points
    ax.scatter(data1, data2, c=labels, cmap=cm_bright)
    # and testing points
    # ax.scatter(X_test[:, 0], X_test[:, 1], marker='+', c=y_test, cmap=cm_bright, alpha=0.6)
    if ylabel:
        plt.ylabel(ylabel)  # label x and y axes
    if xlabel:
        plt.xlabel(xlabel)
    if title:
        plt.title(title)
    plt.show()

'''------------------------------------------------------------------------------------------'''
#following lines load train data, train the parameters and the test on the testing set
trainig_data_crab = load_Data(r"D:\UF\ECE\Fundamentals of ML\Github\assignment-02-Mohit-U-Patil\CrabDatasetforTrain.txt", label_axis = 'last', output = 'array', comments="#", delimiter=" ", unpack=False)
test_data_crab = load_Data(r"D:\UF\ECE\Fundamentals of ML\Github\assignment-02-Mohit-U-Patil\CrabDatasetforTest.txt", label_axis = 'last', output = 'array', comments="#", delimiter=" ", unpack=False)
trainig_data_crab_nonsingular = trainig_data_crab[:,[0,1,2,3,4,5,7]]
test_data_crab_nonsingular = test_data_crab[:,[0,1,2,3,4,5,7]]

trainig_data_10d = load_Data(r"D:\UF\ECE\Fundamentals of ML\Github\assignment-02-Mohit-U-Patil\10dDataSetforTrain.txt", label_axis = 'last', output = 'array', comments="#", delimiter=" ", unpack=False)
test_data_10d = load_Data(r"D:\UF\ECE\Fundamentals of ML\Github\assignment-02-Mohit-U-Patil\10dDataSetforTest.txt", label_axis = 'last', output = 'array', comments="#", delimiter=" ", unpack=False)

"""for Probabilistic Generative Model"""
'''for cross validation'''
# result_matrix_prob_crab = cross_validation_probgen(trainig_data_crab_nonsingular, [0.1,0.2,0.3,0.4,0.5,0.7,0.8])
# result_matrix_prob_10d = cross_validation_probgen(trainig_data_10d, [0.1,0.2,0.3,0.4,0.5,0.7,0.8])

'''for running the test data'''
# mu1,mu2,cov1,cov2,pC1,pC2 = prob_gen_mod_trainer(trainig_data_crab_nonsingular)
# test_data_classified, probabilities = prob_gen_mod_classifier(test_data_crab_nonsingular, None, mu1,mu2,cov1,cov2,pC1,pC2)
# mu1,mu2,cov1,cov2,pC1,pC2 = prob_gen_mod_trainer(trainig_data_10d)
# prob_gen_mod_classifier(test_data_10d, None, mu1,mu2,cov1,cov2,pC1,pC2)
'''for plotting ROC'''
ROC_plots(test_data_10d, trainig_data_10d)

"""for KNN Model"""
'''for cross validation'''
# result_matrix_knn_crab = cross_validation_knn(trainig_data_crab,[2,3,4,5,6,7,8,9], 'distance', [0.3, 0.4, 0.5,0.6, 0.7], 0.4)
# result_matrix_knn_10d = cross_validation_knn(trainig_data_10d,[2,3,4,5,6,7,8,9], 'distance', [0.1, 0.2, 0.3, 0.4, 0.5], 0.4)

'''for running the test data'''
# clf_train_crab = knn_trainer(trainig_data_crab,2, 'distance', None)
# classified_crab = knn_classifier(test_data_crab, clf_train_crab)
k_plot_knn(trainig_data_crab, test_data_crab,[2,3,4,5,6,7,8,9],['uniform','distance'])

# clf_train_10d = knn_trainer(trainig_data_10d,3, 'distance', None)
# classified_10d = knn_classifier(test_data_10d, clf_train_10d)
k_plot_knn(trainig_data_10d, test_data_10d,[2,3,4,5,6,7,8,9],['uniform','distance'])

'''plotting 10d dataset'''
# plotData(trainig_data_10d[:,5], trainig_data_10d[:,6], title='10dDataSet 4th vs 5th features', xlabel='4th Feature', legend=None,
#          ylabel='5th Feature', type = 'point', cmap= trainig_data_10d[:,-1])
plot_10dData(trainig_data_10d[:,4], trainig_data_10d[:,5],trainig_data_10d[:,-1],title='10dDataSet 4th vs 5th features', xlabel='4th Feature', ylabel='5th Feature')