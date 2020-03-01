# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:26:27 2019

@author: id127392
"""
from keras.datasets import cifar10
import util
import numpy as np

isDataLoaded = False
x_train = None
y_train = None
x_test = None
y_test = None

#--------------------------------------------------------------------------
def getFilteredData():
    x_train, y_train, x_test, y_test = getAllData()
    
    # Only train and test on the first 5 classes
#     dataFilter = util.getParameter('DataFilter')
#     if(dataFilter == '5classes'):
#         classes = [0,1,2,3,4]    # Todo: specify this in setup file?
#     elif(dataFilter == '2classes'):
    classes = util.getParameter('DataClasses')
    classes = np.asarray(classes.replace('[','').replace(']','').split(',')).astype(int)
    util.thisLogger.logInfo('Data classes to be used: %s'%(classes))
    
    #print('before filtering: classarray: %s, x_train: %s'%(classes, len(x_train)))
    #print('classarray: %s, y_train: %s'%(classes, len(y_train)))
    x_train, y_train, x_test, y_test = filterData(x_train, y_train, x_test, y_test, classes)
    #print('after filtering: classarray: %s, x_train: %s'%(classes, len(x_train)))
    #print('classarray: %s, y_train: %s'%(classes, len(y_train)))
        
    return x_train, y_train, x_test, y_test

#--------------------------------------------------------------------------
def filterData(x_train, y_train, x_test, y_test, classes):
    x_train, y_train = util.filterDataByClass(x_train, y_train, classes)
    x_test, y_test = util.filterDataByClass(x_test, y_test, classes)    
    return x_train, y_train, x_test, y_test

#--------------------------------------------------------------------------
def getOutOfFilterData():
    x_train, y_train, x_test, y_test = getAllData()
    dataDiscrepancyClass = util.getParameter('DataDiscrepancyClass')
    classes = [dataDiscrepancyClass]
    x_train, y_train, x_test, y_test = filterData(x_train, y_train, x_test, y_test, classes)
    return x_train, y_train, x_test, y_test

#--------------------------------------------------------------------------
def getAllData():
    global isDataLoaded
    global x_train
    global y_train
    global x_test
    global y_test
    
    if isDataLoaded == False:
        # get the cifar 10 dataset from Keras and normalise
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype('float32')/255
        x_train = x_train.astype('float32')/255
        isDataLoaded = True
    
    return x_train, y_train, x_test, y_test


    