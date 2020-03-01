# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:21:59 2019

@author: id127392
"""
from dsce import extract, reduce
import numpy as np
import datetime
import h5py
import os
import util


#----------------------------------------------------------------------------
#yes
def getActivations(x_train, numActivationTrainingInstances, model, dnnModel, y_train):   
    util.thisLogger.logInfo("------ start of activation data extraction for training data -------")
    startTime = datetime.datetime.now()
    
    # Only get activations from the instances that are correctly classified
    y_predict = np.argmax(dnnModel.predict(x_train),axis=1)
    
    # The DNN is trained to output 0 or 1 only.
    # get the original classes it was trained on and transform the outputs
    classes = util.getParameter('DataClasses')
    classes = np.asarray(classes.replace('[','').replace(']','').split(',')).astype(int)
    util.thisLogger.logInfo('Data classes to be used: %s'%(classes))
    count = 0
    for c in classes:
        y_predict = np.where(y_predict==count, c, y_predict) 
        count += 1
    
    incorrectPredictIndexes = []
    for i in range(0, len(y_predict)-1):
        if (y_predict[i] != y_train[i]):
            incorrectPredictIndexes.append(i)
    
    x_train = np.delete(x_train, incorrectPredictIndexes, axis=0)
    y_train = np.delete(y_train, incorrectPredictIndexes, axis=0)
    y_predict = np.delete(y_predict, incorrectPredictIndexes, axis=0)
        
    # train in batches
    activationTrainingBatchSize = util.getParameter('ActivationTrainingBatchSize')
    
    if numActivationTrainingInstances == -1:
        numActivationTrainingInstances = len(x_train)
        
    xData = x_train[:numActivationTrainingInstances,]
    batchData = list(util.chunks(xData, activationTrainingBatchSize))

    activationData = []
    numBatches = len(batchData)
    batchActivationData = [[] for i in range(numBatches)]
    for batchIndex in range(numBatches):
        batch = batchData[batchIndex]
        util.thisLogger.logInfo("Training batch " + str(batchIndex+1) + " of " + str(len(batchData)) + " (" + str(len(batch)) + " instances)")
        # Get activations and set up streams for the training data
        # get reduced activations for all training data in one go
        
        # Train in a loop
        util.thisLogger.logInfo(str(len(batch)) + " instances selected from training data")
        
        activations, numLayers = extract.getActivationData(model, batch)
        batchActivationData[batchIndex].append(activations)
        activationData.append(activations)
        
        util.thisLogger.logInfo("Filter Layers: DNN has %s activation layers, getting activation data for %s instances."%(numLayers,len(batch)))
           
    endTime = datetime.datetime.now()
    util.thisLogger.logInfo('Total training time: ' + str(endTime - startTime))
    util.thisLogger.logInfo("------- end of activation data extraction for training data --------")
    util.thisLogger.logInfo("")
    
    return numLayers, batchData, activationData, batchActivationData 
       