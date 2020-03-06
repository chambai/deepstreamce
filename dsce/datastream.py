# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:22:46 2019

@author: id127392
"""
import numpy as np
import random
from random import shuffle
import uuid
import tensorflow as tf
from numpy import linalg as LA
from dsce import extract, datamanip, reduce, analyse, dataset
import util

unseenDataList = None
   
#---------------------------------------------------------------------------- 
def startDataInputStream(streamList, clustererList, reductionModel, dnnModel, x_test, maxClassValues1, maxClassValues2, dataDir, filePrefix):
    numUnseenInstances = util.getParameter('NumUnseenInstances')
    util.thisLogger.logInfo("---------- start of data input stream ----------")
    unseenInstances, unseenResults = startDataInputStream_Time(streamList, clustererList, reductionModel, dnnModel, x_test, numUnseenInstances, 
                                                                 maxClassValues1, maxClassValues2, dataDir, filePrefix)
    util.thisLogger.logInfo("----------- end of %s data input stream ----------")
    
    return unseenInstances, unseenResults
            
#---------------------------------------------------------------------------- 
def startDataInputStream_Time(streamList, clustererList, reductionModel, dnnModel, x_test, numUnseenInstances, maxClassValues1, maxClassValues2, dataDir, filePrefix):
    instanceIndex = 0
    
    # calculate the total number of loops of instances
    batchHandle = True
    if(batchHandle):
        numLoops = 1
    else:
        numDiscrepancy, numNonDiscrepancy = getInstanceParameters()
        numLoops = numUnseenInstances//numNonDiscrepancy
    
    allUnseenInstances = []
    allUnseenResults = []
    allUnseenInstances = batchProcessObjList(allUnseenInstances, allUnseenResults, dnnModel, maxClassValues1, maxClassValues2, streamList, dataDir, filePrefix)
    
    return np.asarray(allUnseenInstances), np.asarray(allUnseenResults)
    
#----------------------------------------------------------------------------
def addInstancesByStream(allUnseenInstances, streamList, dataDir, filePrefix):
    # get a distinct list of predicted results
    predictions = set([x.predictedResult for x in allUnseenInstances])
    for prediction in predictions:
        # get all the instances for this prediction
        instances = [x for x in allUnseenInstances if x.predictedResult == prediction]
        
        streamName = str(prediction)
        ids = [str(x.id) for x in instances]
        
        instances = [x.instance for x in instances]
                     
        labels = np.arange(len(instances[0]))
            
        instances = np.concatenate(([labels], instances), axis=0) # axis=0 means add row
        instancesCsvFile = '%s/%s_unseenactivations_%s.csv'%(dataDir,filePrefix,streamName)
        util.saveToCsv(instancesCsvFile, instances)

        analyse.addToClusterer(streamList[streamName], instancesCsvFile, ids)

#----------------------------------------------------------------------------
def setPredictions(dnnModel):
    global unseenDataList
    instances = [x.instance for x in unseenDataList]
    instances = np.reshape(instances, (len(unseenDataList),32,32,3))
    instances = np.asarray(instances)
    predictions = np.argmax(dnnModel.predict(instances),axis=1)
    instances = [] # reset instances to save memory
    
    # The DNN is trained to output 0 or 1 only.
    # get the original classes it was trained on and transform the outputs
    classes = util.getParameter('DataClasses')
    classes = np.asarray(classes.replace('[','').replace(']','').split(',')).astype(int)
    util.thisLogger.logInfo('Data classes to be used: %s'%(classes))
    count = 0
    for c in classes:
        predictions = np.where(predictions==count, c, predictions) 
        count += 1
    
    for index in range(len(predictions)):
        unseenDataList[index].predictedResult = predictions[index]

#----------------------------------------------------------------------------
def batchProcessObjList(allUnseenDataList, allUnseenResults, dnnModel, maxClassValues1, maxClassValues2, streamList, dataDir, filePrefix):
    global unseenDataList
    
    util.thisLogger.logInfo('Start of instance processing,%s'%(len(unseenDataList)))
    # collect instances
    instances = [x.instance for x in unseenDataList]
    
    # todo: get dimensions from instance data so it will work with data other than cifar
    instances = np.reshape(instances, (len(unseenDataList),32,32,3))
    # delete instances from objects to save memory
    for u in unseenDataList:
        u.instance = []
    
    activations, numLayers = extract.getActivationData(dnnModel, [instances])
    instances = [] # reset instances to save memory

    # place activations in an activation batch
    activationBatch = [[] for i in range(1)]
    activationBatch[0].append(activations)

    flatActivations, flatActivationBatch = datamanip.flattenActivationBatches(activationBatch)
    flatActivations = tf.constant(flatActivations)

    # Normalize raw flat activations
    if maxClassValues1 != None :
        util.thisLogger.logInfo('Max value of raw activations: %s'%(maxClassValues1))
        flatActivations = flatActivations/maxClassValues1

    # reduce activations
    flatActivations = reduce.reduce(flatActivations, None)
    
    # normalise the reduced instance activations
    if maxClassValues2 != None :
        util.thisLogger.logInfo('Max value of reduced activations: %s'%(maxClassValues2))
        flatActivations = flatActivations/maxClassValues2
        
    # save flat instances in objects
    for index in range(len(flatActivations)):
        unseenDataList[index].instance = flatActivations[index]
        
    flatActivations = []  # delete flat activations

    for unseenData in unseenDataList:       
        isCorrectlyClassified = False
        outcome = ''
        if(unseenData.discrepancyName == 'ND' and unseenData.correctResult != unseenData.predictedResult):
            outcome = 'DNN predicted the wrong class, not adding this instance to the stream'
        elif(unseenData.discrepancyName == 'ND' and unseenData.correctResult == unseenData.predictedResult):
            outcome = 'DNN predicted the correct class, adding this instance to the correct stream'
            unseenData.streamNames.append(str(unseenData.correctResult))
            isCorrectlyClassified = True
        else:
            unseenData.streamNames.append(str(unseenData.predictedResult))
            isCorrectlyClassified = True
        
        util.thisLogger.logInfo('[%s] %s: Actual Class: %s, Predicted Class: %s. %s'%(unseenData.id, unseenData.discrepancyName, unseenData.correctResult, unseenData.predictedResult, outcome))
          
        if(isCorrectlyClassified == True):
            allUnseenDataList.append(unseenData)
    
    # add instances to stream
    addInstancesByStream(allUnseenDataList, streamList, dataDir, filePrefix)
    
    return allUnseenDataList
    
#----------------------------------------------------------------------------
def getInstancesWithResultsBatchObj():
    # Gets the unseen data and also returns the result
    #np.random.seed(42) # get the same random indexes each time for now
    global unseenDataList
    unseenDataList = []
    numDiscrepancy, numNonDiscrepancy = getInstanceParameters()
    
    # if num discrepancies and num non-discrepancies match, set numNonDiscrepancy to zero, so none are generated
    if(numDiscrepancy == numNonDiscrepancy):
        numNonDiscrepancy = 0
    else:
        numNonDiscrepancy = numNonDiscrepancy - numDiscrepancy        
        
    # get non-discrepancy data
    f_x_train, f_y_train, f_x_test, f_y_test = dataset.getFilteredData()
          
    # get discrepancy data
    oof_x_train, oof_y_train, oof_x_test, oof_y_test = dataset.getOutOfFilterData()
    
    # collect non-discrepancy data
    for count in range(numNonDiscrepancy):
        index = random.randint(0,len(f_x_test)-1)
        nonDiscrepancyInstance = f_x_test[np.array([index])]
        result = f_y_test[index]
        i = UnseenData(nonDiscrepancyInstance, result, 'ND')
        unseenDataList.append(i)
    
    # collect discrepancy data
    for count in range(numDiscrepancy):
        index = random.randint(0,len(oof_x_test)-1)
        discrepancyInstance = oof_x_test[np.array([index])]
        result = oof_y_test[index]
        i = UnseenData(discrepancyInstance, result, 'CE')
        unseenDataList.append(i)
        
    shuffle(unseenDataList)
    
    # number the instances
    idCount = 1
    for data in unseenDataList:
        data.id = str(idCount)
        idCount += 1
    
    return unseenDataList

#----------------------------------------------------------------------------
class UnseenData:
    def __init__(self, instance, correctResult, discrepancyName):
        self.id = ''
        self.instance = instance
        self.correctResult = correctResult
        self.discrepancyName = discrepancyName
        self.predictedResult = 0
        self.streamNames = []
     
        
#----------------------------------------------------------------------------      
def getInstanceParameters():
    dataDiscrepancyFrequency = util.getParameter('DataDiscrepancyFrequency')  # i.e. 1in1
    splitData = dataDiscrepancyFrequency.split('in')
    numDiscrepancy = int(splitData[0].strip())  # first number is number of discrepancies
    numNonDiscrepancy = int(splitData[1].strip()) # second number is number of  non-discrepancies
      
    return numDiscrepancy, numNonDiscrepancy

        
        