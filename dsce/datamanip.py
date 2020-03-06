# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:24:26 2019

@author: id127392
"""
import numpy as np
import tensorflow as tf
import sys

from itertools import chain
import util

#----------------------------------------------------------------------------
def flattenActivationBatches(activationBatches):
    util.thisLogger.logInfo("------ start of flatten activation data -------")

    flatInstances = [] 
    flatBatches = [[] for i in range(len(activationBatches))]
    for batchIndex in range(len(activationBatches)):
        util.thisLogger.logInfo("flattening batch %s of %s"%(batchIndex+1,len(activationBatches)))
        instances = activationBatches[batchIndex]
        flatBatchInstances = flattenActivations(instances)
        flatInstances.extend(flatBatchInstances)
        flatBatches[batchIndex] = flatBatchInstances
    
    util.thisLogger.logInfo("%s instances, each instance has %s activations"%(len(flatInstances),len(flatInstances[0])))
    util.thisLogger.logInfo("------- end of flatten activation data -------")
    
    return flatInstances, flatBatches

#----------------------------------------------------------------------------
def flattenActivations(activationData):
    flatInstances = []   
    for activationDict in activationData:
        numInstances = len(list(activationDict.values())[0])
        layerFlatValues = [[] for i in range(numInstances)]

        for l in range(len(list(activationDict.values()))):
            for i in range(numInstances): # num instances
                layerFlatValues[i].append(list(activationDict.values())[l][i].flatten())
        
        # flatten out the batches
        for flatInstValues in layerFlatValues:
            
            flatArry = np.array(flattenList(flatInstValues))
            flatInstances.append(flatArry)
             
    return flatInstances
    
#----------------------------------------------------------------------------
def flattenList(obj):
    obj = list(chain(*obj))
    obj = list(map(float, obj))
    return obj

#----------------------------------------------------------------------------
def normalizeFlatValues(flatActivations, isTrainingData, maxValue=None):
    # divides each element by the max value of the training data
    if isTrainingData:
        # find max value from the training data, then normalise
        maxValue = np.amax(flatActivations)

    flatActivations = flatActivations/maxValue
    
    return flatActivations, maxValue