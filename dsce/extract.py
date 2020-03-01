# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:04:40 2019

@author: id127392
"""
import numpy as np
from numpy import linalg as LA
from keras import backend as K
from keract import get_activations, display_activations, display_heatmaps
from matplotlib import pyplot as plt
import os
import gc
from dsce import datamanip
import util
import tensorflow as tf

#---------------------------------------------------------------------------
def sliceDictOnKeys(dictionary, substring):
    return{k.lower():v for k,v in dictionary.items() if substring.lower() in k}
    
#---------------------------------------------------------------------------
def sliceDictNotOnKeys(dictionary, substring):
    return{k.lower():v for k,v in dictionary.items() if substring.lower() not in k}
    
#---------------------------------------------------------------------------
def normalizeDictValues(dictionary):
    # divide the dictionary values by the max value in the dictionary
    # This is the max 
    numInstances = list(dictionary.values())[0].shape[0]      
    numLayers = len(list(dictionary.values()))
    for l in range(numLayers): # Layers
        for i in range(numInstances): # Instances
            maxValue = np.amax(np.asarray(list(dictionary.values())[l][i]))
            list(dictionary.values())[l][i] = np.divide(list(dictionary.values())[l][i], maxValue)           
  
    return dictionary   

#---------------------------------------------------------------------------
def getActivationData(model, inData):
    activations = getActivations(model, inData)
    
    numLayers = np.asarray(list(activations.keys())).size
    
    # set all keys in the dictionary to lowercase
    activations =  {k.lower(): v for k, v in activations.items()}

    return activations, numLayers

#-----------------------------------------------------------------------
def getActivations(model, inData):
    actDict = {}

    #ActivationLayerExtraction
    layerExtraction = util.getParameter("ActivationLayerExtraction")
    layers = util.getParameter("IncludedLayers")
    
    if(layers == 'all'):
        layers = np.arange(len(model.layers))
    else:
        layers = np.asarray(layers.replace('[','').replace(']','').split(',')).astype(int)
        
    util.thisLogger.logInfo('Applying activation extraction to layers: %s'%(layers))
    
    for layerNum in layers:
        getLastLayerOutput = K.function([model.layers[0].input],
                                      [model.layers[layerNum].output])
        
        if(layerExtraction == "single"):
            # all values from one layer
            if layerNum in layers:
                layer_output = getLastLayerOutput([inData])
                actDict["activation" +str(layerNum)] = layer_output[0]  
        else:
            raise ValueError("layerExtraction name of '%s' not recognised: "%(layerExtraction))
        
    return actDict

