# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:05:39 2019

@author: id127392
"""

from py4j.java_gateway import JavaClass
from py4j.java_gateway import JavaGateway
from py4j.java_collections import SetConverter, MapConverter, ListConverter
import numpy as np
import sys
import uuid
import json, codecs
import pickle
import io
import datetime
import multiprocessing
import tensorflow as tf
from numpy import linalg as LA
import util
from dsce import datastream

numCpus = multiprocessing.cpu_count()
moaGatewayStarted = False
gateway = None
process = False
results = None

#----------------------------------------------------------------------------
def setup(flatActivations, y_train, streamList, clustererList, javaDir, filePrefix, unseenDataList):
    global results
    results = None
    util.thisLogger.logInfo("---------- start of MCOD setup for activaton analysis----------")
    
    subprocessName = 'subprocess/MoaGateway.jar'
    global moaGatewayStarted
    global gateway
    if moaGatewayStarted == False:
        moaGatewayStarted = True;
        if moaGatewayStarted == True:
            gateway = JavaGateway()
        else:
            raise ValueError(subprocessName + ' has not started')
        createClusterers(flatActivations, y_train, streamList, clustererList, None, javaDir, filePrefix, unseenDataList)
    
    util.thisLogger.logInfo("----------- end of MCOD setup for activaton analysis----------\n")

#----------------------------------------------------------------------------
def checkConnection():
    nTrys = 10
    for i in range(nTrys):
        try:
            # create the attribute names for the flat data
            attributeNames = [str(i) for i in range(10)]
            # this is the first time we attemp to connect to moa gateway
            jAttributeNames = ListConverter().convert(attributeNames, gateway._gateway_client)
            break
        except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                util.thisLogger.logInfo('problem connecting to gateway: %s, %s, %s, %s, %s'%(e,exc_type, exc_obj, exc_tb, exc_tb.tb_lineno))
                util.thisLogger.logInfo('trying again (try %s of %s)...'%(i,nTrys))
                if nTrys-1 == i:
                    raise Exception("problem connecting to gateway: %s: %s" % (exc_tb.tb_lineno, e))
    


#----------------------------------------------------------------------------
def getAttributeNames(flatActivations):
    # create the attribute names for the flat data
    attributeNames = [str(i) for i in range(0, len(flatActivations[0]))]
    # this is the first time we attemp to connect to moa gateway, check the connection
    checkConnection()
    jAttributeNames = ListConverter().convert(attributeNames, gateway._gateway_client)
    return jAttributeNames

#----------------------------------------------------------------------------
def createClusterers(flatActivations, y_train, streamList, clustererList, batch, javaDir, filePrefix, unseenDataList):
            
    classes = np.unique(y_train)
    util.thisLogger.logInfo("\n%s classes found in entire training data"%(classes))
    
    jAttributeNames = getAttributeNames(flatActivations)
    
    # get data into java
    for dataClass in classes:
        # create stream
        stream = createStream(jAttributeNames)
        streamList[str(dataClass)] = stream
        # create clusterer
        clusterer = createClusterer(stream)
        clustererList[str(dataClass)] = clusterer
        util.thisLogger.logInfo("Created stream and cluster " + str(dataClass))
    
        # add training instances to class stream from csv file
        javaFile = "%s/%s_trainingactivations_%s.csv"%(javaDir,filePrefix,dataClass)
        strDataClass = str(dataClass)       
        
    # prepare trained MCODs for use
    util.thisLogger.logInfo("Number of CPUs: " + str(numCpus))
    numDiscrepancy, numNonDiscrepancy = datastream.getInstanceParameters()
    
    for i in clustererList.keys():
        numClassInstances = len([x.predictedResult for x in unseenDataList if x.predictedResult == int(i)])
        util.thisLogger.logInfo('Number of instances of class %s: %s'%(i,numClassInstances))
        trainFilename = '%s/%s_trainingactivations_%s.csv'%(javaDir,filePrefix,i)
        util.thisLogger.logInfo("creating trained clusterers for class " + str(i))
        gateway.entry_point.CreateTrainedClusterers(clustererList[str(i)], trainFilename, numClassInstances, numCpus)       

#----------------------------------------------------------------------------
def addToClusterer(stream, javaFile, ids):
    # get data into java
    jIds = ListConverter().convert(ids, gateway._gateway_client)
 
    util.thisLogger.logInfo(javaFile)
    gateway.entry_point.Moa_Clusterers_Outliers_Mcod_AddCsvDataToStream(stream, javaFile, jIds)

#----------------------------------------------------------------------------
def setAnalysisParameters_Mcod(clusterer, k, radius, windowSize):
    params = []
    params = gateway.entry_point.Moa_Clusterers_Outliers_MCOD_SetParameters(clusterer, k, radius, windowSize)
    util.thisLogger.logInfo('MCOD Parameters: %s'%(params))
    return params.split(',')

#----------------------------------------------------------------------------
def convert(object, gateway_client):
        ArrayList = JavaClass("java.util.ArrayList", gateway_client)
        java_list = ArrayList()
        java_list.addAll(object)
        return java_list

#----------------------------------------------------------------------------
def processStreamInstances(streamList, clustererList, numInstances, dataDir, prefix, processOnce=False, saveOutlierResults=False):   
    if processOnce:
        for i in streamList.keys():
            processStreamInstance(streamList[i], clustererList[i], i, numInstances, dataDir, prefix, saveOutlierResults)
    else:
        global process
        process = True
        while process:
            for i in streamList.keys():
                processStreamInstance(streamList[i], clustererList[i], i, numInstances, dataDir, prefix, saveOutlierResults)       
    util.thisLogger.logInfo('processing stopped')
    
#----------------------------------------------------------------------------
def stopProcessing():
    global process
    process = False
    util.thisLogger.logInfo('processing stop request received: process=' + str(process))

#----------------------------------------------------------------------------
def processStreamInstance(stream, clusterer, i, numInstances, dataDir, prefix, saveOutlierResults=False):
    global results
    if saveOutlierResults == True:
        if results == None:
            results = []
            results.append(['ID','Instance = class %s'%(util.getParameter('DataDiscrepancyClass')), 'Class', 'Outlier'])
    
    trainFilename = '%s/%s_trainingactivations_%s.csv'%(dataDir,prefix,i)

    numSamples = 1  

    try:
        while stream.hasMoreInstances():
            newInstEx = stream.nextIdInstance()
            if(newInstEx is not None):
                instId = newInstEx.getIdAsString()
                newInst = newInstEx.getInstanceExample().getData()
                
                if saveOutlierResults == True:
                    # determine if the instance is an outlier
                    outlierResult = 'clusterer not defined'
                    clusterResult = None
                    inlierResult = None
                    outlierResult = gateway.entry_point.Moa_Clusterers_Outliers_MCOD_addAndAnalyse(clusterer, newInst, trainFilename, numCpus)

                    if(numInstances == -1):
                        if(outlierResult[0] == 'DATA,OUTLIER,OUTLIER'):
                            util.thisLogger.logInfoColour("[%s] Activation data for stream %s, instance %s: %s" % (instId,i,numSamples,outlierResult), 'red')
                        elif (outlierResult[0] == 'DATA,OUTLIER,NOT_OUTLIER'):
                            util.thisLogger.logInfoColour("[%s] Activation data for stream %s, instance %s: %s" % (instId,i,numSamples,outlierResult), 'green')
                        else:
                            util.thisLogger.logInfoColour("[%s] Activation data for stream %s, instance %s: %s" % (instId,i,numSamples,outlierResult), 'magenta')

                    # at this point we don't know if the original instance was an ND instance or CE as this is on a different thread
                    # mark is as ND for now, and update relevant entries with CE when creating the results
                    result = [instId,'ND', i, outlierResult[0].replace('DATA,OUTLIER,','')]
                    results.append(result)
                else:
                    # add instance as a training instance and do not do any outlier anlysis on it
                    gateway.entry_point.Moa_Clusterers_Outliers_MCOD_processNewInstanceImplTrain(clusterer, newInst)
   
                numSamples += 1
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        util.thisLogger.logInfo('problem reading stream: %s, %s, %s, %s, %s'%(e,exc_type, exc_obj, exc_tb, exc_tb.tb_lineno))
        if gateway != None:
            gateway.entry_point.StopCreatingTrainedClusterers()       
        
#----------------------------------------------------------------------------   
def createStream(jActLabels):
    # set the stream
    stream = gateway.entry_point.Moa_Streams_AddStream_New(jActLabels)
        
    stream.prepareForUse()
    return stream
    
#----------------------------------------------------------------------------
def createClusterer(stream):
    # set the clusterer
    clusterer = gateway.entry_point.Moa_Clusterers_Outliers_MCOD_New()
    k = util.getParameter('mcod_k')
    radius = util.getParameter('mcod_radius')
    windowSize = util.getParameter('mcod_windowsize')
    setAnalysisParameters_Mcod(clusterer, k, radius, windowSize)
     
    clusterer.setModelContext(stream.getHeader())  
    clusterer.prepareForUse()
    return clusterer

#---------------------------------------------------------------------------
def setAnalysisParametersMcod(clusterer):
    k = util.getParameter('mcod_k')
    radius = util.getParameter('mcod_radius')
    windowSize = util.getParameter('mcod_windowsize')
    result = setAnalysisParameters_Mcod(clusterer, k, radius, windowSize)
    clusterer.prepareForUse()
    util.thisLogger.logInfo(result)