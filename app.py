# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:48:17 2019

@author: id127392
"""
import os
from keras.models import load_model
import threading
import datetime
import time
import util
from dsce import dataset, train, datamanip, reduce, analyse, datastream
import tensorflow as tf
import os.path
from numpy import genfromtxt
import pandas as pd
import csv
import numpy as np

class App:
    def __init__(self):
        
        # Setup Parameters
        util.params = None
        self.dnnModelPath = util.getFullFileName(util.getParameter('DnnModelPath'))
        self.numTrainingInstances = util.getParameter('NumActivationTrainingInstances')
        self.timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.outputName = util.getSetupFileDescription() + '--' + self.timestamp
        self.outputDir = 'output/%s'%(self.outputName)
        util.makeDirectory(self.outputDir)
        util.isLoggingEnabled = util.getParameter('LoggingEnabled')
        util.logPath = self.outputDir + '/%s.log'%(self.outputName)
        util.logLevel = util.getParameter('LogLevel')
        util.thisLogger = util.Logger()
        util.storeSetupParamsInLog()
        
        # Setup memory environment
        self.processorType = processorType = util.getParameter('ProcessorType')
            
        self.startTime = datetime.datetime.now()
        
        self.streamList = None
        self.clustererList = None
        self.classMaxValues1 = None  # max value of raw activation data
        self.classMaxValues2 = None  # max value of reduced activation data
        
        self.flatActivations = None
        self.activationBatches = None
        self.batchFlatActivations = None
        self.reducedFlatActivations = None
        
    def loadDnnModel(self, resetParams=False):
        if resetParams == True:
            util.params = None
        
        # Load the pre-trained DNN model
        util.thisLogger.logInfo("Loading DNN model %s..."%(self.dnnModelPath))
        self.dnnModel = load_model(self.dnnModelPath)
        util.thisLogger.logInfo(self.dnnModel.summary())
        
        # Load the input dataset
        util.thisLogger.logInfo("Loading input data...")
        self.x_train, self.y_train, self.x_test, self.y_test = dataset.getFilteredData()
        # limit the number of y data to the num training instances
        if self.numTrainingInstances != -1:
            self.x_train = self.x_train[:self.numTrainingInstances]
            self.y_train = self.y_train[:self.numTrainingInstances]
        
                
    def getActivations(self, resetParams=False):
        if resetParams == True:
            util.params = None
        self.numLayers, self.batchData, self.activations, self.activationBatches = train.getActivations(self.x_train, self.numTrainingInstances, self.dnnModel, self.dnnModel, self.y_train)
        
        # flatten the data
        self.flatActivations, self.batchFlatActivations = datamanip.flattenActivationBatches(self.activationBatches)
        self.flatActivations = tf.constant(self.flatActivations)          
        
        # Normalize the activations
        self.flatActivations, self.classMaxValues1  = datamanip.normalizeFlatValues(self.flatActivations, True)
        
        # tidy up memory - remove variables no longer needed
        del self.batchData
        del self.activations
        del self.activationBatches    
        
    def reduceActivations(self, resetParams=False):
        if resetParams == True:
            util.params = None
        
        # flat activations provided as one list and as batches - let the reduction training technique decide which to use
        reduce.train(self.flatActivations, self.batchFlatActivations)
        
        del self.batchFlatActivations
        self.batchFlatActivations = None
        
        # Reduce the training data activations
        self.reducedFlatActivations = reduce.reduce(self.flatActivations, None)
        
        # Normalize the reduced activations
        self.reducedFlatActivations, self.classMaxValues2  = datamanip.normalizeFlatValues(self.reducedFlatActivations, True)
        
        # Store the reduced training data activations
        reduce.saveReducedData('%s/%s_trainingactivations'%(self.outputDir,self.outputName), self.reducedFlatActivations, self.y_train)
        
    def setupActivationAnalysis(self, resetParams=False):
        if resetParams == True:
            util.params = None

        self.streamList = {}
        self.clustererList = {}
        
        # make sure a new MOA gateway is setup for each run
        util.killMoaGateway()
        time.sleep(3)
        util.startMoaGateway()
        time.sleep(3)

        # get unseen data and their predictions so we can efficiently set up the empty MCOD clusterers
        datastream.unseenDataList = datastream.getInstancesWithResultsBatchObj()
        datastream.setPredictions(self.dnnModel)
        analyse.setup(self.reducedFlatActivations, self.y_train, self.streamList, self.clustererList, self.outputDir, self.outputName, datastream.unseenDataList)
            
    def processDataStream(self):        
        # start the thread to process the streams so that new instances get clustered
        thread1 = threading.Thread(target=analyse.processStreamInstances, args=(self.streamList, self.clustererList, self.numTrainingInstances, self.outputDir, self.outputName, False, True), daemon=True)
        thread1.start()    

        unseenInstancesObjList = datastream.startDataInputStream(self.streamList, self.clustererList, reduce.reductionModel, self.dnnModel, self.x_test, self.classMaxValues1, self.classMaxValues2, self.outputDir, self.outputName)
        
        # reshape into original array types
        unseenInstances = [x.instance for x in unseenInstancesObjList[0]]
        unseenResults = [x.correctResult for x in unseenInstancesObjList[0]]
            
        dataDiscrepancyClass = self.layerExtraction = util.getParameter("DataDiscrepancyClass");
        
        # append unseen instances to the training instances
        unseenInstances = np.append(unseenInstances, unseenResults, axis=1)
        classes = np.unique(self.y_train)
        for dataClass in classes:
            # Filter unseen instances to only include CE and data discrepancy class
            filteredInstances = list(filter(lambda x: (x[len(unseenInstances[0])-1] == dataClass or x[len(unseenInstances[0])-1] == dataDiscrepancyClass), unseenInstances))
            
            trainingActivations = util.readFromCsv('%s/%s_trainingactivations_%s.csv'%(self.outputDir,self.outputName,dataClass))
            labels = np.arange(len(trainingActivations[0]))
            labels = np.append(labels,len(labels))
            classValues = np.full((trainingActivations.shape[0],1), 'Train_' + str(dataClass))
            
            trainingActivations = np.append(trainingActivations, classValues, axis=1)   # axis=1 means add columns
            trainingActivations = np.concatenate((trainingActivations, filteredInstances), axis=0) # axis=0 means add rows
            
            trainingActivations = np.concatenate(([labels], trainingActivations), axis=0) # axis=0 means add rows        
        
        analyse.stopProcessing()
        thread1.join()
                   
        # capture any unprocessed instances
        analyse.processStreamInstances(self.streamList, self.clustererList, self.numTrainingInstances, self.outputDir, self.outputName, True, True)
            
        util.thisLogger.logInfo('End of instance processing')
        
        # get outlier results and store in csv
        if analyse.results != None:
            util.createResults(unseenInstancesObjList, analyse.results, self.outputDir, self.outputName)
                
        util.killMoaGateway()      
        endTime = datetime.datetime.now()
        util.thisLogger.logInfo('Total run time: ' + str(endTime - self.startTime))
        util.thisLogger.closeLog()
        