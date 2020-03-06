# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:05:18 2019

@author: id127392
"""

import numpy as np
import bottleneck
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,Dense
from keras.models import load_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import util
import os

isPrinted = False
reductionModel = None
encodedLayerNumber = None

#--------------------------------------------------------------------------
def reduce(flatActivations, batchFlatActivations):
    # reduce the activations
    numInstances = len(flatActivations)

    if numInstances > 1:
        util.thisLogger.logInfo("---------- start of activation data reduction ----------")
        reducedFlatActivations = reduce_autoenc(flatActivations, batchFlatActivations) 
        
    if numInstances > 1:
        util.thisLogger.logInfo("----------- end of activation data reduction ----------\n")
        
    return reducedFlatActivations

#--------------------------------------------------------------------------
def saveReducedData(filename, flatActivations, y_train):        
    classes = np.unique(y_train)
    util.thisLogger.logInfo("\n%s classes found in entire training data"%(classes))
    
    # create a csv file for each class
    csvFileNames = []
    for dataClass in classes:
        util.thisLogger.logInfo("saving data for class %s to csv file"%(dataClass))
        csvFileName = "%s_%s.csv"%(filename, dataClass) 
        #flatActivations = list(filter(lambda x: (y_train[i] == dataClass), flatActivations))
        #list(filter(lambda x: x[1]['name'] == 'orange', enumerate(fruits)))
        filteredActivations = []
        for index in range(len(flatActivations)):
            if(y_train[index] == dataClass):
                filteredActivations.append(flatActivations[index])
        #filteredActivations = np.asarray(list([(i, x) for i, x in enumerate(flatActivations) if y_train[i] == dataClass]))
        #np.asarray(filteredActivations)
        labels = np.arange(len(filteredActivations[0]))
        filteredActivations = np.concatenate(([labels], filteredActivations), axis=0) # axis=0 means add rows
        util.saveToCsv(csvFileName, filteredActivations)
        csvFileNames.append(csvFileName)
        
    return 

#----------------------------------------------------------------------------
def reduce_autoenc(flatActivations, batchFlatActivations):
    # reduces the activations for an unseen instance and returns 
    # the reduced activations and the neuron labels
    global reductionModel
    encodedOutput = None

    # Take only the first and middle layers of the model
    encoder = Model(inputs=reductionModel.input, outputs=reductionModel.layers[1].output)

    processorType = util.getParameter('ProcessorType')
    if processorType == "GPU":
        encodedOutput = encoder.predict(flatActivations, steps=1)
    else:
        encodedOutput = encoder.predict(np.array(flatActivations))

    # change to python lists
    encodedOutput = encodedOutput.tolist()
    
    util.thisLogger.logInfo("Activations reduced from %s to %s elements"%(len(flatActivations[0]), len(encodedOutput[0])))       
    
    return encodedOutput

# ---------------------------------------------------------------------------
# Reduction technique training
#----------------------------------------------------------------------------
#--------------------------------------------------------------------------
def train(flatActivations, batchFlatActivations):
    # Perform any model trainig that is required by the activation reduction technique
    numInstances = len(flatActivations)
    if numInstances > 1:
        util.thisLogger.logInfo("---------- start of activation data reduction training ----------")    
        train_autoencoder(flatActivations, batchFlatActivations)
        util.thisLogger.logInfo("----------- end of activation data reduction training ----------\n")      

#--------------------------------------------------------------------------
def train_autoencoder(flatActivations, batchFlatActivations):
    # TODO: remove batchFlatActivations when happy that it is scaled up to handle flat activations
    global reductionModel
    autoencoder_trained = None

    # check if there is already a model saved
    modelFileName = getModelFileName()
    try:
        autoencoder_trained = load_model(modelFileName)
    except Exception as e:
        util.thisLogger.logInfo("%s autoencoder model could not be loaded: %s, training a new one"%(modelFileName,e))
        
    if(autoencoder_trained == None):
        autoencoder_trained = train_undercompleteAutoencoder(flatActivations, batchFlatActivations)
    
        # save the reduction model
        autoencoder_trained.save(modelFileName, include_optimizer=True)
    else:
        util.thisLogger.logInfo("%s autoencoder model loaded"%(modelFileName))

    # set the reduction model
    reductionModel = autoencoder_trained    

#----------------------------------------------------------------------------
def getModelFileName():
    datasetName = util.getParameter('DatasetName')
    classes = util.getParameter('DataClasses')
    classes = np.asarray(classes.replace('[','').replace(']','').split(',')).astype(int)
    classesStr = '%sclasses'%(len(classes))
    for c in classes:
        classesStr += str(c)

    numActivationTrainingInstances = util.getParameter('NumActivationTrainingInstances')
    
    modelFileName = '%s/models/reduce/%s_%s_single_undercomp_%s.h5'%(os.getcwd(),datasetName,classesStr,str(numActivationTrainingInstances))
    util.thisLogger.logInfo('Reduction model filename: %s'%(modelFileName))
    #modelFileName = '/home/jupyter/deepactistream/models/reduce/%s_%s_single_undercomp_%s.h5'%(datasetName,classesStr,str(numActivationTrainingInstances))
    return modelFileName

#----------------------------------------------------------------------------
def train_undercompleteAutoencoder(flatActivations, batchFlatActivations):
    # trains an undercomplete autoencoder on the activations from the training 
    # data
    
    useBatches = True
    
    data = None
    input_size = None
    processorType = util.getParameter('ProcessorType')
    if processorType == "GPU":
        if useBatches == False:
            data = np.array(flatActivations)
        input_size = len(flatActivations[0])
    else:
        # activations are a list/numpy array
        data = np.array(flatActivations)
        input_size = flatActivations[0].size()[0]

    autoencoder = design_autoencoder(input_size)
        
    epochs = 50
    
    autoencoder_trained = None
    # training in batches
    if(useBatches):
        autoencoder = trainInBatches(autoencoder, batchFlatActivations, epochs)
    else:
        act_train, act_valid = train_test_split(data, test_size=0.33, shuffle= True)
        autoencoder_trained = autoencoder.fit(act_train, act_train, batch_size=128,epochs=epochs,verbose=1,validation_data=(act_valid, act_valid))
    
        # plot
        loss = autoencoder_trained.history['loss']
        val_loss = autoencoder_trained.history['val_loss']
        epochs = range(epochs)
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
    
    return autoencoder

#----------------------------------------------------------------------------
def design_autoencoder(input_size):
    hidden_size =  100
    output_size = input_size
    util.thisLogger.logInfo('Input size of ' + str(input_size) + ' will be reduced to a size of ' + str(hidden_size))
    
    # create autoencoder network
    inputLayer = Input(shape=(input_size,))

    numGpus = 1; # get
    device = '/gpu:1'   # assume 2 GPUs (0 and 1)
    if(numGpus == 1):
        device = '/gpu:0'
            
    # Encoder - place on separate GPU as it is big. Is keras placing the gradients on GPU0 anyway?
    with tf.device(device): 
        hiddenLayer = Dense(hidden_size, activation='relu')(inputLayer)

    # Decoder
    outputLayer = Dense(output_size, activation='sigmoid')(hiddenLayer)
        
    # train autoencoder (encoder + decoder)
    autoencoder = Model(inputs=inputLayer, outputs=outputLayer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

#----------------------------------------------------------------------------
def trainInBatches(autoencoder, batchFlatActivations, epochs):
    for epochNum in range(1,epochs+1):
        util.thisLogger.logInfo('Epoch %s'%(epochNum))
        for batchIndex in range(len(batchFlatActivations)):
            batch = batchFlatActivations[batchIndex]
            util.thisLogger.logInfo('Training autoencoder with batch %s of %s'%(batchIndex+1,len(batchFlatActivations)))
            act_train, act_valid = train_test_split(np.asarray(batch), test_size=0.33, shuffle= True)
            trainingLoss = autoencoder.train_on_batch(act_train,act_train)
            testloss = autoencoder.test_on_batch(act_valid, act_valid, sample_weight=None, reset_metrics=False)
            util.thisLogger.logInfo('Training loss: %s'%(trainingLoss))
            util.thisLogger.logInfo('Test loss: %s:'%(testloss))

        print ("Reconstruction Loss:", autoencoder.evaluate(act_train, act_train, verbose=0))
    return autoencoder

