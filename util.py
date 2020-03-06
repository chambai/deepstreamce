# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:50:23 2019

@author: id127392
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import logging
import pandas as pd
import seaborn as sns
from scipy import stats
import csv
import datetime
import os
import subprocess
import time
import tensorflow as tf
from sklearn.manifold import TSNE

paramDefFile = None
setupFile = None
thisLogger = None
isLoggingEnabled = True
logPath = None
logLevel = 'INFO'
params = None
sns.set(color_codes=True)

#------------------------------------------------------------------------
def getFullFileName(relativeFileName):
    scriptDir = os.path.dirname(__file__) #<-- absolute dir the script is in
    absFilePath = os.path.join(scriptDir, relativeFileName)
    return absFilePath
    
#------------------------------------------------------------------------
def getAllParameters():
    global params
    if(params == None):
        # load param def file
        params = {}
        paramdef = getFileLines(paramDefFile)
        
        # get param names from def file
        paramNames = []
        for param in paramdef:
            splitParams = param.split("#")
            paramName = splitParams[0].strip()                         
            paramNames.append(paramName)
            
        # get param values from setup file
        setup = getFileLines(setupFile)
        paramValues = {}  
        setup = [s for s in setup if "#" not in s]
        for paramName in paramNames:
            for param in setup:
                splitParams = param.split("=")
                setupName = splitParams[0].strip()
                if paramName == setupName:
                    paramValues[paramName] = splitParams[1].strip()
                    
        # store params in dictionary
        for param in paramdef:
            splitParams = param.split("#")
            if len(splitParams) == 2:
                paramType = 'string'
                paramComment = splitParams[1].strip() 
            else:  
                paramType = splitParams[1].strip() 
                paramComment = splitParams[2].strip() 
                
            paramName = splitParams[0].strip() 
            if paramName in paramValues: 
                paramValue = paramValues[paramName]  
            else:
                raise ValueError('Variable: ' + paramName + ' is defined in param def but does not exist in the setup file: ' +  setupFile)
                                    
            params[paramName] = (paramType, paramComment, paramValue)
                    
#------------------------------------------------------------------------       
def getParameter(paramName):
    paramValue = None
    
    # get parameter values
    if(params == None):
        getAllParameters()
           
    # get parameter type and value
    paramType = params[paramName][0]
    if paramType == 'bool':
        paramValue = stringToBool(params[paramName][2])
    elif paramType == 'float':
        paramValue = float(params[paramName][2])
    elif paramType == 'int':
        paramValue = int(params[paramName][2])
    elif paramType == 'string':
        paramValue = params[paramName][2]
    elif paramType == 'stringarray':
        paramValue = []
        splitParams = params[paramName][2].split(',')
        for p in splitParams:
            paramValue.append(p.strip())
    else:
        # The value must be one of the values specified
        allowedValues = paramType.split(',')
        allowedValues = [x.strip() for x in allowedValues] # List comprehension
        paramValue = params[paramName][2]
        if paramValue not in allowedValues:
            print(paramValue + ' is not in the list of allowed values for parameter ' + paramName )
            print('Allowed values are: ' + paramType )
    
    return paramValue

#------------------------------------------------------------------------
def getFileLines(filename):
    file = open(filename, "r")
    if file.mode == 'r':
        setup = file.readlines()
    return setup

#------------------------------------------------------------------------
def storeSetupParamsInLog():
    setup = getFileLines(setupFile)
    thisLogger.logInfo("Setup file:  %s"%(setupFile))
    thisLogger.logInfo("----------------- start of setup parameters ----------------------")
    for s in setup:
        thisLogger.logInfo(s.strip())
    thisLogger.logInfo("------------------ end of setup parameters ------------------")
    thisLogger.logInfo("")

#------------------------------------------------------------------------
def stringToBool(string):
    if string == 'True': 
        return True
    elif string == 'False':
        return False
    else:
        raise ValueError
    
#------------------------------------------------------------------------   
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]  

#------------------------------------------------------------------------ 
def filterDataByClass(x_data, y_data, class_array):
    ix = np.isin(y_data, class_array)
    ixArry = np.where(ix)
    indexes = ixArry[0] # list of indexes that have specified classes
    x_data = x_data[indexes]
    y_data = y_data[indexes]
    return x_data, y_data

#------------------------------------------------------------------------
# saves a vector to a csv file
def saveToCsv(csvFilePath, vector, append= False):
    fileWriteMethod = 'w'
    if append == True:
        fileWriteMethod = 'a'
        
    with open(csvFilePath, fileWriteMethod) as csv_file:
        newVec = []       
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        
        for vec in vector:
            if tf.is_tensor(vec):
                newVec.append(vec.numpy())
            else:
                #print('data is not a tensor')
                newVec = vector
                break
                
        wr.writerows(newVec)
    
    now = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    print("%s : saved"%(now))
    
#------------------------------------------------------------------------
def readFromCsv(csvFile):
    data = []
    df = pd.read_csv(csvFile) 
    for index, row in df.iterrows():
        data.append(row)
    
    return np.asarray(data)

#------------------------------------------------------------------------
def getSetupFileDescription():
    global setupFile
    path = os.path.dirname(os.path.abspath(setupFile))
    print(path)
    description = setupFile.replace(path + "/", "").replace(".txt","")
    return description

#------------------------------------------------------------------------
def makeDirectory(directoryName):
    try:
        os.mkdir(directoryName)
    except OSError:
        print ("Creation of the directory %s failed:%s" %(directoryName, OSError))
    else:
        print ("Successfully created the directory %s " % directoryName)
    
#------------------------------------------------------------------------
def killMoaGateway():
    stream = os.popen('ps aux | grep -i moagateway')
    output = stream.read()
    #print(output)
    if 'MoaGateway' in output:
        print('yes')
        splitoutput = output.strip().split('\n')
        print(splitoutput)
        for line in splitoutput:
            pid = " ".join(line.split()).split(' ')[1] # remove multiple whitespaces, split on whitespace and get the second element, which is the process ID
            print('killing process: ' + pid)
            stream = os.popen('sudo kill ' + pid)
            stream = os.popen('ps aux | grep -i moagateway')
            output = stream.read()
            print(output)
            
#------------------------------------------------------------------------
def startMoaGateway():
    success = False
    #stream = os.popen('sudo java -Xmx393216m -jar /home/jupyter/deepactistream/subprocess/MoaGateway.jar')
    stream = os.popen('sudo java -Xmx393216m -jar subprocess/MoaGateway.jar')
    stream = os.popen('ps aux | grep -i moagateway')
    output = stream.read()
    #print(output)
    if 'MoaGateway' in output:
        success = True
    print('Started MoaGateway: %s'%(success))
    stream = os.popen('ps aux | grep -i moagateway')
    output = stream.read()
    print(output)
    return success

#------------------------------------------------------------------------
def createResults(unseenInstancesObjList, results, outputDir, outputName):
    # update results with correct discrepancy name
    discrepancyIds = [x.id for x in unseenInstancesObjList[0] if x.discrepancyName != 'ND' ]
    for result in results:
        if result[0] in discrepancyIds:
            result[1] = [x.discrepancyName for x in unseenInstancesObjList[0] if x.id == result[0]][0]
                        
    # calculations
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    precision = 0
    recall = 0
    fMeasure = 0
    accuracy = 0
    for result in results:
        if result[1] == 'CE' and result[3] == 'OUTLIER':
            tp += 1
        if result[1] == 'ND' and result[3] == 'OUTLIER':
            fp += 1
        if result[1] == 'ND' and result[3] == 'NOT_OUTLIER':
            tn += 1
        if result[1] == 'CE' and result[3] == 'NOT_OUTLIER':
            fn += 1
        if result[1] == 'CE' and result[3] == 'NO_OUTLIERS_REPORTED': # NO_OUTLIERS_REPORTED treated like an outlier
            tp += 1
        if result[1] == 'ND' and result[3] == 'NO_OUTLIERS_REPORTED': # NO_OUTLIERS_REPORTED treated like an outlier
            fp += 1
                
    if tp+fp != 0:
        precision = tp/(tp+fp) # ratio of correctly identified CE's to all outliers
    if tp+fn != 0:
        recall = tp/(tp+fn)    # ratio of correctly identified CE's to all CE's
    if precision + recall != 0:
        fMeasure = 2*precision*recall/(precision + recall)
    if tp+fp+tn+fn != 0:
        accuracy = (tp+tn)/(tp+fp+tn+fn) # accuracy is is not a good indicator of performance when the class distribution is imbalanced                             
                
    # store parameters with results
    k = getParameter('mcod_k')
    radius = getParameter('mcod_radius')
    windowSize = getParameter('mcod_windowsize')
    
    calculations = []       
    calculations.append(['mcod_k', 'mcod_radius', 'mcod_windowsize', ''])
    calculations.append([k, radius, windowSize, ''])
    calculations.append(['TP (CE-OUTLIER)', 'FP (ND-OUTLIER)', 'TN (ND-NOT_OUTLIER)', 'FN (CE-NOT_OUTLIER)'])
    calculations.append([tp, fp, tn, fn])
    calculations.append(['Accuracy','Precision', 'Recall', 'F-Measure'])
    calculations.append([accuracy, precision, recall, fMeasure])
    calculations.append(['', '', '', ''])
    finalResults = calculations
    [finalResults.append(x) for x in results]
                                
    saveToCsv('%s/%s_outlierresults.csv'%(outputDir,outputName), finalResults)


#------------------------------------------------------------------------
def getColourText(text, colour):
    printCode = None
    if(colour == 'red'):
        printCode = "\x1b[31m" + text + "\x1b[0m"
    elif(colour == 'green'):
        printCode = "\x1b[32m" + text + "\x1b[0m"
    elif(colour == 'blue'):
        printCode = "\x1b[34m" + text + "\x1b[0m"
    elif(colour == 'magenta'):
        printCode = "\x1b[35m" + text + "\x1b[0m"
    elif(colour == 'cyan'):
        printCode = "\x1b[36m" + text + "\x1b[0m"
    elif(colour == 'black'):
        printCode = "\x1b[30m" + text + "\x1b[0m"
    else:
        raise ValueError('Colour: ' + colour + ' is not a recognised colour')
        
    return printCode
    
#------------------------------------------------------------------------    
class Logger:    
    def __init__(self): # double underscores means the function is private
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        global logLevel
        if(logLevel == 'INFO'):
            logging.basicConfig(filename=logPath,level=logging.INFO)
        else:
            logging.basicConfig(filename=logPath,level=logging.DEBUG)
        
    #class methods always take a class instance as the first parameter
    def logInfo(self, item):
        global logLevel
        item = prefixDateTime(item)
        if(logLevel == 'INFO'):
            print(item)
            #print("\x1b[31m\"red\"\x1b[0m")
        global isLoggingEnabled
        if (isLoggingEnabled == True):
            logging.info(item)
            
    def logInfoColour(self, item, colour):
        global logLevel
        item = prefixDateTime(item)
        if(logLevel == 'INFO'):
            print(getColourText(item, colour))
            #print("\x1b[31m\"red\"\x1b[0m")
        global isLoggingEnabled
        if (isLoggingEnabled == True):
            logging.info(item)
            
    def logDebug(self, item):
        global logLevel
        item = prefixDateTime(item)
        if(logLevel == 'DEBUG'):
            print(item)
        global isLoggingEnabled
        if (isLoggingEnabled == True):
            logging.debug(item)
            
    def logError(self, item):
        item = prefixDateTime(item)
        print(item)
        global isLoggingEnabled
        if (isLoggingEnabled == True):
            logging.error(item)
            
    def closeLog(self):
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

#------------------------------------------------------------------------ 
def prefixDateTime(item):
    if item:
        item = '%s %s'%(datetime.datetime.now(),item)
    return item
     
    
