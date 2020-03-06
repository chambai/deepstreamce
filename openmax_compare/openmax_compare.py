from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential,load_model,save_model,model_from_config,model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model


from evt_fitting_compare import weibull_tailfitting, query_weibull
from compute_openmax_compare import computeOpenMaxProbability,recalibrate_scores
from openmax_compare_utils import compute_distance

import scipy.spatial.distance as spd
import h5py

import libmr

import numpy as np
import scipy

import pickle
import matplotlib.pyplot as plt

label=None
displayFullResults = False

# input image dimensions
img_rows, img_cols = 32, 32
x_train_nd = None
y_train_nd = None
x_test_nd = None
y_test_nd = None
x_train_ce = None
y_train_ce = None
x_test_ce = None
y_test_ce = None
classes = None


def seperate_data(x,y):
    ind = y.argsort()  # returns an array of indices of the same shape as y that index data along the given axis in sorted order
    sort_x = x[ind[::-1]]
    sort_y = y[ind[::-1]]

    dataset_x = []
    dataset_y = []
    mark = 0

    
    for a in range(len(sort_y)-1):
        if sort_y[a] != sort_y[a+1]:
            dataset_x.append(np.array(sort_x[mark:a]))
            dataset_y.append(np.array(sort_y[mark:a]))
            mark = a
        if a == len(sort_y)-2:
            dataset_x.append(np.array(sort_x[mark:len(sort_y)]))
            dataset_y.append(np.array(sort_y[mark:len(sort_y)]))
    return dataset_x,dataset_y

def compute_feature(x,model):
    score = get_activations(model,len(model.layers)-1,x) # get activations at final layer
    fc8 = get_activations(model,len(model.layers)-2,x) # get activations at penultimate layer
    return score,fc8

def compute_mean_vector(feature):
    return np.mean(feature,axis=0)

def compute_distances(mean_feature,feature,category_name): # mean_feature = MAV, feature = activations, category_name = y
    eucos_dist, eu_dist, cos_dist = [], [], []
    eu_dist,cos_dist,eucos_dist = [], [], []
    for feat in feature:   # for the activations in each instance (function is called class by class)
        #print(mean_feature.shape)
        #print(feat.shape)
        mean_feature.flatten()   # flatten the MAV for this instance
        # calculate the euclid, cosine and eucos distances between the MAV and the activations
        eu_dist += [spd.euclidean(mean_feature, feat)]
        cos_dist += [spd.cosine(mean_feature, feat)]
        eucos_dist += [spd.euclidean(mean_feature, feat)/200. + spd.cosine(mean_feature, feat)]
    distances = {'eucos':eucos_dist,'cosine':cos_dist,'euclidean':eu_dist}
    return distances

def filterData(x_train, y_train, x_test, y_test, classes):
    x_train, y_train = filterDataByClass(x_train, y_train, classes)
    x_test, y_test = filterDataByClass(x_test, y_test, classes)    
    return x_train, y_train, x_test, y_test

def filterDataByClass(x_data, y_data, class_array):
    ix = np.isin(y_data, class_array)
    ixArry = np.where(ix)
    indexes = ixArry[0] # list of indexes that have specified classes
    x_data = x_data[indexes]
    y_data = y_data[indexes]
    return x_data, y_data

def getNdData():
    global x_train_nd
    global y_train_nd
    global x_test_nd
    global y_test_nd
    return x_train_nd, y_train_nd, x_test_nd, y_test_nd

def getCeData():
    global x_train_ce
    global y_train_ce
    global x_test_ce
    global y_test_ce
    return x_train_ce, y_train_ce, x_test_ce, y_test_ce
    
    
def loadData(ndClasses, ceClass):   
    global x_train_nd
    global y_train_nd
    global x_test_nd
    global y_test_nd
    global x_train_ce
    global y_train_ce
    global x_test_ce
    global y_test_ce
    global classes
    global label
       
    classes = ndClasses
    num_classes = len(classes)
    label = classes
    
    # the data, shuffled and split between train and test sets
    # get data the DNN was trained on
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32')/255
    x_train = x_train.astype('float32')/255
    x_train_nd, y_train_nd, x_test_nd, y_test_nd = filterData(x_train, y_train, x_test, y_test, classes)

    # get CE data
    x_train_ce, y_train_ce, x_test_ce, y_test_ce = filterData(x_train, y_train, x_test, y_test, ceClass)

    if K.image_data_format() == 'channels_first':
        x_train_nd = x_train_nd.reshape(x_train_nd.shape[0], 1, img_rows, img_cols)
        x_test_nd = x_test_nd.reshape(x_test_nd.shape[0], 1, img_rows, img_cols)
        x_train_ce = x_train_ce.reshape(x_train_ce.shape[0], 1, img_rows, img_cols)
        x_test_ce = x_test_ce.reshape(x_test_ce.shape[0], 1, img_rows, img_cols)
        #x_valid = x_valid.reshape(x_valid.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train_nd = x_train_nd.reshape(x_train_nd.shape[0], img_rows, img_cols, 3)
        x_test_nd = x_test_nd.reshape(x_test_nd.shape[0], img_rows, img_cols, 3)
        x_train_ce = x_train_ce.reshape(x_train_ce.shape[0], img_rows, img_cols, 3)
        x_test_ce = x_test_ce.reshape(x_test_ce.shape[0], img_rows, img_cols, 3)   
        input_shape = (img_rows, img_cols, 3)


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input,K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch,0])[0]
    return activations

def get_correct_classified(pred,y):
    pred = (pred > 0.5 ) * 1
    res = np.all(pred == y,axis=1)
    return res

def get_correct_classified_indexes(pred,y):
    #pred = (pred > 0.5 ) * 1
    res = np.all(pred == y,axis=0)
    return res

def get_correct_predictions(dnnModel, x_train, y_train, classes):
    # Only get activations from the instances that are correctly classified
    y_predict = np.argmax(dnnModel.predict(x_train),axis=1)
    
    # The DNN is trained to output 0 or 1 only.
    # get the original classes it was trained on and transform the outputs
    count = 0
    for c in classes:
        y_predict = np.where(y_predict==count, c, y_predict) 
        count += 1
    
    incorrectPredictIndexes = []
    for i in range(0, len(y_predict)-1):
        if (y_predict[i] != y_train[i]):
            incorrectPredictIndexes.append(i)
    #print(incorrectPredictIndexes)
    
    
    x_train = np.delete(x_train, incorrectPredictIndexes, axis=0)
    y_train = np.delete(y_train, incorrectPredictIndexes, axis=0)
    y_predict = np.delete(y_predict, incorrectPredictIndexes, axis=0)
    y_train = y_train.reshape(len(y_train))

    # convert predicted data to 0s and 1s to it can be compared with softmax from the DNN 
    # (the DNN always produces 0 and 1 as it was trained on categorical data of 0 and 1 regrdless of the actual class number in the data)
    count = 0
    #print(classes)
    for c in classes:
        y_train_nd[y_train_nd == c] = count
        y_test_nd[y_test_nd == c] = count
        count += 1
        
    return x_train, y_train

def create_model(model):

    output = model.layers[-1]
    
    x1_test, y1_test = get_correct_predictions(model, x_train_nd, y_train_nd, classes)

    sep_x, sep_y = seperate_data(x1_test, y1_test)

    feature = {}
    feature["score"] = []
    feature["fc8"] = []
    weibull_model = {}
    feature_mean = []
    feature_distance = []

    for i in range(len(sep_y)):
        print (i, sep_x[i].shape)
        weibull_model[label[i]] = {}
        score,fc8 = compute_feature(sep_x[i], model) # gets activations at penultimate (fc8) and final layers (score = softmax values)
        mean = compute_mean_vector(fc8)
        distance = compute_distances(mean, fc8, sep_y)
        feature_mean.append(mean)
        feature_distance.append(distance)
    np.save('mean',feature_mean)
    np.save('distance',feature_distance)

def build_weibull(mean,distance,tail):
    weibull_model = {} 
    for i in range(len(mean)):
        weibull_model[label[i]] = {}        
        weibull = weibull_tailfitting(mean[i], distance[i], tailsize = tail)
        #print(weibull_model,label[i])
        weibull_model[label[i]] = weibull
    return weibull_model
        
def compute_openmax(model,imagearr, ALPHA_RANK, TAIL_SIZE):
    mean = np.load('mean.npy')
    distance = np.load('distance.npy', allow_pickle=True)
    #Use loop to find the good parameters
    #alpharank_list = [1,2,3,4,5,5,6,7,8,9,10]
    #tail_list = list(range(0,21))

#     Alpha_rank is the total number of top classes to revise.  
#     They are operating with 1000 calsses in original OpenMax  
#     Alpa_rank cannot be more than the total number of classes
    alpharank_list = [ALPHA_RANK]

    tail_list = [TAIL_SIZE]

    total = 0
    for alpha in alpharank_list:
        weibull_model = {}
        openmax = None
        softmax = None        
        for tail in tail_list:
            weibull_model = build_weibull(mean, distance, tail) 
            openmax , softmax = recalibrate_scores(weibull_model, label, imagearr, alpharank=alpha, display=displayFullResults)
    
            if displayFullResults:
                print ('Alpha ',alpha,' Tail ',tail)
                print ('++++++++++++++++++++++++++++')           
                print ('openmax shape',openmax.shape)
                print ('openmax array',openmax) # value array
                print ('openmax index of largest value',np.argmax(openmax)) # argmax returns the indices of the max value along an axis

                print ('softmax shape',softmax.shape)
                print ('softmax array',softmax)
                print ('softmax index of largest value',np.argmax(softmax))
                if np.argmax(openmax) == np.argmax(softmax):
                    if np.argmax(openmax) == 0 and np.argmax(softmax) == 0:            
                        print ('########## Parameters found ############')
                        print ('Alpha ',alpha,' Tail ',tail)                
                        print ('########## Parameters found ############')                
                        total += 1
                print ('----------------------------')
    return np.argmax(softmax),np.argmax(openmax)
    
def process_input(model,ind):
    imagearr = {}
    plt.imshow(np.squeeze(x_train[ind]))    
    plt.show()    
    image = np.reshape(x_train[ind],(1,28,28,1))
    score5,fc85 = compute_feature(image, model)    
    imagearr['scores'] = score5
    imagearr['fc8'] = fc85
    return imagearr

def compute_activation(model,img):
    imagearr = {}
    img = np.reshape(img,(1,32,32,3))
    
    score5,fc85 = compute_feature(img, model)
    imagearr['scores'] = score5
    imagearr['fc8'] = fc85
    return imagearr

def image_show(img,label):
    img = scipy.misc.imresize(np.squeeze(img),(28,28))
    img = img[:,0:28*28]    
    plt.imshow(img,cmap='gray')
    plt.show()


def openmax_unknown_class(model):
    f = h5py.File('HWDB1.1subset.hdf5','r')
    total = 0
    i = np.random.randint(0,len(f['tst/y']))
    print ('label',np.argmax(f['tst/y'][i]))
    print (f['tst/x'][i].shape)
    imagearr = process_other_input(model,f['tst/x'][i])
    compute_openmax(model,imagearr)

def openmax_known_class(model,y):
    total = 0
    for i in range(15):
        j = np.random.randint(0,len(y_train_cat[i]))
        imagearr = process_input(model,j)
        print (compute_openmax(model, imagearr))


