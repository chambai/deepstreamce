import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
from scipy.io import loadmat

from openmax_compare_utils import *
from evt_fitting_compare import weibull_tailfitting, query_weibull

try:
    import libmr
except ImportError:
    print ("LibMR not installed or libmr.so not found")
    print ("Install libmr: cd libMR/; ./compile.sh")
    sys.exit()


#---------------------------------------------------------------------------------
# params and configuratoins
NCHANNELS = 1 # There's no channels in VGG16 DNN
NCLASSES = 2  # Only training the DNN on 2 classes

#---------------------------------------------------------------------------------
def computeOpenMaxProbability(openmax_fc8, openmax_score_u):
    """ Convert the scores in probability value using openmax
    
    Input:
    ---------------
    openmax_fc8 : modified FC8 layer from Weibull based computation
    openmax_score_u : degree

    Output:
    ---------------
    modified_scores : probability values modified using OpenMax framework,
    by incorporating degree of uncertainity/openness for a given class
    
    """
    prob_scores, prob_unknowns = [], []
    for channel in range(NCHANNELS):
        channel_scores, channel_unknowns = [], []
        for category in range(NCLASSES):
            channel_scores += [sp.exp(openmax_fc8[channel, category])]

        total_denominator = sp.sum(sp.exp(openmax_fc8[channel, :])) + sp.exp(sp.sum(openmax_score_u[channel, :]))

        prob_scores += [channel_scores/total_denominator ]

        prob_unknowns += [sp.exp(sp.sum(openmax_score_u[channel, :]))/total_denominator]
        
    prob_scores = sp.asarray(prob_scores)
    prob_unknowns = sp.asarray(prob_unknowns)

    scores = sp.mean(prob_scores, axis = 0)
    unknowns = sp.mean(prob_unknowns, axis=0)
    modified_scores =  scores.tolist() + [unknowns]
    assert len(modified_scores) == NCLASSES + 1
    return modified_scores

#---------------------------------------------------------------------------------
def recalibrate_scores(weibull_model, labellist, imgarr,
                       layer = 'fc8', alpharank = 6, distance_type = 'eucos', display=False):
    """ 
    Given FC8 features for an image, list of weibull models for each class,
    re-calibrate scores

    Input:
    ---------------
    weibull_model : pre-computed weibull_model obtained from weibull_tailfitting() function
    labellist : ImageNet 2012 labellist
    imgarr : features for a particular image extracted using caffe architecture
    
    Output:
    ---------------
    openmax_probab: Probability values for a given class computed using OpenMax
    softmax_probab: Probability values for a given class computed using SoftMax (these
    were precomputed from caffe architecture. Function returns them for the sake 
    of convienence)

    """
    
    imglayer = imgarr[layer] # imglayer = activations of penultimate layer
    ranked_list = imgarr['scores'].argsort().ravel()[::-1]  # ranked_list = softmax values in probabilty order
    if display:
        print('ranked_list (softmax values in probability order): %s'%(ranked_list)) # [1 0]
    
    # alpha_weights = weightings of the top classes to revise
    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]  
    if display:
        print('alpha_weights (weightings of the top classes to revise): %s'%(alpha_weights)) # [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    # This reverses the alpha_weights
    ranked_alpha = sp.zeros(NCLASSES)   
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]
    if display:
        print('ranked_alpha: %s'%(ranked_alpha))
    
    
    # Now recalibrate each fc8 score for each channel and for each class
    # to include probability of unknown
    openmax_fc8, openmax_score_u = [], []
    for channel in range(NCHANNELS):
        channel_scores = imglayer[channel, :] # activations of penultimate layer per channel (but we only have 1 channel) channel was just a name used by the authors to describe multiple outputs from caffe model
        openmax_fc8_channel = []
        openmax_fc8_unknown = []
        count = 0
        for categoryid in range(NCLASSES):
            # get distance between current channel and mean vector
            category_weibull = query_weibull(labellist[categoryid], weibull_model, distance_type = distance_type)

            channel_distance = compute_distance(channel_scores, channel, category_weibull[0],
                                                distance_type = distance_type)
            
            # obtain w_score for the distance and compute probability of the distance
            # being unknown wrt to mean training vector and channel distances for
            # category and channel under consideration
            wscore = category_weibull[2][channel].w_score(channel_distance)
            if display:
                print ('wscore',wscore)
            modified_fc8_score = channel_scores[categoryid] * ( 1 - wscore*ranked_alpha[categoryid] )
            
            if display:
                print ('modified_fc8_score',modified_fc8_score)
            openmax_fc8_channel += [modified_fc8_score]
            openmax_fc8_unknown += [channel_scores[categoryid] - modified_fc8_score ]

        # gather modified scores fc8 scores for each channel for the given image
        openmax_fc8 += [openmax_fc8_channel]
        openmax_score_u += [openmax_fc8_unknown]
    openmax_fc8 = sp.asarray(openmax_fc8)
    openmax_score_u = sp.asarray(openmax_score_u)
    
    # Pass the recalibrated fc8 scores for the image into openmax    
    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u)
    softmax_probab = imgarr['scores'].ravel() 
    return sp.asarray(openmax_probab), sp.asarray(softmax_probab)


