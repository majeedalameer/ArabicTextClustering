#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 19:38:59 2019

@author: mishz
"""

import GWO as gwo
import csv
import numpy as np
import time
import neurolab as nl
import costNN
import minisom as minisom
import solution
import pandas as pd
import GWO as gwo
import numpy as np
import neurolab as nl
import costNN
from data_training import optimized_som
import solution
import pandas as pd
from minisom_qe_error import minisom_qe_error




def selector(algo,func_details,popSize,Iter,trainDataset,testDataset, epochs):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]


    dataTrain="~/MishzThesis/datasets/"+trainDataset
    dataTest="~/MishzThesis/datasets/"+testDataset
    
    print('done reading train data')    
        
    Dataset_train=pd.read_csv(dataTrain)
    print('done reading train data')
    Dataset_test=pd.read_csv(dataTest)
    print('done reading test data')

          
    TotalTrinInput=np.shape(Dataset_train)[0]    # number of instances in the train dataset
    TotalTrainFeatures=np.shape(Dataset_train)[1]-1 #number of features in the train dataset
    
    TotalTestInput=np.shape(Dataset_test)[0]    # number of instances in the test dataset
    TotalTestFeatures=np.shape(Dataset_test)[1]-1 #number of features in the test dataset
      
    ### splitting the datasets into inputs and outputs
    
    c=list(Dataset_train.columns)
    c.remove('filename')
    trainOutput= Dataset_train.filename #this is the output of train data
    trainInput=Dataset_train.loc[:,c] #this is the input of train data
                
    print('train inputs shape is #')
    print(trainInput.shape)
    print('train outputs shape is #')
    print(trainOutput.shape)
    print('done with training data')
    
    
    c=list(Dataset_test.columns)
    c.remove('filename')
    testOutput= Dataset_test.filename
    testInput=Dataset_test.loc[:,c]

    print('DONE reading and dividing the data')
    
    
    ## NOW WTHE CREATION OF THE KOHONEN MAP
    print('NOW THE NETWORK')
    ############
    ############
    ############
    ############

    numofClusters =  np.unique(trainOutput).shape[0]
    print('number of clusters is: '+ str(numofClusters))
    print ('number of features is: ' + str(TotalTrainFeatures))

    #creating the kohonen network
    #net = nl.net.newc([[0, 1]]*TotalTrainFeatures, numofClusters)
    #net = nl.net.newc([[0, 1]] * TotalTrainFeatures, numofClusters*numofClusters)


#%%
    map_dim= 16
    gwo_dim= map_dim*map_dim*TotalTrainFeatures #this represents the dimension

    print('done with the gwo dimension, its= ' + str(gwo_dim))


    #x= gwo.GWO(getattr(CostNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)

    # def GWO(objf,lb,ub,gwo_dim,SearchAgents_no,Max_iter, map_dim):
    x= gwo.GWO(minisom_qe_error,lb,ub,gwo_dim,popSize,Iter,trainInput,map_dim)

    
    #after getting the x, this should be injected to the weights layer in the self]organizing maps, trained and tested
    
    
    #evaluate the som model based on the training set
    print('done with gwo')

    train_results= optimized_som(trainInput,trainOutput, testInput , testOutput, x, map_dim, epochs)

    print('done with whole, 7mdellah')

    #x.trainF1=train_results[0]
    #x.trainPR=train_results[1]
    #x.trainR=train_results[2]
    #x.trainAcc=train_results[3]
       
    #evaluate som model based on the testing set   
    #test_results=minisom.AdjustedSom(x,testInput,testOutput,net)
    #x.testF1=test_results[0]
    #x.testPR=test_results[1]
    #x.testR=test_results[2]
    #x.testAcc=test_results[3]

    return 0
