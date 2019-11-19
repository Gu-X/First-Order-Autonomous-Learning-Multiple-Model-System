import numpy
import scipy.io
import pandas
import math
import os
import sklearn.metrics
from ALMMo1_System import ALMMo1_classification_learning
# Import training function of ALMMo-1 system for classification 
from ALMMo1_System import ALMMo1_classification_testing
# Import function of the ALMMo-1 system for classification during validation stage
##
# Read the data
dirpath = os.getcwd() 
dr =dirpath+'\\classificationexample.mat'
mat_contents = scipy.io.loadmat(dr)
datain=mat_contents['tradata']
dataout=mat_contents['tralabel']
## Trianin stage
datain=numpy.matrix(datain) # Input
dataout=numpy.matrix(dataout) # The labels of the respective data samples
Estimation,SystemParam=ALMMo1_classification_learning(datain,dataout)#Train the ALMMo-1 system
# Estimation: output of the ALMMo system (estimated labels); SystemParam: the trained ALMMo system

## Validation stage
datain=mat_contents['tesdata']
dataout=mat_contents['teslabel']
datain=numpy.matrix(datain)
dataout=numpy.matrix(dataout)
estimation=ALMMo1_classification_testing(datain,SystemParam)# Validate the trained ALMMo-1 system
print(sklearn.metrics.confusion_matrix(dataout,estimation)) # Calculate the confusion matrix
