# Copyright 2018, Plamen P. Angelov and Xiaowei Gu

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.




# This code is the Autonomous Learning Multi-Model System of First Order described in:
#==========================================================================================================
# P. Angelov, X. Gu, J. Principe, "Autonomous learning multi-model systems from data streams," 
# IEEE Transactions on Fuzzy Systems, vol.26(4), pp. 2213-2224, 2017.
#==========================================================================================================
# Please cite the paper above if this code helps.

# For any queries about the code, please contact Prof. Plamen P. Angelov and Mr. Xiaowei Gu 
# {p.angelov,x.gu3}@lancaster.ac.uk
# Programmed by Xiaowei Gu


import numpy
import scipy.io
import pandas
import math
import os
Omega=10 # Parameter for initialising the covariance matrixes. (Changeable)
Lambda0=0.8 # Threshold for adding new rules. The lower the value is, the larger the rule base will be. (Changeable)
Eta0=0.1 #Forgetting factor, recommended value 0~0.1. The higher the eta0 is, the quicker the system removes stale rule. (Changeable)
##
def Global_param_update(data,GlobalMean,GlobalX,K): # Update the global mean (GlobalMean) and average scalar product (GlobalX)
    data=numpy.array(data)
    GlobalMean=(GlobalMean*(K-1)+data)/K
    GlobalX=(GlobalX*(K-1)+numpy.sum(data**2))/K
    return GlobalMean,GlobalX

def Centre_lambda_calculator(ModeNumber,data,Centre,LocalX,Support,GlobalMean,GlobalX):
    # Calculate: the global density at the centres (CentreDensity); the local density of the new data sample at each data cloud (LocalDensity); the activation level of each fuzzy rule (CentreLambda)
    CentreDensity=numpy.zeros(ModeNumber)
    GlobalMean=numpy.matrix(GlobalMean)
    GDelta=GlobalX-numpy.sum(numpy.power(GlobalMean,2))
    for ii in range(0,ModeNumber):
        CentreDensity[ii]=1/(numpy.sum(numpy.power((Centre[ii,:]-GlobalMean),2))+GDelta)
    LocalDensity=numpy.zeros(ModeNumber)
    Ce1=Centre
    Lo1=LocalX
    Ce2=numpy.zeros(Ce1.shape)
    Lo2=numpy.zeros(Lo1.shape)
    for ii in range(0,ModeNumber):
        Ce2[ii,:]=(numpy.multiply(Ce1[ii,:],Support[ii])+data)/(Support[ii]+1)
        Lo2[ii]=(Lo1[ii]*Support[ii]+numpy.sum(numpy.power(data,2)))/(Support[ii]+1)
        AA=Lo2[ii]-numpy.sum(numpy.power(Ce2[ii,:],2))
        BB=numpy.sum(numpy.power((data-Ce2[ii,:]),2))
        if (AA+BB)==0:
            LocalDensity[ii]=1
        else:
            LocalDensity[ii]=AA/(AA+BB)
    if numpy.sum(LocalDensity)==0:
        CentreLambda=numpy.ones(ModeNumber)/ModeNumber
    else:
        CentreLambda=numpy.array(LocalDensity/numpy.sum(LocalDensity))
    return CentreLambda,CentreDensity,LocalDensity

def Testing_centre_lambda_calculator(ModeNumber,data,Centre,LocalX,Support):
    # Calculate the activation level of each fuzzy rule during the training stage
    LocalDensity=numpy.zeros(ModeNumber)
    Ce1=Centre
    Lo1=LocalX
    Ce2=numpy.zeros(Ce1.shape)
    Lo2=numpy.zeros(Lo1.shape)
    for ii in range(0,ModeNumber):
        Ce2[ii,:]=(numpy.multiply(Ce1[ii,:],Support[ii])+data)/(Support[ii]+1)
        Lo2[ii]=(Lo1[ii]*Support[ii]+numpy.sum(numpy.power(data,2)))/(Support[ii]+1)
        AA=Lo2[ii]-numpy.sum(numpy.power(Ce2[ii,:],2))
        BB=numpy.sum(numpy.power((data-Ce2[ii,:]),2))
        if (AA+BB)==0:
            LocalDensity[ii]=1
        else:
            LocalDensity[ii]=AA/(AA+BB)
    if numpy.sum(LocalDensity)==0:
        CentreLambda=numpy.ones(ModeNumber)/ModeNumber
    else:
        CentreLambda=numpy.array(LocalDensity/numpy.sum(LocalDensity))
    return CentreLambda

def System_output(data,CentreLambda,ModeNumber,A):
    # Calculate the system output
    Estimation=0
    data=numpy.append([1],data)
    for ii in range(0,ModeNumber):
        Estimation=Estimation+CentreLambda[ii]*data*A[ii,:].transpose()
    return Estimation

def Global_density_calculator(data,GlobalMean,GlobalX):
    # Calculate the global density at the new data sample
    GlobalMean=numpy.matrix(GlobalMean)
    GDelta=GlobalX-numpy.sum(numpy.power(GlobalMean,2))
    DataDensity=1/(numpy.sum(numpy.power(data-GlobalMean,2))+GDelta)
    return DataDensity

def Overlap_detection(ModeNumber,LocalDensity):
    # Detect the potential overlapping fuzzy rule
    Seq=[]
    count=0
    for ii in range(0,ModeNumber):
        if LocalDensity[ii]>Lambda0:
            count=count+1
            Seq.append(ii)
    Seq=numpy.array(Seq)
    if count>0:
        OL=1
        target=Seq[int(numpy.argmax(LocalDensity[Seq]))]
    else:
        OL=0
        target=0
    return OL,target

def Overlap_remove(data,K,ModeNumber,CentreLambda,LambdaSum,Centre,LocalX,Support,Index,A,C,W,target):
    # Remove the fuzzy rule and the corresponding data cloud from the system
    Seq=numpy.array(range(0,ModeNumber))
    Seq1=Seq[Seq!=target]
    Centre1=Centre[Seq1,:]
    Centre1=numpy.append(Centre1,(data+Centre[target,:])/2,axis=0)
    LocalX1=LocalX[Seq1]
    LocalX1=numpy.append(LocalX1,(numpy.matrix(numpy.sum(numpy.power(data,2)))+LocalX[target])/2,axis=0)
    Index1=Index[Seq1]
    Index1=numpy.append(Index1,[K],axis=0)
    CentreLambda1=CentreLambda[Seq1]/numpy.sum(CentreLambda[Seq1])
    CentreLambda1=numpy.append(CentreLambda1,[0],axis=0)
    LambdaSum1=LambdaSum[Seq1]
    LambdaSum1=numpy.append(LambdaSum1,[0],axis=0)
    Support1=Support[Seq1]
    Support1=numpy.append(Support1,[math.ceil((1+Support[target])/2)],axis=0)
    A1=A[Seq1,:]
    A1=numpy.append(A1,A[target,:],axis=0)
    if Seq!=[]:
        C1=[C[i] for i in Seq1]
        C1=C1+[numpy.multiply(numpy.eye(W+1),Omega)]
    else:
        C1=[numpy.multiply(numpy.eye(W+1),Omega)]
    return Centre1,LocalX1,Support1,Index1,LambdaSum1,CentreLambda1,A1,C1

def New_data_cloud_add(data,K,ModeNumber,LambdaSum,Centre,LocalX,Support,Index,A,C,W,A_inherit):
    # Add a new data cloud and the corresponding fuzzy rule to the system
    ModeNumber=ModeNumber+1
    Centre=numpy.append(Centre,numpy.matrix(data),axis=0)
    LocalX=numpy.append(LocalX,numpy.matrix([numpy.sum(numpy.power(data,2))]),axis=0)
    Support=numpy.append(Support,[1],axis=0)
    Index=numpy.append(Index,[K],axis=0)
    A=numpy.append(A,A_inherit,axis=0)
    C=C+[numpy.multiply(numpy.eye(W+1),Omega)]
    LambdaSum=numpy.append(LambdaSum,[0],axis=0)
    return ModeNumber,LambdaSum,Centre,LocalX,Support,Index,A,C

def Local_parameter_update(data,Centre,LocalX,Support):# Update the meta-parameters of the nearest data cloud and the fuzzy rule
    dist0=scipy.spatial.distance.cdist(data, Centre, 'euclidean')
    idx0=numpy.argmin(dist0,axis=1)
    Centre[idx0,:]=(numpy.multiply(Centre[idx0,:],Support[idx0])+data)/(Support[idx0]+1)
    LocalX[idx0]=(LocalX[idx0]*Support[idx0]+numpy.sum(numpy.power(data,2)))/(Support[idx0]+1)
    Support[idx0]=Support[idx0]+1
    return Centre,LocalX,Support

def Stale_data_cloud_remove(ModeNumber,LambdaSum,CentreLambda,Centre,LocalX,Support,Index,K,A,C,W):
    # Remove the stale data cloud and the corresponding fuzzy rule
    LambdaSum1=LambdaSum+CentreLambda
    Utility=numpy.zeros(ModeNumber)
    for ii in range(0, ModeNumber):
        if Index[ii]!=K:
            Utility[ii]=LambdaSum1[ii]/(K-Index[ii])
        else:
            Utility[ii]=1
    Seq=numpy.array(numpy.where(Utility>=Eta0))
    ModeNumber1=Seq.size
    if ModeNumber1<ModeNumber:
        Centre1=Centre[numpy.ndarray.tolist(Seq)[0],:]
        LocalX1=LocalX[numpy.ndarray.tolist(Seq)[0],:]
        Index1=Index[numpy.ndarray.tolist(Seq)[0]]
        LambdaSum1=LambdaSum1[numpy.ndarray.tolist(Seq)[0]]
        Support1=Support[numpy.ndarray.tolist(Seq)[0]]
        A1=A[numpy.ndarray.tolist(Seq)[0],:]
        C1=[C[i] for i in numpy.ndarray.tolist(Seq)[0]]
        CentreLambda1=CentreLambda[numpy.ndarray.tolist(Seq)[0]]/numpy.sum(CentreLambda[numpy.ndarray.tolist(Seq)[0]])
    else:
        A1=A[:]
        Centre1=Centre[:]
        LocalX1=LocalX[:]
        Index1=Index[:]
        LambdaSum1=LambdaSum1[:]
        Support1=Support[:]
        C1=C[:]
        CentreLambda1=CentreLambda[:]
    return ModeNumber1,LambdaSum1,CentreLambda1,Centre1,LocalX1,Support1,Index1,A1,C1
        
def Consequent_parameters_update(ModeNumber,data,y,CentreLambda,A,C):
    # Update the consequent parameters
    A1=A[:];
    C1=C[:];
    xe=numpy.append(numpy.matrix([1]),data,axis=1)
    for ii in range(0,ModeNumber):
        C1[ii]=C[ii]-numpy.multiply((C[ii]*xe.transpose()*xe*C[ii]),CentreLambda[ii])/(1+numpy.multiply(CentreLambda[ii],xe*C[ii]*xe.transpose()))
        A1[ii,:]=A[ii,:]+numpy.multiply((C1[ii]*xe.transpose()*(y-xe*A[ii,:].transpose())).transpose(),CentreLambda[ii])
    return A1,C1

#################
def ALMMo1_regression_learning(datain,dataout):
    # This is for training an ALMMo system for regression
    K=1
    L,W=datain.shape
    Centre=datain[0,:]
    GlobalMean=numpy.array(datain[0,:])
    GlobalX=numpy.sum(numpy.array(datain[0,:])**2)
    LocalX=numpy.array(numpy.sum(numpy.power(datain[0,:],2),axis=1))
    Support=numpy.array([1])
    ModeNumber=1
    LambdaSum=numpy.array([1])
    Index=numpy.array([1])
    A=numpy.matrix(numpy.zeros(W+1))
    C=[numpy.multiply(numpy.eye(W+1),Omega)]
    Estimation=numpy.zeros(L)
    for ii in range(1,L):
        K=K+1
        GlobalMean,GlobalX=Global_param_update(datain[ii,:],GlobalMean,GlobalX,K)
        CentreLambda,CentreDensity,LocalDensity=Centre_lambda_calculator(ModeNumber,datain[ii,:],Centre,LocalX,Support,GlobalMean,GlobalX)
        Estimation[ii]=System_output(datain[ii,:],CentreLambda,ModeNumber,A)
        DataDensity=Global_density_calculator(datain[ii,:],GlobalMean,GlobalX)
        if DataDensity>numpy.amax(CentreDensity) or DataDensity<numpy.amin(CentreDensity):
            OL,target=Overlap_detection(ModeNumber,LocalDensity)
            if OL==1:
                Centre,LocalX,Support,Index,LambdaSum,CentreLambda,A,C=Overlap_remove(datain[ii,:],K,ModeNumber,CentreLambda,LambdaSum,Centre,LocalX,Support,Index,A,C,W,target)
            else:
                A_inherit=numpy.matrix(numpy.sum(A,axis=0)/ModeNumber)
                ModeNumber,LambdaSum,Centre,LocalX,Support,Index,A,C=New_data_cloud_add(datain[ii,:],K,ModeNumber,LambdaSum,Centre,LocalX,Support,Index,A,C,W,A_inherit)
        else:
            Centre,LocalX,Support=Local_parameter_update(datain[ii,:],Centre,LocalX,Support)
        CentreLambda,CentreDensity,LocalDensity=Centre_lambda_calculator(ModeNumber,datain[ii,:],Centre,LocalX,Support,GlobalMean,GlobalX)
        ModeNumber,LambdaSum,CentreLambda,Centre,LocalX,Support,Index,A,C=Stale_data_cloud_remove(ModeNumber,LambdaSum,CentreLambda,Centre,LocalX,Support,Index,K,A,C,W)
        A,C=Consequent_parameters_update(ModeNumber,datain[ii,:],dataout[ii,0],CentreLambda,A,C)
    SystemParam={} # The trained system
    SystemParam['Centre']=Centre # Gentres of the identified data clouds
    SystemParam['GlobalMean']=GlobalMean # Global mean
    SystemParam['GlobalX']=GlobalX # Global average scalar product
    SystemParam['LocalX']=LocalX # Average scalar products of the identified data clouds
    SystemParam['Index']=Index # Time indexes at which the fuzzy rules are initialised
    SystemParam['Support']=Support # Supports of the identified data clouds
    SystemParam['LambdaSum']=LambdaSum # Accumulated activation levels of fuzzy rules
    SystemParam['ModeNumber']=ModeNumber # Number of the identified data clouds
    SystemParam['K']=K # Current time instance
    SystemParam['A']=A # Consequent parameters of the identified fuzzy rules
    SystemParam['C']=C # Coveriance Matrixes
    return Estimation,SystemParam # Estimation: output of the ALMMo-1 system

#################
def ALMMo1_classification_learning(datain,dataout):
    # This is for training an ALMMo system for classification
    K=1
    L,W=datain.shape
    Centre=datain[0,:]
    GlobalMean=numpy.array(datain[0,:])
    GlobalX=numpy.sum(numpy.array(datain[0,:])**2)
    LocalX=numpy.array(numpy.sum(numpy.power(datain[0,:],2),axis=1))
    Support=numpy.array([1])
    ModeNumber=1
    LambdaSum=numpy.array([1])
    Index=numpy.array([1])
    A=numpy.matrix(numpy.zeros(W+1))
    C=[numpy.multiply(numpy.eye(W+1),Omega)]
    Estimation=numpy.zeros(L)
    for ii in range(1,L):
        K=K+1
        GlobalMean,GlobalX=Global_param_update(datain[ii,:],GlobalMean,GlobalX,K)
        CentreLambda,CentreDensity,LocalDensity=Centre_lambda_calculator(ModeNumber,datain[ii,:],Centre,LocalX,Support,GlobalMean,GlobalX)
        Estimation[ii]=numpy.round(System_output(datain[ii,:],CentreLambda,ModeNumber,A))
        DataDensity=Global_density_calculator(datain[ii,:],GlobalMean,GlobalX)
        if DataDensity>numpy.amax(CentreDensity) or DataDensity<numpy.amin(CentreDensity):
            OL,target=Overlap_detection(ModeNumber,LocalDensity)
            if OL==1:
                Centre,LocalX,Support,Index,LambdaSum,CentreLambda,A,C=Overlap_remove(datain[ii,:],K,ModeNumber,CentreLambda,LambdaSum,Centre,LocalX,Support,Index,A,C,W,target)
            else:
                A_inherit=numpy.matrix(numpy.sum(A,axis=0)/ModeNumber)
                ModeNumber,LambdaSum,Centre,LocalX,Support,Index,A,C=New_data_cloud_add(datain[ii,:],K,ModeNumber,LambdaSum,Centre,LocalX,Support,Index,A,C,W,A_inherit)
        else:
            Centre,LocalX,Support=Local_parameter_update(datain[ii,:],Centre,LocalX,Support)
        CentreLambda,CentreDensity,LocalDensity=Centre_lambda_calculator(ModeNumber,datain[ii,:],Centre,LocalX,Support,GlobalMean,GlobalX)
        A,C=Consequent_parameters_update(ModeNumber,datain[ii,:],dataout[ii,0],CentreLambda,A,C)
    SystemParam={} # The trained system
    SystemParam['Centre']=Centre # Gentres of the identified data clouds
    SystemParam['GlobalMean']=GlobalMean # Global mean
    SystemParam['GlobalX']=GlobalX # Global average scalar product
    SystemParam['LocalX']=LocalX # Average scalar products of the identified data clouds
    SystemParam['Index']=Index # Time indexes at which the fuzzy rules are initialised
    SystemParam['Support']=Support # Supports of the identified data clouds
    SystemParam['LambdaSum']=LambdaSum # Accumulated activation levels of fuzzy rules
    SystemParam['ModeNumber']=ModeNumber # Number of the identified data clouds
    SystemParam['K']=K # Current time instance
    SystemParam['A']=A # Consequent parameters of the identified fuzzy rules
    SystemParam['C']=C # Coveriance Matrixes
    return Estimation,SystemParam # Estimation: output of the ALMMo-1 system

#################
def ALMMo1_regression_testing(datain,SystemParam): # Use the trained ALMMo system for regression on validation data
    Centre=SystemParam['Centre']
    LocalX=SystemParam['LocalX']
    Support=SystemParam['Support']
    ModeNumber=SystemParam['ModeNumber']
    A=SystemParam['A']
    L,W=datain.shape
    Estimation=numpy.zeros(L)
    for ii in range(0,L):
        CentreLambda=Testing_centre_lambda_calculator(ModeNumber,datain[ii,:],Centre,LocalX,Support)
        Estimation[ii]=System_output(datain[ii,:],CentreLambda,ModeNumber,A)
    return Estimation # Estimation: output of the ALMMo-1 system

    
#################
def ALMMo1_classification_testing(datain,SystemParam): # Use the trained ALMMo system for binary classification on validation data
    Centre=SystemParam['Centre']
    LocalX=SystemParam['LocalX']
    Support=SystemParam['Support']
    ModeNumber=SystemParam['ModeNumber']
    A=SystemParam['A']
    L,W=datain.shape
    Estimation=numpy.zeros(L)
    for ii in range(0,L):
        CentreLambda=Testing_centre_lambda_calculator(ModeNumber,datain[ii,:],Centre,LocalX,Support)
        Temp=System_output(datain[ii,:],CentreLambda,ModeNumber,A)
        if Temp>0.5:
            Estimation[ii]=1
        else:
            Estimation[ii]=0
    return Estimation # Estimation: output of the ALMMo-1 system
