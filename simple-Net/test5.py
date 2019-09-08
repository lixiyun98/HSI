import scipy.io
import numpy as np
from random import shuffle
import random
import scipy.ndimage
from skimage.util import pad
import os
import time
import pandas as pd
from utils import convertToOneHot
import math
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines')
parser.add_argument('--patch_size', type=int, default=11)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--train_size', type=float, default=0.8)
parser.add_argument('--isnorm', type=int, default=1)
parser.add_argument('--training_iters', type=int, default=10000)
opt = parser.parse_args()
# f=open("result_pavia.txt","w")
f=open("result_salinas.txt","w")
# f=open("result_india_pines.txt","w")
DATA_PATH = os.path.join(os.getcwd(),"Data")
Data = scipy.io.loadmat('./Data/' + opt.data + '.mat')[opt.data.lower()]
Label = scipy.io.loadmat('./Data/' + opt.data + '_gt.mat')[opt.data.lower() + '_gt']
# Data = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines.mat'))['indian_pines']
# Label = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_gt.mat'))['indian_pines_gt']
Height, Width, Band = Data.shape[0], Data.shape[1], Data.shape[2]
Num_Classes = len(np.unique(Label))-1
patch_size=opt.patch_size
batch_size=opt.batch_size
training_iters = opt.training_iters
Data = Data.astype(float)

if opt.isnorm == 1:
    for band in range(Band):
        Data[:, :, band] = (Data[:, :, band] - np.min(Data[:, :, band])) / (
        np.max(Data[:, :, band]) - np.min(Data[:, :, band]))


Data_Padding = np.zeros((Height+int(patch_size-1),Width+int(patch_size-1),Band))
for band in range(Band):
    Data_Padding[:,:,band] = pad(Data[:,:,band],int((patch_size-1)/2),'symmetric')

# def Patch(height_index,width_index):
#     """ function to extract patches from the orignal data """
#     transpose_array = np.transpose(Data_Padding,(2,0,1))
#     height_slice = slice(height_index, height_index + patch_size)
#     width_slice = slice(width_index, width_index + patch_size)
#     patch = transpose_array[:,height_slice, width_slice]
#     return np.array(patch)
def Patch(height_index,width_index):
    """ function to extract patches from the orignal data """
    # transpose_array = np.transpose(Data_Padding,(2,0,1))
    height_slice = slice(height_index, height_index + patch_size)
    width_slice = slice(width_index, width_index + patch_size)
    patch = Data_Padding[height_slice, width_slice,:]
    return np.array(patch)

All_Patches, All_Labels ,Al_Labels = [],[],[]
count=0
k=0
res=0

# All_Patches.append([])
# All_Labels.append([])
# Al_Labels.append([])
# for j in range(0,Width):
#     for i in range(0,Height):
#         if Label[i,j]!=0:
#             res+=1
# for i in range(int(res/batch_size)):
#     All_Patches.append([])
#     All_Labels.append([])
#     Al_Labels.append([])
for j in range(0,Width):
    for i in range(0,Height):
        curr_patch=Patch(i,j)
        if Label[i,j]!=0:
            All_Patches.append(curr_patch)
            # Label[i,j]=np.array(Label[i,j])
            # Labels = convertToOneHot(Label[i,j], num_classes=Num_Classes)
            All_Labels.append(Label[i, j]-1)
            count = count + 1
        # if count % batch_size == 0:
        #     k = k + 1
        # if Label[i,j]!=0:
print(curr_patch.shape)
All_Labels=np.array(All_Labels)
All_Patches=np.array(All_Patches)
print("the count is %d"%count)

print("k is %d"%k)
print("the shape of all_patches")
print(All_Patches.shape)
print("the shape of all_labels")
print(All_Labels.shape)
Height1, Width1, Band1 = All_Patches.shape[1], All_Patches.shape[2], All_Patches.shape[3]
# 4105*5*11*11*220
# 4105*5*1
print(curr_patch.shape)

# for i in range(int(Height*Width/batch_size)):
#     All_Labels[i] = np.array(All_Labels[i])
    # print("the shape is")
    # print(All_Labels[i].shape)
# Al_Labels = convertToOneHot(All_Labels, num_classes=Num_Classes)
# All_Labels=np.array(All_Labels)
# print(All_Labels.shape)
#4105*5*17
print("the shape of All_patchs",file=f)
print(All_Patches.shape,file=f)
print("the shape of All_Labels",file=f)
print(All_Labels.shape,file=f)
Train_Patch,Train_Label,Test_Patch, Test_Label = [],[],[],[]
Train_portition=opt.train_size
Num=count
Num_Train_Classes=int(Train_portition*Num)
print("Num is %d"%Num,file=f)
print("Num_Train is %d"%Num_Train_Classes,file=f)
Test_portition=1-Train_portition
np.random.seed(0)
idx = np.random.choice(Num, Num_Train_Classes, replace=False)
idx_test = np.setdiff1d(range(Num),idx)#求集合的差
Train_Patch = [All_Patches[i] for i in idx]
Train_Label = [All_Labels[i] for i in idx]
Test_Patch = [All_Patches[i] for i in idx_test]
Test_Label = [All_Labels[i] for i in idx_test]
Train_Label=np.array(Train_Label)
Train_Patch=np.array(Train_Patch)
Test_Label=np.array(Test_Label)
Test_Patch=np.array(Test_Patch)
y_test=Test_Label.ravel()
y_train=Train_Label.ravel()
# y_test=np.reshape(Test_Label,(-1,1))
X_test=np.reshape(Test_Patch,(-1,Height1*Width1*Band1))
# y_train=np.reshape(Train_Label,(-1,1))
X_train=np.reshape(Train_Patch,(-1,Height1*Width1*Band1))
print("the shape of test_label",file=f)
print(Test_Label.shape,file=f)
print("the shape of test_patch",file=f)
print(Test_Patch.shape,file=f)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
lr=LogisticRegression()
sgdc=SGDClassifier()
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)
# y_predict=lr.predict(X)
sgdc.fit(X_train,y_train)
sgdc_y_predict=sgdc.predict(X_test)

print('Auccary of LR classifier:',lr.score(X_test,y_test),file=f)
print(classification_report(y_test,lr_y_predict),file=f)
print('Auccary of SGD classifier:',sgdc.score(X_test,y_test),file=f)
print(classification_report(y_test,sgdc_y_predict),file=f)

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
clf=SVC(kernel='rbf',gamma=0.125,C=10)
clf.fit(X_train,y_train)
pre=clf.predict(X_test)
# y_predict=clf.predict(X)
print('Auccary of RBF SVC :',clf.score(X_test,y_test),file=f)
print(classification_report(y_test,pre),file=f)
lsvc=LinearSVC()
lsvc.fit(X_train,y_train)
lsvc_y_predict=lsvc.predict(X_test)
print('Auccary of Linear SVC :',lsvc.score(X_test,y_test),file=f)
print(classification_report(y_test,lsvc_y_predict),file=f)

from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()
knc.fit(X_train,y_train)
knc_y_predict=knc.predict(X_test)
print('Auccary of KNeighborsClassifier is :',knc.score(X_test,y_test),file=f)
print(classification_report(y_test,knc_y_predict),file=f)
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_predict=dtc.predict(X_test)
print('Auccary of DecisionTree is :',dtc.score(X_test,y_test),file=f)
print(classification_report(y_test,dtc_y_predict),file=f)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_predict=rfc.predict(X_test)
# # y_predict=rfc.predict(X)
print('Auccary of RandomForest is :',rfc.score(X_test,y_test),file=f)
print(classification_report(y_test,rfc_y_predict),file=f)
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_predict=gbc.predict(X_test)
#
print('Auccary of GradientBoosting is :',gbc.score(X_test,y_test),file=f)
print(classification_report(y_test,gbc_y_predict),file=f)
f.close()
