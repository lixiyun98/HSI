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
import keras
from keras.utils import plot_model

def get_model2(Num):
    input_data = keras.Input(shape=(11,11,220))
    conv1=keras.layers.Conv2D(32, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')(input_data)
    conv2=keras.layers.Conv2D(64, (3,3), padding='valid', activation= 'relu', kernel_initializer='he_normal')(conv1)
    flat=keras.layers.Flatten()(conv2)
    hc1=keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(flat)
    output=keras.layers.Dense(Num, activation='softmax', kernel_initializer='he_normal')(hc1)
    model = keras.Model(inputs=input_data, outputs=output)
    sgd=keras.optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
def scheduler(epoch):
    learning_rate_init=0.02
    if epoch>= 80:
        learning_rate_init=0.01
    if epoch>= 140:
        learning_rate_init=0.004
    return  learning_rate_init
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines')
parser.add_argument('--epoches', type=int, default=1000)
parser.add_argument('--patch_size', type=int, default=11)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--train_size', type=float, default=0.8)
parser.add_argument('--isnorm', type=int, default=1)
opt = parser.parse_args()


DATA_PATH = os.path.join(os.getcwd(),"Data")
Data = scipy.io.loadmat('./Data/' + opt.data + '.mat')[opt.data.lower()]
Label = scipy.io.loadmat('./Data/' + opt.data + '_gt.mat')[opt.data.lower() + '_gt']
# Data = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines.mat'))['indian_pines']
# Label = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_gt.mat'))['indian_pines_gt']
Height, Width, Band = Data.shape[0], Data.shape[1], Data.shape[2]
Num_Classes = len(np.unique(Label))
patch_size=opt.patch_size
batch_size=opt.batch_size
epoches=opt.epoches
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

for i in range(int(Height*Width/batch_size)):
    All_Patches.append([])
    All_Labels.append([])
    Al_Labels.append([])
for j in range(0,Width):
    for i in range(0,Height):
        curr_patch=Patch(i,j)
        All_Patches[k].append(curr_patch)
        # Label[i,j]=np.array(Label[i,j])
        # Labels = convertToOneHot(Label[i,j], num_classes=Num_Classes)
        All_Labels[k].append(Label[i,j])
        count = count+1
        if count % batch_size == 0:
            k=k+1
All_Labels=np.array(All_Labels)
All_Patches=np.array(All_Patches)
Height1, Width1, Band1 = All_Patches.shape[2], All_Patches.shape[3], All_Patches.shape[4]
# 4105*5*11*11*220
# 4105*5*1
print(curr_patch.shape)

for i in range(int(Height*Width/batch_size)):
    All_Labels[i] = np.array(All_Labels[i])
    # print("the shape is")
    # print(All_Labels[i].shape)
    Al_Labels[i] = convertToOneHot(All_Labels[i], num_classes=Num_Classes)
Al_Labels=np.array(Al_Labels)

#4105*5*17
print("the shape of All_patchs")
print(All_Patches.shape)
print("the shape of Al_Labels")
print(Al_Labels.shape)
Train_Patch,Train_Label,Test_Patch, Test_Label = [],[],[],[]
Train_portition=opt.train_size
Num=int(Height*Width/batch_size)
Num_Train_Classes=int(Train_portition*Num)
print("Num is %d"%Num)
print("Num_Train is %d"%Num_Train_Classes)
Test_portition=1-Train_portition
np.random.seed(0)
idx = np.random.choice(Num, Num_Train_Classes, replace=False)
idx_test = np.setdiff1d(range(Num),idx)#求集合的差
Train_Patch = [All_Patches[i] for i in idx]
Train_Label = [Al_Labels[i] for i in idx]
Test_Patch = [All_Patches[i] for i in idx_test]
Test_Label = [Al_Labels[i] for i in idx_test]
Train_Label=np.array(Train_Label)
Train_Patch=np.array(Train_Patch)
Test_Label=np.array(Test_Label)
Test_Patch=np.array(Test_Patch)
print("the shape of test_label")
print(Test_Label.shape)
print("the shape of test_patch")
print(Test_Patch.shape)
Test_Label=np.reshape(Test_Label,(-1,Num_Classes))
Test_Patch=np.reshape(Test_Patch,(-1,Height1,Width1,Band1))
Train_Label=np.reshape(Train_Label,(-1,Num_Classes))
Train_Patch=np.reshape(Train_Patch,(-1,Height1,Width1,Band1))

# Height, Width, Band = All_Patches.shape[1], All_Patches.shape[2], All_Patches.shape[3]
# print(All_Patches.shape)
logfilepath = './simplelog'
model = get_model2(17)
print(model.summary())
tb=keras.callbacks.TensorBoard(log_dir=logfilepath,histogram_freq=0)
change_lr=keras.callbacks.LearningRateScheduler(scheduler)
cbtk=[change_lr,tb]
model.fit(Train_Patch,Train_Label,batch_size=batch_size,epochs=epoches,callbacks=cbtk,validation_data=(Test_Patch,Test_Label),shuffle=True)
model.save('simplenet.h5')
# print(Data.astype)
# Data = Data.astype(float)
# print("%d"%(Num_Classes))
# print(Data.shape)
# print(Data[20,1,1])
f.close()