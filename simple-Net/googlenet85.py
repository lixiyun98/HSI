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
import numpy as np
import tensorflow.contrib.layers
from sklearn.decomposition import PCA
f=open("googlenet85_india_pines.txt","w")
# f=open("test15_india_pines.txt","w")
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines')
parser.add_argument('--patch_size', type=int, default=11)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--train_size', type=float, default=0.7)
# parser.add_argument('--train_size', type=float, default=0.08)
parser.add_argument('--isnorm', type=int, default=1)
parser.add_argument('--training_iters', type=int, default=10000)
# parser.add_argument('--training_iters', type=int, default=10)
opt = parser.parse_args()
# f=open("out1.txt","w")
DATA_PATH = os.path.join(os.getcwd(),"Data")
Data = scipy.io.loadmat('./Data/' + opt.data + '.mat')[opt.data.lower()]
Label = scipy.io.loadmat('./Data/' + opt.data + '_gt.mat')[opt.data.lower() + '_gt']
# dataval=np.mean(Data,axis=2)
# newdata=Data-dataval
# covdata=np.cov(newdata)
Height, Width, Band = Data.shape[0], Data.shape[1], Data.shape[2]
Data=Data.reshape((Height*Width,Band))
channel=69
estimator=PCA(n_components=channel)
Band=channel
Data=estimator.fit_transform(Data)
print(Data.shape)
Data=Data.reshape((Height,Width,Band))
Num_Classes = len(np.unique(Label))
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
def Patch(height_index,width_index):
    """ function to extract patches from the orignal data """
    # transpose_array = np.transpose(Data_Padding,(2,0,1))
    height_slice = slice(height_index, height_index + patch_size)
    width_slice = slice(width_index, width_index + patch_size)
    patch = Data_Padding[height_slice, width_slice,:]
    return np.array(patch)
All_Patches, All_Labels ,Al_Labels,Num = [],[],[],[]
count=0
k=0
res=0
# for i in range(Num_Classes):
#     All_Patches.append([])
#     Num.append([0])
# for i in range(int(Height*Width/batch_size)):
#     All_Patches.append([])
#     All_Labels.append([])
#     Al_Labels.append([])
# for j in range(0,Width):
#     for i in range(0,Height):
#         curr_patch=Patch(i,j)
#         All_Patches[Label[i,j]].append(curr_patch)
#         # Label[i,j]=np.array(Label[i,j])
#         # Labels = convertToOneHot(Label[i,j], num_classes=Num_Classes)
#         All_Labels.append(Label[i, j])
#         Num[Label[i,j]]+=1
#         count = count + 1
        # if count % batch_size == 0:
        #     k = k + 1
        # if Label[i,j]!=0:

for j in range(0,Width):
    for i in range(0,Height):
        curr_patch=Patch(i,j)
        All_Patches.append(curr_patch)
        # Label[i,j]=np.array(Label[i,j])
        # Labels = convertToOneHot(Label[i,j], num_classes=Num_Classes)
        All_Labels.append(Label[i, j])
        count = count + 1

All_Labels=np.array(All_Labels)
All_Patches=np.array(All_Patches)
print("the count is %d"%count)
print("k is %d"%k)
print(All_Patches.shape)
Height1, Width1, Band1 = All_Patches.shape[1], All_Patches.shape[2], All_Patches.shape[3]
print(curr_patch.shape)
# for i in range(int(Height*Width/batch_size)):
#     All_Labels[i] = np.array(All_Labels[i])
All_Labels = convertToOneHot(All_Labels, num_classes=Num_Classes)
All_Labels=np.array(All_Labels)
#4105*5*17
print("the shape of All_patchs")
print(All_Patches.shape)
print("the shape of All_Labels")
print(All_Labels.shape)
Train_Patch,Train_Label,Test_Patch, Test_Label = [],[],[],[]
Train_portition=opt.train_size
Num=count
Num_Train_Classes=int(Train_portition*Num)
print("Num is %d"%Num)
print("Num_Train is %d"%Num_Train_Classes)
Test_portition=1-Train_portition
np.random.seed(0)
# for i in range(Num_Classes):
#     if Train_portition*Num[i]>10:
#         idx = np.random.choice(Num[i], int(Num[i]*Train_portition), replace=False)
#         Train_Patch .append([All_Patches[i][j] for j in idx])
#         Train_Label.append([i]* int(Num[i]*Train_portition))
#     else:
#         idx = np.random.choice(Num[i], int(Num[i] * 0.5), replace=False)
#         Train_Patch.append([All_Patches[i][j] for j in idx])
#         Train_Label.append([i] * int(Num[i] * 0.5))
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
print("the shape of test_label")
print(Test_Label.shape)
print("the shape of test_patch")
print(Test_Patch.shape)
Test_Label=np.reshape(Test_Label,(-1,Num_Classes))
Test_Patch=np.reshape(Test_Patch,(-1,Height1,Width1,Band1))
x=tf.placeholder(tf.float32, [None,Height1,Width1,Band1])
y=tf.placeholder(tf.float32, [None,Num_Classes])
# print("%d %d %d"%(Height,Width,Band))
def batch_norm(inputs,is_training,is_conv_out=True,decay=0.999):
    scale=tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean=tf.Variable(tf.zeros([inputs.get_shape()[-1]]),trainable=False)
    pop_var=tf.Variable(tf.ones([inputs.get_shape()[-1]]),trainable=False)
    if is_training:
        if is_conv_out:
            batch_mean,batch_var=tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean=tf.assign(pop_mean,pop_mean*decay+batch_mean*(1-decay))
        train_var=tf.assign(pop_var,pop_var*decay+batch_var*(1-decay))
        with tf.control_dependencies([train_mean,train_var]):
            return tf.nn.batch_normalization(inputs,batch_mean,batch_var,beta,scale,0.001)
    else:
        return tf.nn.batch_normalization(inputs,pop_mean,pop_var,beta,scale,0.001)

def weight_variable(shape):
    initial=tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial =tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def con2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")
    # return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

size=11

W_conv1=weight_variable([3,3,Band1,32])
tf.add_to_collection(tf.GraphKeys.WEIGHTS,W_conv1)
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(con2d(x,W_conv1)+b_conv1)
# h_conv1=tf.nn.relu(tf.layers.BatchNormalization(trainable=True)(con2d(x,W_conv1)+b_conv1))
# conv1=con2d(x,W_conv1)+b_conv1
# conv1=batch_norm(conv1,True)
# h_conv1=tf.nn.relu(conv1)

# h_conv1=max_pool(h_conv1)
W_conv2=weight_variable([1,1,32,64])
tf.add_to_collection(tf.GraphKeys.WEIGHTS,W_conv2)
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(con2d(h_conv1,W_conv2)+b_conv2)
# h_conv2=tf.nn.relu(tf.layers.BatchNormalization(trainable=True)(con2d(h_conv1,W_conv2)+b_conv2))

# conv2=con2d(h_conv1,W_conv2)+b_conv2
# conv2=batch_norm(conv2,True)
# h_conv2=tf.nn.relu(conv2)

# h_conv1=max_pool(h_conv1)
W_conv3=weight_variable([3,3,64,64])
tf.add_to_collection(tf.GraphKeys.WEIGHTS,W_conv3)
b_conv3=bias_variable([64])
h_conv3=tf.nn.relu(con2d(h_conv2,W_conv3)+b_conv3)
# h_conv3=tf.nn.relu(tf.layers.BatchNormalization(trainable=True)(con2d(h_conv2,W_conv3)+b_conv3))

#inception block
W_conv4=weight_variable([1,1,64,64])
tf.add_to_collection(tf.GraphKeys.WEIGHTS,W_conv4)
b_conv4=bias_variable([64])
h_conv4=tf.nn.relu(con2d(h_conv3,W_conv4)+b_conv4)
# h_conv4=tf.nn.relu(tf.layers.BatchNormalization(trainable=True)(con2d(h_conv3,W_conv4)+b_conv4))

W_conv5=weight_variable([1,1,64,64])
tf.add_to_collection(tf.GraphKeys.WEIGHTS,W_conv5)
b_conv5=bias_variable([64])
h_conv5=tf.nn.relu(con2d(h_conv3,W_conv5)+b_conv5)
# h_conv5=tf.nn.relu(tf.layers.BatchNormalization(trainable=True)(con2d(h_conv3,W_conv5)+b_conv5))

W_conv6=weight_variable([3,3,64,64])
tf.add_to_collection(tf.GraphKeys.WEIGHTS,W_conv6)
b_conv6=bias_variable([64])
h_conv6=tf.nn.relu(con2d(h_conv5,W_conv6)+b_conv6)
# h_conv6=tf.nn.relu(tf.layers.BatchNormalization(trainable=True)(con2d(h_conv5,W_conv6)+b_conv6))

W_conv7=weight_variable([1,1,64,64])
tf.add_to_collection(tf.GraphKeys.WEIGHTS,W_conv7)
b_conv7=bias_variable([64])
h_conv7=tf.nn.relu(con2d(h_conv3,W_conv7)+b_conv7)
# h_conv7=tf.nn.relu(tf.layers.BatchNormalization(trainable=True)(con2d(h_conv3,W_conv7)+b_conv7))
W_conv8=weight_variable([3,3,64,64])
tf.add_to_collection(tf.GraphKeys.WEIGHTS,W_conv8)
b_conv8=bias_variable([64])
h_conv8=tf.nn.relu(con2d(h_conv7,W_conv8)+b_conv8)
# h_conv8=tf.nn.relu(tf.layers.BatchNormalization(trainable=True)(con2d(h_conv7,W_conv8)+b_conv8))
W_conv9=weight_variable([3,3,64,64])
tf.add_to_collection(tf.GraphKeys.WEIGHTS,W_conv9)
b_conv9=bias_variable([64])
h_conv9=tf.nn.relu(con2d(h_conv8,W_conv9)+b_conv9)
# h_conv9=tf.nn.relu(tf.layers.BatchNormalization(trainable=True)(con2d(h_conv8,W_conv9)+b_conv9))
output4=tf.concat([h_conv4,h_conv6,h_conv9],3)
# h_conv2=max_pool(h_conv2)
W_fc1=weight_variable([size*size*192,128])
b_fc1=bias_variable([128])
h_flat=tf.reshape(output4,[-1,size*size*192])
h_fc1=tf.nn.relu(tf.matmul(h_flat,W_fc1)+b_fc1)
W_fc2=weight_variable([128,Num_Classes])
b_fc2=bias_variable([Num_Classes])
h_fc2=tf.nn.relu(tf.matmul(h_fc1,W_fc2)+b_fc2)
# scale=0.1
# regularizer=tf.contrib.layers.l2_regularizer(scale)
# reg_term=tf.contrib.layers.apply_regularization(regularizer)
# cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=h_fc2))
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=h_fc2))
optimizer=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(h_fc2,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()
# sess=tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
count=0
print("the shape of train_label")
print(Train_Label.shape)
print("the shape of train_patch")
print(Train_Patch.shape)
Test_Label=np.reshape(Test_Label,(-1,Num_Classes))
Test_Patch=np.reshape(Test_Patch,(-1,Height1,Width1,Band1))
Train_Label=np.reshape(Train_Label,(-1,Num_Classes))
Train_Patch=np.reshape(Train_Patch,(-1,Height1,Width1,Band1))
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for iteration in range(training_iters):
        idx = np.random.choice(Num_Train_Classes, size=batch_size, replace=False)
        # Use the random index to select random images and labels.
        # indian_pines
        batch_x = Train_Patch
        batch_y = Train_Label
        # pavia
        # batch_x = Train_Patch[idx, :]
        # batch_y = Train_Label[idx, :]
        # Run optimization op (backprop) and cost op (to get loss value)
        _, batch_cost, train_acc = sess.run([optimizer, cross_entropy, accuracy],feed_dict={x: batch_x,y: batch_y})
        # Display logs per epoch step
        if iteration % 100 == 0:
            print("Iteraion", '%04d,' % (iteration), \
            "Batch cost=%.4f," % (batch_cost),\
            "Training Accuracy=%.4f" % (train_acc),file=f)
        if iteration % 1000 ==0:
            print('Training Data Eval: Training Accuracy = %.4f' % sess.run(accuracy,\
                feed_dict={x: Train_Patch,y: Train_Label}),file=f)
            print('Test Data Eval: Test Accuracy = %.4f' % sess.run(accuracy,\
                feed_dict={x: Test_Patch,y: Test_Label}),file=f)
    print("Optimization Finished!",file=f)

# for i in range(len(Train_Patch)):
#     train_accuracy = accuracy.eval(feed_dict={x: Train_Patch[i], y: Train_Label[i]})
#     count += train_accuracy * batch_size
#     print("step %d,training accuracy %g" % (i,train_accuracy))
#     train_step.run(feed_dict={x: Train_Patch[i], y: Train_Label[i]})
# print(count)
# #     # All_Patches[i] = np.reshape(All_Patches[i], (1, Height, Width, Band))
# #     # print(All_Patches[i].shape)
# #     if i %100 ==0:
# print("test accuracy: %g"%accuracy.eval(feed_dict={x:Test_Patch,y:Test_Label}))


# print(Data.astype)
# Data = Data.astype(float)
# print("%d"%(Num_Classes))
# print(Data.shape)
# print(Data[20,1,1])
f.close()