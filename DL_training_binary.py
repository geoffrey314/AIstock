# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:23:12 2017

@author: I062460
"""
from generate_training_data import generate_rand_startdate
from generate_training_data import generate_kdata
from generate_training_data import sampling_history_data, sampling_future_data
from generate_training_data import unison_shuffled_copies
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras import regularizers
import numpy as np
import random
from sklearn import preprocessing

twindow = 30
npredict = 3
# Total number of training data should be 300 * ntimes  rows.
ntimes = 300
codelist,dictmap,slist,plist = \
sampling_history_data(twindow=twindow,npredict = npredict,ntimes=ntimes,retry = 15)
features = ['open','close','high','low','volume']
nlen = len(slist)
relen = twindow * len(features)
xraw = np.array(slist).reshape((nlen,relen))
# !!!!尝试保留个股差异，不做scale
#xarray = preprocessing.scale(xraw)
xarray = xraw
yarray = np.array(plist)

cutpoint = 85000
X, Y = unison_shuffled_copies(xarray,yarray)
X_train, y_train = X[:cutpoint],Y[:cutpoint]
X_test,y_test = X[cutpoint:],Y[cutpoint:]

Y_train = y_train
Y_test = y_test

#input_shape = (50,6,1)
# to build the CNN model
#model = Sequential()
#model.add(Conv2D(32, (2, 2), input_shape=input_shape))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(32, (2, 2)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(64, (2, 2)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Flatten())
#model.add(Dense(64))
#model.add(Activation('relu'))
#model.add(Dropout(0.8))
#model.add(Dense(64))
#model.add(Activation('softmax'))

# 尝试全连接层
input_shape = relen
model = Sequential()
#model.add(Dense(units=256,input_dim=input_shape, kernel_regularizer=regularizers.l2(0.05)))
model.add(Dense(units=256,input_dim=input_shape, kernel_regularizer=regularizers.l2(0.02)))
model.add(Activation('tanh'))
model.add(Dense(256))
model.add(Activation('relu'))
#model.add(Dropout(0.9))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('selu'))
#model.add(BatchNormalization())
model.add(Dense(1))
model.add(Activation('sigmoid'))

rmsprop = RMSprop(lr=0.001,rho=0.09,epsilon=1e-8,decay=0.01)

#model.compile(loss='categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])

model.compile(loss='binary_crossentropy',optimizer=rmsprop,metrics=['accuracy'])
print('Training ------------')
# Another way to train the model
model.fit(X_train, Y_train, epochs=20, batch_size=2048)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, Y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

model.save('C:\pysource\glodeneye\AIstock\learning_binary.h5') 

#%%
def compare(ypred, yfuture):

    count0 = 0
    count1 = 0
    correct0 = 0
    correct1 = 0
    
    i = 0
    
    for val in yfuture :
        if val == 0 :
            count0 = count0 + 1
        if val == 1 :
            count1 = count1 + 1
            
        yp = ypred[i]
        if yp <= 0.5 :
            yp = 0
        else :
            yp = 1
        i = i + 1
        
        if ( yp == val) and ( val == 1) :
            correct1 = correct1 + 1
        if ( yp == val) and ( val == 0) :
            correct0 = correct0 + 1
            
    print ("上涨预测正确百分比 %.2f%%" %(correct1/count1*100))
    print ("下跌预测正确百分比 %.2f%%" %(correct0/count0*100))   

print("================predict with test data================")
twindow = 30
npredict = 3
# Total number of Test data should be 300 * ntimes  rows.
ntimes = 10
codelist,dictmap,slist,plist = \
sampling_future_data(twindow=twindow,npredict = npredict,ntimes=ntimes,retry = 15)
features = ['open','close','high','low','volume']
nlen = len(slist)
relen = twindow * len(features)
xraw = np.array(slist).reshape((nlen,relen))
# !!!!尝试保留个股差异，不做scale
#xarray = preprocessing.scale(xraw)
xfuture = xraw
yfuture = np.array(plist)
print('\nEvaluate with real data ------------')
loss, accuracy = model.evaluate(xfuture, yfuture)
ypredict = model.predict(xfuture)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
compare(ypredict,yfuture)   