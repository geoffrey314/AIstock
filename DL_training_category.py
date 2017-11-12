# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:23:12 2017

@author: I062460
"""

from generate_training_data import sampling_history_data
from generate_training_data import unison_shuffled_copies
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout
from keras.optimizers import RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from sklearn import preprocessing
from keras import regularizers
import numpy as np
import random
from sklearn import preprocessing


twindow = 50
npredict = 5
# Total number of training data should be 300 * 400 = 120000 rows.
ntimes = 400
codelist,dictmap,slist,plist = \
sampling_history_data(twindow=twindow,npredict = npredict,ntimes=ntimes,retry = 15)
features = ['open','close','high','low','volume']
# Now begin to train data
nlen = len(slist)
relen = twindow * len(features)
xraw = np.array(slist).reshape((nlen,relen))
# !!!!尝试保留个股差异，不做scale
#xarray = preprocessing.scale(xraw)
xarray = xraw
yarray = np.array(plist)

X, Y = unison_shuffled_copies(xarray,yarray)
cutpoint = 110000
X_train, y_train = X[:cutpoint],Y[:cutpoint]
X_test,y_test = X[cutpoint:],Y[cutpoint:]

Y_train = np_utils.to_categorical(y_train, num_classes=7)
Y_test = np_utils.to_categorical(y_test, num_classes=7)


# to build the CNN model
#input_shape = (50,5,1)
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
# inputshape = nday_window * feature_columns
input_shape = relen
model = Sequential()
#model.add(Dense(units=256,input_dim=input_shape, kernel_regularizer=regularizers.l2(0.05)))
model.add(Dense(units=256,input_dim=input_shape))
model.add(Activation('tanh'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dropout(0.75))
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(BatchNormalization())
model.add(Dense(7))
model.add(Activation('softmax'))


rmsprop = RMSprop(lr=0.001,rho=0.09,epsilon=1e-8,decay=0.01)

model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])


print('Training ------------')
# Another way to train the model
model.fit(X_train, Y_train, epochs=1500, batch_size=2048)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, Y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)        

model.save('C:\pysource\glodeneye\AIstock\learning_category.h5')    