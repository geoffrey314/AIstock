# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:15:10 2017

@author: I062460
"""

import tushare as ts
import os
import numpy as np
import pandas as pd
from keras.utils import np_utils

def regular_y ( y_value ):
    if ( y_value < -0.1 ) :
        return 1
    elif ( (y_value >= -0.1) and (y_value < -0.05)) :
        return 2
    elif ( (y_value >= -0.05) and (y_value <0)) :
        return 3
    elif ( (y_value >=0) and (y_value) < 0.05 ) :
        return 4
    else:
        return 5
    
def load_data (scode, nday):
    # Due to SAP internet , no need to use proxy
    #os.environ['HTTPS_PROXY'] = ''
    #os.environ['HTTP_PROXY'] = ''
    #del os.environ['HTTPS_PROXY']
    #del os.environ['HTTP_PROXY']
    
    df = ts.get_hist_data(scode)
    df['y_label'] = 0.0
    number = len(df) - 1
    # go through each row to calculate the y_lavel base on nday price
    for i in range(number,nday,-1):
        current_open_price = df.ix[i,'open']
        j = i - nday
        nday_close = df.ix[j,'close']
        price_percent = (nday_close - current_open_price) / current_open_price
        df.ix[i,'y_label'] = regular_y(price_percent)
        #df.ix[i,'y_label'] = price_percent
    return df[i:number]
        

def ticks_to_columns (scode, n, topN) :
    
    S = load_data(scode , 3) 
    temp = S.iloc[0:n,:]

    result = []
    datelist = []
    newlen = n
    for idxdate in temp.index :
        print("date is ",idxdate)
        T = ts.get_tick_data(scode, date = idxdate)
        T.dropna(axis=0,how='any')
        if ( len(T) < topN):
            newlen = newlen - 1 
            continue
        T_head = T.sort_values(by='volume',ascending=False).head(topN)
        # normalize the value of the type
        T_head['type'] = T_head['type'].apply(lambda x:np.where(x=='买盘',1,x))
        T_head['type'] = T_head['type'].apply(lambda x:np.where(x=='卖盘',-1,x))
        T_head['type'] = T_head['type'].apply(lambda x:np.where(x=='中性盘',0,x))
        # the to_categorial will make following scale() failed.
        #T_head['type'] = T_head['type'].apply(lambda x:np_utils.to_categorical(x, 4))
        F = T_head.ix[:,['price','volume','type']].values.flatten()
        y = len(F)        
        print ("len of x is ", y)
        # For each day, append today's ticks
        result.append(F)
        datelist.append(idxdate)
    
    F = np.array(result).reshape(newlen,3*topN)
    D = np.array(datelist).reshape(newlen,1)
    return F,datelist
       

#F, D = ticks_to_columns('002415',600, 50)

#df = pd.DataFrame(F, index = D)
#df.to_csv('hkvs.csv')
   
    
    