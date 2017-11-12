# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:25:27 2017

@author: I062460
"""

from download_data_tocsv import load_csv_file
import tushare as ts
import sys
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing

def regular_y ( y_value ):
    if ( y_value < -0.1 ) :
        return 1
    elif ( (y_value >= -0.1) and (y_value < -0.05)) :
        return 2
    elif ( (y_value >= -0.05) and (y_value <0)) :
        return 3
    elif ( (y_value >=0) and (y_value) < 0.05 ) :
        return 4
    elif ( (y_value >=0.05) and (y_value) < 0.10 ) :
        return 5
    else:
        return 6

def binary_y ( y_value ):
    if y_value <= 0.03 :
        return 0
    else :
        return 1

def generate_kdata (code, dictmap, start, twindow, npredict ):

    # get the DataFrame for specific code
    dfdata = dictmap[code]
    
    rowidx = dfdata[dfdata.date == start].index[0]
    
    row_end = rowidx + twindow - 1
    row_predict = row_end + npredict
    
    price_end = dfdata['close'].loc[row_end]
    price_predict = dfdata['close'].loc[row_predict]
    p_change = (price_predict - price_end) / price_end
    
    dfsampling = dfdata.loc[rowidx:row_end]
    x = len(dfsampling)
    if x != twindow :
        print ("Warning : expect %d length for code %s, but got %d" %(twindow,code,x))
        
    sample_array = dfsampling.as_matrix(columns = ['open','close','high','low','volume'] )   
    
    #p_regular = regular_y(p_change)
    p_regular = binary_y(p_change)            
    return sample_array, p_regular

# Genereate a random valid start date
def generate_rand_startdate(code,dictmap, twindow, npredict ):
    dfdata = dictmap[code]
    totalrow = dfdata.shape[0]
    validstart = totalrow - twindow - npredict - 3
    if ( validstart <= 0 ) :
        print ("Error : can't find valid start row for stock %s" %(code))
        return '9999-01-01'
        #sys.exit(-1)
    
    randstart = random.randint(0, validstart)
    startdate = dfdata['date'].loc[randstart]
    if (startdate == "") or (startdate is None ):
        print ("Error : can't find valid stardate for stock %s" %(code))
        sys.exit(-1)
    return startdate    

def sampling_history_data(twindow=50,npredict=5,ntimes=400,retry=10):
    # Below is example of how to load the csv files into one dictionary    
    dictmap = {}
    codelist = pd.read_csv('C:\pysource\glodeneye\hs300.csv',dtype=str).code
    csvdir = 'c:\pysource\Kcsvfiles06'   
    for code in codelist:
        d = load_csv_file(code,csvdir)
        dictmap[code] = d
        
    slist = []
    plist = []
    
    for code in codelist:
        print ("Loading stock %s" %(code))
        startd_list = []
        for j in range(0,ntimes) :
            # 尽量避免生成重复的数据    
            for k in range(0,retry):
                startd = generate_rand_startdate(code, dictmap, twindow, npredict)
                if startd in startd_list:
                    continue
                else:
                    break
            if ( k == (retry-1) ):
                print ("Hit duplidate start date %s " %(startd))
            print ("Retry %d, length of list is %s" %(k, len(startd_list)))
            startd_list.append(startd)                                    
            x, y = generate_kdata(code, dictmap, startd, twindow, npredict)
            # Normalize and scale the history data
            slist.append( preprocessing.scale(x) )
            plist.append(y)
    return codelist,dictmap,slist,plist

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def sampling_future_data(twindow=50,npredict=5,ntimes=10,retry=10):
    # Below is example of how to load the csv files into one dictionary    
    dictmap = {}
    codelist = pd.read_csv('C:\pysource\glodeneye\hs300.csv',dtype=str).code
    csvdir = 'c:\pysource\Kcsvfiles_future'   
    for code in codelist:
        d = load_csv_file(code,csvdir)
        dictmap[code] = d
        
    slist = []
    plist = []

    for code in codelist:
        print ("Loading stock %s" %(code))
        valid = generate_rand_startdate(code, dictmap, twindow, npredict)
        if valid == '9999-01-01' :
            # This is not a stock, 可能是停牌太长时间或数据不全
            continue
            
        startd_list = []
        for j in range(0,ntimes) :
            # 尽量避免生成重复的数据    
            for k in range(0,retry):
                startd = generate_rand_startdate(code, dictmap, twindow, npredict)
                if startd in startd_list:
                    continue
                else:
                    break
            if ( k == (retry-1) ):
                print ("Hit duplidate start date %s " %(startd))
            print ("Retry %d, length of list is %s" %(k, len(startd_list)))
            startd_list.append(startd)                                    
            x, y = generate_kdata(code, dictmap, startd, twindow, npredict)
            # Normalize and scale the history data
            slist.append( preprocessing.scale(x) )
            plist.append(y)
    return codelist,dictmap,slist,plist