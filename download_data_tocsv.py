# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import tushare as ts

# given a stock code, generate ths k data since IPO, save it to local csv file
# input : csvdir -- Where the csvfile will be saved
# iput : stype -- 1 use ts.get_hist_data , else use ts.get_k_data 
def gen_csv_file(code,csvdir,stype=1,startd='IPO',end = '2017-06-01'):
    df1 = ts.get_stock_basics()
    if startd == 'IPO' :
        s = str(df1.ix[code]['timeToMarket'])
        startd = "%s-%s-%s" %(s[0:4],s[4:6],s[6:8])
        
    if stype == 1 :
        kdata = ts.get_hist_data(code,start=startd,end=end)
    else :
        kdata = ts.get_k_data(code,start=startd,end=end)
        
    csvfile = "%s\%s.csv" %(csvdir,code)
    kdata.to_csv(csvfile)

def load_csv_file(code,csvdir):
    csvfile = "%s\%s.csv" %(csvdir,code)
    kdata = pd.read_csv(csvfile)
    return kdata


def download_hs300_tocsv(csvdir,stype=1,startd='IPO',enddate='2017-12-30'):
    hs300 = ts.get_hs300s()
    stocklist=hs300.code
    i = 0
    for code in stocklist :
        i = i + 1
        print ("processing number %s : %s" %(i,code))
        gen_csv_file(code,csvdir,stype=stype,startd=startd,end=enddate)    

#%%    
# Below is example of how to generate the csv files
# When stype = 1, using get_hist_data API, otherwise, using get_k_data API
#csvdir = "c:\pysource\\Kcsvfiles_future"
#download_hs300_tocsv(csvdir,2,startd='2017-06-02',enddate='2017-11-11')

## Below is example of how to load the csv files into one dictionary    
#dictmap = {}
#codelist = ts.get_hs300s().code
#for code in codelist:
#    d = load_csv_file(code,csvdir)
#    dictmap[code] = d
    