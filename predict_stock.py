# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:04:25 2017

@author: I062460
"""

import numpy as np
from keras.models import load_model
import tushare as ts
from sklearn import preprocessing
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def plot_single_report (stockinfo, y_pred,yc):

    matplotlib.rcdefaults()
    p = matplotlib.rcParams    
    p["font.family"] = "sans-serif"
    p["font.sans-serif"] = ["SimHei", "Tahoma"]
    zhfont1 = matplotlib.font_manager.FontProperties(fname='c:/Windows/Fonts/simsun.ttc')
    p["font.size"] = 10
    p["axes.unicode_minus"] = False
    stockname = stockinfo.name.values[0]
    code = stockinfo.code.values[0]
    plt.suptitle("%s %s 涨跌概率 : %f" %(stockname,code,y_pred), fontsize=10, color="r",fontproperties=zhfont1)
    x = [-2,-1,0,1,2,3]
    y = yc[1:]
    plt.bar(x,y,width=0.4)
    plt.show()


model1 = load_model('C:\pysource\glodeneye\AIstock\learning_binary.h5')
model2 = load_model('C:\pysource\glodeneye\AIstock\learning_category.h5')


hs300 = pd.read_csv('C:\pysource\glodeneye\hs300.csv',dtype=str)
twindow = 50
input_shape = twindow*5
startd = '2017-06-01'

while (True):
    code = input("请输入要预测的股票代码 :")
    if ( code == "quit" ):
        break 
    stockinfo = hs300[hs300.code == code ]
    print ("股票基本信息 :\n") 
    print (stockinfo.as_matrix() )
    print ("========================================")
    x = []
    kdata = ts.get_k_data(code,start=startd,end ='2017-11-11').tail(twindow)
    if len(kdata) != twindow :
        print ("Warning, didn't get %d length data of %s\n" %(twindow,code))
    k = kdata.as_matrix(columns = ['open','close','high','low','volume'] )
    x.append( preprocessing.scale(k) )
    x_input = np.array(x).reshape((1,input_shape))
    y_pred = model1.predict(x_input)
    yc = model2.predict(x_input)[0]
#    print ("0代表5日后涨幅 < 1.5%, 1代表5日后涨幅 > 1.5% \n计算概率: ", y_pred[0])
#    print ("========================================")
#    print ("以下数据仅供参考, 预测5日之后涨幅， 61%正确性 \n")
#    print ("<-10%% %.2f%% , -10%% ~ -5%% %.2f%% " %(yc[0]*100,yc[1]*100) )
#    print ("-5%% ~ 0%% %.2f%% , 0%% ~ 5%% %.2f%% " %(yc[2]*100,yc[3]*100) )
#    print ("5%% ~ 10%% %.2f%% , >10%% %.2f%% " %(yc[4]*100,yc[5]*100) )
    plot_single_report(stockinfo,y_pred,yc)