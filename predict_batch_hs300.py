# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 19:37:35 2017

@author: GAO
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:04:25 2017

@author: I062460
"""
import datetime
import numpy as np
from keras.models import load_model
import tushare as ts
from sklearn import preprocessing
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from generate_training_data import binary_y    

def predict_hs300_5days(hs300,model1,model2,twindow,input_shape,start,end) :
    hs300_codes = hs300.code
    x = []
    namelist = []
    errstock_list = []
    codelist = []
    if end == 'TODAY' :
        end = datetime.datetime.now().strftime('%Y-%m-%d')
    for code in hs300_codes:
        stockinfo = hs300[hs300.code == code ]
        name = hs300.ix[hs300['code'] == code]['name'].values[0]
        print ("股票基本信息 :\n") 
        print (stockinfo.as_matrix() )
        print ("========================================")
        kdata = ts.get_k_data(code,start=startd,end = end).tail(twindow)
        if len(kdata) != twindow :
            print ("停牌股票或者股票数据异常 : %s" %(name))
            errstock_list.append(name)
            continue
        namelist.append(name)
        k = kdata.as_matrix(columns = ['open','close','high','low','volume'] )
        x.append( preprocessing.scale(k) )
        codelist.append(code)
        
    nlen = len(codelist)
    codearr = np.array(codelist).reshape(nlen,1)
    namearr = np.array(namelist).reshape(nlen,1)
    
    x_raw = np.array(x).reshape((nlen,input_shape))
    # 模型更新为不做二次scale
    #x_input = preprocessing.scale(x_raw)
    x_input = x_raw
    # Now begin to predict the stock info.
    y_pred = model1.predict(x_input)
    yc_pred  = model2.predict(x_input)
    
    com = np.hstack((namearr,codearr,y_pred,yc_pred))
    df = pd.DataFrame(com,\
        columns=['name','code','y','d0','d1','d2','d3','d4','d5','d6'])
    df = df.astype(dtype = {"name":"object","code":"object","y":"float32",\
                            "d0":"float32","d1":"float32","d2":"float32",\
                            "d3":"float32","d4":"float32","d5":"float32","d6":"float32",})
    df_recommend = df [ df['y']>0.8 ] 
    df_sort = df_recommend.sort_values(by='y',ascending=False)
    df_sort_full = df.sort_values(by='y',ascending=False)
    return codelist,df_sort, df_sort_full

def plot_sort_report (df_sort,jpg_file):
    zhfont1 = matplotlib.font_manager.FontProperties(fname='c:/Windows/Fonts/simsun.ttc')
    matplotlib.rcdefaults()
    p = matplotlib.rcParams
    p["font.family"] = "sans-serif"
    p["font.sans-serif"] = ["SimHei", "Tahoma"]
    p["font.size"] = 10
    p["axes.unicode_minus"] = False
    p["figure.figsize"] = (6,100)
    figure = plt.figure()
    figure.subplots_adjust(hspace=0.4)
    figure.suptitle("5日预测报告",fontproperties=zhfont1)
    dfmatrix = list(df_sort.as_matrix())
    x = [-2,-1,0,1,2,3]
    count = 1
    nlen = len(dfmatrix)
    for row in dfmatrix:
        stockname = row[0]
        code = row[1]
        p = row[2]
        y = row[4:]
        sp = figure.add_subplot(nlen,1,count)
        sp.set_title("%s %s 涨跌概率 : %f" %(stockname,code,p), fontsize=10, color="b",fontproperties=zhfont1)
        sp.bar(x,y,width=0.4)
        count = count+1     
    figure.savefig(jpg_file)
    plt.close()

def get_real_5days_change (codelist, startd, endd) :
    dictmap = {}
    for code in codelist :
        key = str(code)
        print ("Getting code %s" %(code))
        kstart = ts.get_k_data(code,start=startd,end=endd)
        kend = ts.get_hist_data(code,start=startd,end=endd)
        if kstart.empty or kend.empty :
            dictmap[key]= -1
            continue
        close_begin = kstart['close'].values[0]
        close_end = kend['close'].values[0]
        pchange = (close_end - close_begin) / close_begin
        y = binary_y(pchange)        
        dictmap[key] = y
    return dictmap        
#%%
# Execution code

model1 = load_model('C:\pysource\glodeneye\AIstock\learning_binary.h5')
model2 = load_model('C:\pysource\glodeneye\AIstock\learning_category.h5')


hs300 = pd.read_csv('C:\pysource\glodeneye\hs300.csv',dtype=str)
# twindow is how many days it look back for history trade data
twindow = 50

# x feature list is : 'open','close','high','low','volume'
input_shape = twindow*5
startd = '2017-05-01'
timestr = datetime.datetime.now().strftime('%Y%m%d')
report_file =  "C:/pysource/glodeneye/AIstock/report_%s.png" %(timestr)    
codelist,df_sort,df_sort_full = predict_hs300_5days(hs300,model1,model2,twindow,input_shape,startd,end='2017-10-25')
plot_sort_report(df_sort,report_file)    


ydict = get_real_5days_change(codelist,startd = '2017-10-27',endd='2017-11-03')
df_compare = df_sort_full.as_matrix(columns = ['code','name','y']).tolist()

count = 0
count1 = 0
count0 = 0
c1 = 0
c0 = 0

for row in df_compare :
    rcode = row[0]
    rname = row[1]
    rpredict = row[2]
    if rpredict > 0.5 :
        ypred = 1
    else :
        ypred = 0
    yreal = ydict[rcode]
    if yreal == 0 :
        count0 = count0 +1
    if yreal == 1 :
        count1 = count1 + 1
        
    print ("rcode is %s, ypredict is %d, yreal is %d" %(rcode,ypred,yreal))
    if ypred == yreal :
        print ("预测正确， %s %s" %(rcode,rname))
        count = count + 1
    
    if (ypred == yreal) and (yreal == 1):
        c1 = c1 + 1
    if (ypred == yreal) and (yreal == 0) :
        c0 = c0 + 1
        
print ("正确百分比 %.2f%%" %(count/len(df_compare)*100))
print ("上涨预测正确百分比 %.2f%%" %(c1/count1*100))
print ("下跌预测正确百分比 %.2f%%" %(c0/count0*100))      