# AIstock
尝试运用深度学习的方法理解A股走势
DL_xxx.py 用于训练, DL_xxx_binary , 二分法归类，预测涨跌。  DL_xxx_category.py 更详细的分类，用于预测涨跌区间。
generate_training_data.py  用于生成训练数据
load_data.py  -- 由于tushare API 不是很稳定，将数据下载并保存至本地csv文件
predict_stock.py -- 单只股票的预测
predict_batch_hs300.py  -- 批量沪深300 股票预测

目前存在的问题： 存在严重过拟合的情况，另外这个模型的输入特征数太少。
loss, accuracy = model.evaluate(X_test, Y_test)
4512/5000 [==========================>...] - ETA: 0s

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

test loss:  0.542101167583

test accuracy:  0.766   <== 验证集准确率

Evaluate with real data ------------
2304/2900 [======================>.......] - ETA: 0s
test loss:  0.462760309104

test accuracy:  0.835517241297  <== 跑在测试数据的结果
上涨预测正确百分比 3.29%      <=== recall 很低,不忍直视。
下跌预测正确百分比 97.37%
总共 2900, 上涨 426 0.146897, 下跌 2474 0.853103

模型还有很大的改进可能性，欢迎有兴趣的同道一起研究。

关于作者 :
高峰，现就职于SAP中国研究院，数据库测试专家，对机器学习应用感兴趣。
Email ： geoffrey314@hotmail.com
https://www.linkedin.com/in/feng-gao-2028a390/
