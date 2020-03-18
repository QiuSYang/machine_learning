#-*- coding: utf-8 -*-
#主成分分析 降维
import pandas as pd

#参数初始化
inputfile = './datasets/principal_component.xls'
outputfile = './datasets/dimention_reducted.xls' #降维后的数据

data = pd.read_excel(inputfile, header = None) #读入数据

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(data)
print(pca.components_) #返回模型的各个特征向量
print(pca.explained_variance_ratio_) #返回各个成分各自的方差百分比
