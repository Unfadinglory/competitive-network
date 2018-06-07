# -*- coding: utf-8 -*-
"""
Created on Tue Jun 5 16:59:40 2017
A test for c_network
@author: Kaku U
"""
import numpy as np
from competitive_network import *
import matplotlib.pyplot as plt

def load_data(file_path):#路径自行定义
    """导入数据
    input: file_path(string):文件路径
    output: data(mat):数据
    """
    f=open(file_path)#导入文件
    data=[]#定义数据集
    for line in f.readlines():#读取每一行的字符串
        data_tmp=[]#定义每一行的数据
        lines=line.strip().split()#去除头尾的空字符并按制表符分成字符数组
        for x in lines:
            data_tmp.append(float(x.strip()))#将每一行的字符串一个一个导入行变量
        data.append(data_tmp)#将行变量一个一个导入数据集
    f.close()#关闭文件
    return np.asarray(data)#返回数据集的矩阵
#测试
if __name__=="__main__":
    
    dataMat=load_data("D:\Anaconda3\work\Competitive Network\data_1_inputs_w_pq.txt")
    weights=load_data("D:\Anaconda3\work\Competitive Network\data_1_inputs_w_w.txt")
    sample=load_data("D:\Anaconda3\work\Competitive Network\data_1_inputs_w_s.txt")
    print('dataMat=',np.array(normalization_all(dataMat)))
    assert (dataMat.shape==(6,8))
    print('weights=',weights)
    assert (weights.shape==(6,8))
    print('sample=',sample)
    assert (sample.shape==(3,8))
    a=0.05
    c=[]

    cn=competitive_network(weights,a)
    cn.train(dataMat,500)
    print(cn.W)

    for i in range(len(sample)):
        prediction=cn.prediction(sample[i])
        c.append(prediction)

    print('c=',c)