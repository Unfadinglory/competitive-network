# -*- coding: utf-8 -*-
"""
Created on Tue Jun 5 16:59:40 2017
A competitive network for pattern recognition
@author: Kaku U
"""
import numpy as np

def sigmoid(x,derivative=False):#激活函数
    return 1/(1+np.exp(-x))
def normalization(M):
    """
    对行向量进行归一化
    :param M:行向量：【dim=len(M)】
    :return: 归一化后的行向量M
    """
    M=M/np.sqrt(np.dot(M,M.T))
    return M

def normalization_all(N):
    """
    对矩阵进行归一化
    :param N: 矩阵：【m,n】
    :return: 归一化后的矩阵M_all:【m,n】
    """
    M_all=[]
    for i in range(len(N)):
        K=normalization(N[i])
        M_all.append(K)
    return M_all

class competitive_network(object):
    def __init__(self,weights,a):
        self.W=np.array(normalization_all(weights))
        self.a=a

    def forward_propagation(self,x):
        """
        前向传播
        """
        x=x.reshape(1,x.shape[0])
        z_layer=np.dot(self.W,x.T)
        a_layer=sigmoid(z_layer)
        argmax=np.where(a_layer==np.amax(a_layer))[0][0]
        return argmax

    def back_propagation(self,argmax,x):
        """
        权值更新
        """
        self.W[argmax] = self.W[argmax]+self.a * (x - self.W[argmax])
        self.W[argmax]=np.array(normalization(self.W[argmax]))
        """self.a-=self.decay"""

    def train(self,X,num_item):
        X=normalization_all(X)
        X=np.array(X)
        self.decay=self.a/num_item
        for item in range(num_item):
            for i in range(X.shape[0]):
                argmax=self.forward_propagation(X[i])
                self.back_propagation(argmax,X[i])
    def prediction(self,x):
        x=np.array(normalization(x))
        argmax=self.forward_propagation(x)
        return argmax




