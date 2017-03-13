#coding='utf-8'
"""
desc: RBM model
author:luchi
date:9/3/17
"""
import numpy as np
class RBM_Model(object):

    def __init__(self,visible_size,hidden_size,lr):
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.lr = lr
        np.random.seed(10)
        self.b_v = np.random.uniform(-1,1,size=[self.visible_size])*0
        self.W = np.random.uniform(-1,1,size=[self.visible_size,self.hidden_size])
        self.b_h = np.random.uniform(-1,1,size=[self.hidden_size])*0

    def sampling(self,data):

        """
        sampling h_0 using v_0
        """
        
        h_0 = self.logist_fun(np.dot(data,self.W)+self.b_h)
	#print h_0
        h_shape = np.shape(h_0)
        #h_0_state = h_0>(np.random.rand(h_shape[0],h_shape[1]))
	h_0_state = h_0>(np.ones_like(h_0)*0.5)
       
       
        """
        building contrastive sampling
        """
        v_1 = self.logist_fun(np.dot(h_0_state,np.transpose(self.W))+self.b_v)
        v_shape = np.shape(v_1)
        #v_1_state = v_1>(np.random.rand(v_shape[0],v_shape[1]))
	v_1_state = v_1>(np.ones_like(v_1)*0.5)
        h_1 = self.logist_fun(np.dot(v_1,self.W)+self.b_h)

        return h_0,v_1,h_1,v_1_state


    def train(self,data,iter_time):

        h_0,v_1,h_1,v_1_state = self.sampling(data)
        if iter_time%100==0:
  	    error = np.sum(np.mean((data-v_1) ** 2,axis=0))
            print("the %i iter_time error is %s" % (iter_time, error))   

        """
        updating weight
        """
        updating_matrix = []
        size = len(data)

        for i in range(size):
            w_v0= np.reshape(data[i],[self.visible_size,1])
            w_h0 = np.reshape(h_0[i],[1,self.hidden_size])
            w_u0 = np.dot(w_v0,w_h0)

            w_v1 = np.reshape(v_1[i],[self.visible_size,1])
            w_h1 =  np.reshape(h_1[i],[1,self.hidden_size])
            w_u1 = np.dot(w_v1,w_h1)

            updating_matrix.append(w_u0-w_u1)
        updating_matrix =  np.mean(np.array(updating_matrix),axis=0)
        self.W =  self.W + self.lr*updating_matrix
        self.b_v = self.b_v + self.lr*np.mean((data-v_1),axis=0)
        self.b_h = self.b_h + self.lr*np.mean((h_0-h_1),axis=0)

    def logist_fun(self,narray):
        narray =  np.clip(narray,-100,100)
        return 1.0/(1+np.exp(-1*narray))

    def softmax(self,narray):
	narray =  np.clip(narray,-100,100)
        num_a = np.exp(narray)
        num_b = np.sum(num_a,axis=1)
        return num_a*1.0/num_b[:,None]

    def recomendation(self,test_data,topK):
       
        h_0,v_1,h_1 ,_= self.sampling(test_data)
        sorted_index = np.argsort(-1*v_1,axis=1)
        return sorted_index[:,:topK]











