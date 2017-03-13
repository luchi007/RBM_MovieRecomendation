#coding='utf-8'
"""
author:luchi
date:9/3/17
"""

import  data_helper as dh
import RBM_Model as RBM
import data_process.preprocess as pp
import numpy as np
import os
import math
W_path ='W.npy'
b_v_path='b_v.npy'
b_h_path='b_h.npy'

def load_param(path):
    f=open(path,'rb')
    ob=np.load(f)
    f.close()
    return np.array(ob)
def run(train_x,visible_size,hidden_szie=30,num_epoch=100,batch_size=16,lr=1e-3,test_rate=0.1):

    print("training begin")
    rbm = RBM.RBM_Model(visible_size=visible_size,hidden_size=hidden_szie,lr=lr)
    if os.path.exists(W_path):
	rbm.W = load_param(W_path)
        rbm_b_v = load_param(b_v_path)
        rbm_b_h = load_param(b_h_path)
    iter_time=0
    for data in dh.batch_iter(train_x,num_epoch,batch_size):
	#rbm.lr = lr*math.pow(10,-1*(iter_time/5000))
	iter_time+=1
        rbm.train(data,iter_time)
    print("saving parameters...")
    np.save(W_path,rbm.W)
    np.save(b_v_path,rbm.b_v)
    np.save(b_h_path,rbm.b_h)
    print("save done!")    
    return rbm

def test(test_x,rbm,batch_size = 16,topK=30):

    
    test_recomendation=[]
    test_s=[]
    for test_data in dh.batch_iter(test_x,num_epoch=1,batch_size=batch_size):
        v1_state=rbm.recomendation(test_data,topK)
        test_s.extend(test_data)
        test_recomendation.extend(v1_state)
    return test_s,test_recomendation

def main():

    user_rating , movie_num ,movie_index2name= pp.load_data()
    user_favor = user_rating.values()
    user_id = user_rating.keys()
    shuffle_index=range(len(user_favor))
    np.random.shuffle(shuffle_index)
    user_favor = [user_favor[j] for j in shuffle_index]
    user_id = [user_id[j] for j in shuffle_index]
    test_size = int(np.round(len(user_id)*0.1))
    print ("test size is %i"%(test_size))
    train_x = user_favor[:-test_size]
    test_x = user_favor[-test_size:]
    
    rbm = run(train_x,movie_num)
    test_source,test_recommendation= test(test_x,rbm)
    #print rbm.W
    print("show result")
    size = len(test_source)
    f=open('recomendation.txt','wa')
    for i in range(size):
        #print("history of user watch movies")
	f.write("history of user watch movies"+'\n')
        history_movie = []
        recommendation_movie=[]
        for idx,j in enumerate(test_source[i]):
            if j ==1 :
                history_movie.append(movie_index2name[idx])
        #print(' '.join(history_movie)+'\n')
	f.write(' '.join(history_movie)+'\n')
        #print("recommendation movie:")
	f.write("recommendation movie:"+'\n')
        #print test_recommendation[i]
	if test_recommendation[i]==None:
            recommendation_movie=['None']
        else:
	    for j in test_recommendation[i]:
	        recommendation_movie.append(movie_index2name[j])
        #print(' '.join(recommendation_movie)+'\n')
	f.write(' '.join(recommendation_movie)+'\n')
        #print('\n')

if __name__ == '__main__':
    main()











