#coding="utf-8"
"""
desc: datahelper
author:luchi
date:9/3/17
"""

import numpy as np
import data_process.preprocess as dp


def batch_iter(data,num_epoch,batch_size):

    user_favor=data

    total_size = len(user_favor)
    batch_num = (total_size-1)/batch_size+1
    for i in range(num_epoch):
        if(num_epoch!=1):
            print("training %i epoch"%i)
        shuffle_index = range(total_size)
	np.random.shuffle(shuffle_index)
        user_favor = [user_favor[j] for j in shuffle_index]
        for k in range(batch_num):
	    if k%100==0:
	        print("the %i epoch  %i -th batch"%(i,k))
            begin_index = batch_size*k
            end_index = min(total_size,(k+1)*batch_size)
            train_x = user_favor[begin_index:end_index]
            yield train_x






