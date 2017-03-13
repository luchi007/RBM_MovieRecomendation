#coding=utf-8
"""
desc:this script is used to process source data from Movielens movie dataset
dataset acknowledgement :F. Maxwell Harper and Joseph A. Konstan. 2015.
                    The MovieLens Datasets: History and Context.
                    ACM Transactions on Interactive Intelligent Systems (TiiS)
                    5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872
author:luchi
date:9/3/2017
"""
import numpy as np

movie_path = 'data/movies.dat'
rating_path = 'data/ratings.dat'

def load_data():
    f= open(movie_path,'rb')
    movie_index2name=dict([])
    movie_id2index=dict([])
    for i,line in enumerate(f.readlines()):
        line = line.strip()
        detail = line.split('::')
        movie_index2name[i]='[ '+detail[1]+" type: "+detail[2].split('|')[0]+" ]"
        movie_id2index[detail[0]]=i

    movies_num = len(movie_id2index.keys())
    print("movie len is %d"%(movies_num))

    f.close()

    user_rating=dict([])
    f=open(rating_path,'rb')
    for line in f.readlines():
	
        line = line.strip()
        detail = line.split('::')
        user_id = detail[0]
        movie_id = detail[1]
        movie_rate = float(detail[2])
        movie_favor = 0 if movie_rate<3 else 1
        movie_index = movie_id2index[movie_id]
        if user_rating.has_key(user_id):
            value = user_rating[user_id]
            value[movie_index]=movie_favor
            user_rating[user_id]=value
        else:
            value =[0]*movies_num
            value[movie_index]=movie_favor
            user_rating[user_id]=value
    f.close()
    return user_rating,movies_num,movie_index2name




