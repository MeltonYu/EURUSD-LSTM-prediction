import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


def rolling_split(array,time_delta):
    sub_array_list = []
    
    for i in range(np.shape(array)[0]-time_delta+1):
        sub_array_list.append(array[i:(i+time_delta),...])
        
    return np.stack(sub_array_list,axis=0)

def splitting_entropy(array,value):
    p_dec = percentileofscore(array,-value)/100
    p_inc = 1-percentileofscore(array,value)/100
    p_sid = 1-p_dec-p_inc
    if p_dec==0 or p_inc == 0 or p_sid ==0:
        return 0
    return -(p_dec*np.log(p_dec)+p_inc*np.log(p_inc)+p_sid*np.log(p_sid))

def threshold_search(array,delta_step=1e-5,upper_boundary=0.85):
    abs_array = np.abs(array)
    abs_upper_point = np.percentile(abs_array,upper_boundary*100)
    entropy_list = []
    #for i in range(1,np.floor(abs_upper_point/delta_step).astype(int)+1):
        #entropy_list.append(splitting_entropy(array,i*delta_step))
    threshold = delta_step
    while threshold <= abs_upper_point:
        entropy_list.append(splitting_entropy(array,threshold))
        threshold += delta_step
        
    return (np.argmax(entropy_list)+1)*delta_step

def labelize(array,threshold):
    return np.where(array>threshold,2,np.where(array<-threshold,0,1))

def vote(result1,result2):
    assert result1.shape[0] == result2.shape[0],"Must be the same length !"
    condition1 = (np.argmax(result1,axis=-1)==1)|(np.argmax(result2,axis=-1)==1)
    
    return np.where(condition1,1,np.argmax(np.concatenate([result1,result2],axis=-1),axis=-1)//3)
    
    
    