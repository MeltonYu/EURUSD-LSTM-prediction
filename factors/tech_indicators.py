import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

# input: 1-d numpy.ndarray
def techindicator_to_array(array,tech_function,*args):
    tech_array = np.full(shape=(len(array)),fill_value=np.nan)
    
    args_ls = list(args)
    
    if len(array)< args_ls[0]:
        return tech_array
    
    else:
        for i in range(args_ls[0]-1,len(array)):
            tech_array[i] = tech_function(array[i-args_ls[0]+1:i+1],*args)
            
        return tech_array
    
def SMA(array,period):
    assert len(array) == period
    return np.mean(array)
    
def EMA(array,period,smoothing = 2):
    assert len(array) == period
    multiplier = smoothing/(1+len(array))
    EMA = array[0]    
    for i in range(1,len(array)):
        EMA = EMA*(1-multiplier)+array[i]*multiplier
        
    return EMA

def RSI(array,period):
    assert len(array) == period
    array = np.diff(array)
    positive_return = [0,0]
    negative_return = [0,0]
    
    for i in range(len(array)):
        if array[i] >= 0 :
            positive_return[0] += 1
            positive_return[1] += array[i]
            
        else:
            negative_return[0] +=1
            negative_return[1] += -array[i]
            
    if positive_return[0] == 0:
        return 0
    
    elif negative_return[0] == 0:
        return 100
    
    else:
        return 100 - 100/(1+(positive_return[1]/positive_return[0])/(negative_return[1]/negative_return[0]))
    
def MACD(array,long_period = 26, short_period = 12):
    assert len(array) == long_period, (
    f"The length of the input array must equal the long_period! But now "
    f"the array length is {len(array)}, long period is {long_period}, short period is {short_period}!"
)
    
    return  EMA(array[len(array)-short_period:],short_period) - EMA(array,long_period)
    
def ROC(array,period):
    assert len(array) == period
    return (array[-1]-array[0])/array[0]

def Bollinger_Bands_lower(array, period,standard_deviation_num = 2):
    assert len(array) == period
    array_mean = np.mean(array)
    array_deviation = np.std(array)
    
    return array_mean-2*array_deviation

def Bollinger_Bands_upper(array, period,standard_deviation_num = 2):
    assert len(array) == period
    array_mean = np.mean(array)
    array_deviation = np.std(array)
    
    return array_mean+2*array_deviation

def CCI(array,period):
    assert len(array) == period
    MA = np.mean(array)
    mean_abs_deviation = np.mean(np.abs(array-MA))
    
    return  (array[-1]-MA)/(0.015*mean_abs_deviation+1e-7)  
    
    
    
    
    


    
        
        
    
    
            
            
        