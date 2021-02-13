import numpy as np
import time
import random
def ReLU(x):

    s=time.time()
    
    ans=np.maximum(x, 0)
    print(time.time()-s)
    
    return ans

def ReLU2(x):
    s=time.time()
    ans=x*(x>0)

    print(time.time()-s)
    return ans

def ReLUd(x):

    s=time.time()
    x=(x>0)*1
    print(time.time()-s)
    return x 
def softmax(x):
    s=time.time()
    e_x=np.exp(x)
    ans=e_x/np.sum(e_x,axis=0)
    print(time.time()-s)
    return ans

def sigmoid(x):
    
    s=time.time()
    x_list=x.tolist()
    
    for i in range(len(x_list)):
        val=x_list[i][0]
        
        x_list[i]=1/(1+np.exp(-1*val))
    
    ans=np.array(x_list)
    print(time.time()-s)
    return ans



x=np.array([0,1,2,0])

print(softmax(x))



