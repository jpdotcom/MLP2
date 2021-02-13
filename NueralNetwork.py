import numpy as np
import tensorflow as tf
import random
import time
import json

with open('data.json','r') as f:
    data=json.load(f)

class NueralNetwork:


    def getrandparameters(self,row,col):
        return np.random.rand(row,col)-0.5
    def __init__(self,sizes,isTrained=False):

        #input layer
        self.flatten=None 
        self.inputsize=sizes[0]
        #layer 1
        self.layer1_weights=np.array(data["0"]) if isTrained else self.getrandparameters(sizes[1],sizes[0])
        self.layer1_bias=np.array(data["1"]) if isTrained else self.getrandparameters(sizes[1],1)
        #layer 2
        self.layer2_weights=np.array(data["2"]) if isTrained else self.getrandparameters(sizes[2],sizes[1])
        self.layer2_bias=np.array(data["3"]) if isTrained else self.getrandparameters(sizes[2],1)
        #layer3
        self.layer3_weights=np.array(data["4"]) if isTrained else self.getrandparameters(sizes[3],sizes[2])
        self.layer3_bias=np.array(data["5"]) if isTrained else self.getrandparameters(sizes[3],1)
    def ReLU(self,x):
        
        return x*(x>0)
    def ReLUd(self,x):
       
        return (x>0)*1
    def softmax(self,x):
        
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x,axis=0)
    def run(self,img):
        
       
        
        #Flatten the image
        self.flatten=img.reshape((self.inputsize,1))
        curr_layer=self.flatten
        
        #layer 1
        self.layer1=self.ReLU(self.layer1_weights.dot(curr_layer)+self.layer1_bias)
        curr_layer=self.layer1
        
        #layer 2
        self.layer2=self.ReLU(self.layer2_weights.dot(curr_layer)+self.layer2_bias)
        curr_layer=self.layer2
        
        #layer3
        
        self.layer3=self.softmax(self.layer3_weights.dot(curr_layer)+self.layer3_bias)
        
        return np.argmax(self.layer3)
    def findgradient(self,truth):
        truth=truth.reshape(truth.shape[0],1)
        curr_error=self.layer3-truth
        
        #layer 3
        self.layer3_weights_gradient=curr_error.dot(np.transpose(self.layer2))
        self.layer3_bias_gradient=curr_error
        curr_error=np.transpose(self.layer3_weights).dot(curr_error)

        #layer 2
        curr_error=curr_error*self.ReLUd(self.layer2)
        self.layer2_weights_gradient=curr_error.dot(np.transpose(self.layer1))
        self.layer2_bias_gradient=curr_error 
        curr_error=np.transpose(self.layer2_weights).dot(curr_error)
        
        #layer 1 
        curr_error=curr_error*self.ReLUd(self.layer1)
        self.layer1_weights_gradient=curr_error.dot(np.transpose(self.flatten))
        self.layer1_bias_gradient=curr_error
        
        
    def gradient_descent(self,n,num_images,s):

        #layer 1
        self.layer1_weights=self.layer1_weights*(1-(s*n)/num_images)-n*self.layer1_weights_gradient
        
        self.layer1_bias-=n*self.layer1_bias_gradient

        #layer 2
        self.layer2_weights=self.layer2_weights*(1-(s*n)/num_images)-n*self.layer2_weights_gradient
        
        self.layer2_bias-=n*self.layer2_bias_gradient

        #layer 3
        
        self.layer3_weights=self.layer3_weights*(1-(s*n)/num_images)-n*self.layer3_weights_gradient
        
        self.layer3_bias-=n*self.layer3_bias_gradient
    
    def train(self,img,label,num_images,s):
        truth=[0]*10
        truth[label]=1
        truth=np.array(truth)
        for i in range(8):
            self.run(img)
            self.findgradient(truth)
            self.gradient_descent(0.005,num_images,s)
    
    def checkaccuracy(self,val_data):

        cor=0
        tot=0

        for img,label in val_data:
            img=np.array(img)
            guess=self.run(img)
            
            if guess==label:
                cor+=1 
            tot+=1 
        return (cor/tot)*100
    def getparameters(self):
        return {0:self.layer1_weights.tolist(),1:self.layer1_bias.tolist(),2:self.layer2_weights.tolist(),3:self.layer2_bias.tolist(),4:self.layer3_weights.tolist(),5:self.layer3_bias.tolist()}



def update_file(data,acc):
    
    with open('data.json','w') as f:
        
        json.dump(data,f)
        f.close()
    print('New parameters saved. Best accuracy: '+ str(acc))

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train=x_train/255
x_test=x_test/255
x_train=list(zip(x_train.tolist(),y_train.tolist()))
val_data=list(zip(x_test.tolist(),y_test.tolist()))

myNetwork=NueralNetwork([784,128,64,10],True)

epoch=60

num_images=60000
begin_accuracy=myNetwork.checkaccuracy(val_data)
best_accuracy=begin_accuracy
print('Initial Accuracy: '+ str(begin_accuracy))
for x in range(epoch):
    s=time.time()
    random.shuffle(x_train)

    for i in range(num_images):
        
        img,label=x_train[i]
        img=np.array(img)
        myNetwork.train(img,label,num_images,0.0001*num_images)
    
    print('Epoch '+ str(x+1)+' Done | Time Taken: '+ str(time.time()-s))
    acc=myNetwork.checkaccuracy(val_data)
    
    if best_accuracy<acc:
        best_accuracy=acc
        params=myNetwork.getparameters()
      
        update_file(params,acc)

