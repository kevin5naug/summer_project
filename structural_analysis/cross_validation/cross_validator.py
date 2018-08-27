import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import pickle
from copy import deepcopy
import random
device = torch.device(0 if torch.cuda.is_available() else "cpu")

class CrossValidator:
    def __init__(self, model, data, compute_acc, loss_function,
                 partition=5, decoder=None, batch_size=100, epochs=50, lr=1e-3):
        self.model=model
        self.data=data
        self.data_size=len(data)
        self.partition=partition
        self.decoder=decoder
        self.compute_acc=compute_acc
        self.train_data=[]
        self.val_X=[]
        self.val_Y=[]
        self.acc01=0
        self.acc10=0
        self.loss_history=[]
        self.loss_function=loss_function
        self.batch_size=batch_size
        self.epochs=epochs
        self.lr=lr
        
    def create_data(self, part):
        train_data=[]
        val_X=[]
        val_Y=[]
        cut=int(self.data_size/self.partition)
        for i, x in enumerate(self.data):
            if i>=cut*part and i<cut*(part+1):
                train_data.append([x[0],x[1]])
            else:
                val_X.append(x[0])
                val_Y.append(x[1])
        return train_data, val_X, val_Y
    
    def tensorize(self, p):
        p=np.array(p)
        p=torch.from_numpy(p).float()
        p=p.to(device)
        return p
    
    def create_batch(self, index):
        output_X=[]
        output_Y=[]
        for i in range(self.batch_size):
            if self.batch_size*index+i<len(self.train_data):
                output_X.append(self.train_data[self.batch_size*index+i][0])
                output_Y.append(self.train_data[self.batch_size*index+i][1])
        return output_X, output_Y
                
    
    def compute(self):
        for i in range(self.partition):
            self.train_data, self.val_X, self.val_Y = self.create_data(i)
            self.val_X=self.tensorize(self.val_X)
            self.val_Y=self.tensorize(self.val_Y)
            self.val_Y=self.val_Y.long()
            cur_model=deepcopy(self.model).to(device)
            optimizer=optim.Adam(cur_model.parameters(), lr=self.lr, weight_decay=5e-8)
            loss=0
            train_len=len(self.train_data)
            self.loss_history.append([])
            for j in range(self.epochs):
                random.shuffle(self.train_data)
                for k in range(int(train_len/self.batch_size)):
                    batch_X, batch_Y = self.create_batch(k)
                    batch_X = self.tensorize(batch_X)
                    batch_Y = self.tensorize(batch_Y)
                    batch_Y = batch_Y.long()
                    optimizer.zero_grad()
                    output_Y = cur_model(batch_X)
                    loss=self.loss_function(output_Y, batch_Y)
                    loss.backward()
                    optimizer.step()
                self.loss_history[i].append(loss.cpu().data)
                print("compeleted: ", float(i*self.epochs+j)/float(self.partition*self.epochs))
            output_Y = cur_model(self.val_X)
            if self.decoder!=None: output_Y=self.decoder(output_Y)
            temp01, temp10 = self.compute_acc(output_Y, self.val_Y)
            self.acc01+=temp01
            self.acc10+=temp10
            print(self.acc01, self.acc10)
        self.acc01 /= float(self.partition)
        self.acc10 /= float(self.partition)
        return self.acc01, self.acc10, self.loss_history
