import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import os
import time
from multiprocessing import Process

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import torch.distributions as D
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
import random

class Member_discrete(nn.Module):
    def __init__(self, inputdim, outputdim, n_layers, hiddendim, activation, output_activation):
        super(Member_discrete, self).__init__()
        self.score=-99999
        if (output_activation==None):
            self.original_output=True
        else:
            self.original_output=False
        self.history_of_log_probs=[]
        self.layers=nn.ModuleList()
        self.activation=activation
        self.output_activation=output_activation
        self.n_layers=n_layers+1
        if self.n_layers==1:
            self.layers.append(nn.Linear(inputdim, outputdim))
        else:
            for i in range(self.n_layers):
                if(i==0):
                    self.layers.append(nn.Linear(inputdim, hiddendim))
                elif(i==(self.n_layers-1)):
                    self.layers.append(nn.Linear(hiddendim, outputdim))
                else:
                    self.layers.append(nn.Linear(hiddendim, hiddendim))
    def forward(self, x):
        for i, l in enumerate(self.layers):
            if (i<(self.n_layers-1)):
                x=l(x)
                x=self.activation(x)
            else:
                x=l(x)
                if self.original_output:
                    return x
                else:
                    x=self.output_activation(x)
                    return x
    def run(self, x):
        p=self(x)
        if self.original_output:
            d=Categorical(logits=p)
        else:
            #Suppose after the output_activation, we get the probability(i.e. a softmax activation)
            #This assumption might be false.
            d=Categorical(probs=p)
        action=d.sample()
        log_prob=d.log_prob(action)
        return action, log_prob
    def setScore(self, score):
        self.score=score
    def get_params(self):
        return [(k,v) for k,v in zip(self.state_dict().keys(), self.state_dict().values())]

class Member_continuous(nn.Module):
    def __init__(self, inputdim, outputdim, n_layers, hiddendim, activation, output_activation):
        super(Member_continuous, self).__init__()
        if (output_activation==None):
            self.original_output=True
        else:
            self.original_output=False
        self.activation=activation
        self.output_activation=output_activation
        self.history_of_log_probs=[]
        self.n_layers=n_layers+1
        self.layers=nn.ModuleList()
        if self.n_layers==1:
            self.mean=nn.Linear(inputdim, outputdim)
            self.logstd_raw=nn.Linear(inputdim, outputdim)
        else:
            for i in range(self.n_layers-1):
                if(i==0):
                    self.layers.append(nn.Linear(inputdim, hiddendim))
                else:
                    self.layers.append(nn.Linear(hiddendim, hiddendim))
            self.mean=nn.Linear(hiddendim, outputdim)
            self.logstd_raw=nn.Linear(hiddendim, outputdim)
    def forward(self, x):
        for i, l in enumerate(self.layers):
            x=l(x)
            x=self.activation(x)
        u=self.mean(x)
        logstd=self.logstd_raw(x)
        if self.original_output:
            return u, logstd
        else:
            u=self.output_activation(u)
            logstd=self.output_activation(logstd)
            return u, logstd
    def run(self, x):
        x=Variable(x)
        u, logstd=self(x)
        d=D.Normal(loc=u, scale=logstd.exp()) #might want to use N Gaussian instead
        action=d.sample().detach()
        log_prob=d.log_prob(action).sum(1).view(-1,1)
        return action, log_prob
    def get_params(self):
        return [(k,v) for k,v in zip(self.state_dict().keys(), self.state_dict().values())]

def build_mlp(
        input_size, 
        output_size, 
        n_layers, 
        size, 
        activation,
        output_activation,
        discrete
        ):
    
    if discrete:
        return Member_discrete(input_size, output_size, n_layers, size, activation, output_activation)
    else:
        return Member_continuous(input_size, output_size, n_layers, size, activation, output_activation)

def perturb_member(member, sigma, input_size, output_size,\
        n_layers, size, activation,\
        output_activation, discrete):
    
    new_member=build_mlp(input_size, output_size, n_layers, size, activation, output_activation, discrete)
    anti_new_member=build_mlp(input_size, output_size, n_layers, size, activation, output_activation, discrete)
    new_member.load_state_dict(member.state_dict())
    anti_new_member.load_state_dict(member.state_dict())
    for(k,v), (anti_k, anti_v) in zip(new_model.get_params(), anti_model.get_params()):
        eps=np.random.normal(0,1,v.size())
        v+=torch.from_numpy(sigma*eps).float()
        anti_v+=torch.from_numpy(sigma*(-eps)).float()
    return [new_member, anti_new_member]


