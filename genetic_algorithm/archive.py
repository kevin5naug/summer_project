import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
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

class Archive():
    def __init__(self):
        self.saved_member_list=[]
    def save(self, member, prob=0.05):
        if random.random()<=prob:
            self.saved_member_list.append(member)
    def policy_space_distance(self, target, p_norm=2):
        min_dist=-1
        for member in self.saved_member_list:
            current_dist=0
            for(current_k, current_v), (target_k, target_v) in zip(member.get_params(), target.get_params()):
                current_dist+=torch.dist(current_v, target_v, p=p_norm)
            if min_dist==-1:
                min_dist=current_dist
            elif min_dist>current_dist:
                min_dist=current_dist
        return min_dist

