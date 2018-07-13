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

def compute_fitness(env, member, archive, n_episodes, gamma, max_steps):
    return compute_reward_fitness(env, member, n_episodes, max_steps)+gamma*compute_novelty_fitness(member, archive)

def compute_reward_fitness(env, member, n_episodes, max_steps):
    fitness=0
    for i in range(n_episode):
        observation=env.reset()
        for t in range(max_steps):
            ob=torch.from_numpy(observation).float().unsqueeze(0).detach()
            action, log_prob=member.run(ob)
            observation, reward, done, info=env.step(action)
            fitness+=reward
            if done:
                break
    return fitness/n_episode

def compute_novelty_fitness(member, archive):
    return archive.policy_space_distance(member)
