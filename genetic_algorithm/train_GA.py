import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import os
import time

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
import operator
class Member_discrete(nn.Module):
    def __init__(self, inputdim, outputdim, n_layers, hiddendim, activation, output_activation):
        super(Member_discrete, self).__init__()
        self.score=-99999
        self.reward_score=-99999
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
        d=Categorical(logits=p)
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
        self.score=-99999
        self.reward_score=-99999
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
    def setScore(self, score):
        self.score=score
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
    for(k,v), (anti_k, anti_v) in zip(new_member.get_params(), anti_new_member.get_params()):
        eps=np.random.normal(0,1,v.size())
        v+=torch.from_numpy(sigma*eps).float()
        anti_v+=torch.from_numpy(sigma*(-eps)).float()
    return [new_member, anti_new_member]

class Archive():
    def __init__(self, prob):
        self.saved_member_list=[]
        self.prob=prob
    def save(self, member):
        if random.random()<=self.prob:
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

def compute_fitness(env, member, archive, n_episodes, gamma, max_steps, discrete):
    return compute_reward_fitness(env, member, n_episodes, max_steps, discrete)+gamma*compute_novelty_fitness(member, archive)

def compute_reward_fitness(env, member, n_episodes, max_steps, discrete):
    fitness=0
    for i in range(n_episodes):
        observation=env.reset()
        for t in range(max_steps):
            ob=torch.from_numpy(observation).float().unsqueeze(0).detach()
            action, log_prob=member.run(ob)
            if discrete:
                ac=int(action)
            else:
                ac=action.squeeze(0).numpy()
            observation, reward, done, info=env.step(ac)
            fitness+=reward
            if done:
                break
    member.reward_score=fitness/n_episodes
    return fitness/n_episodes

def compute_novelty_fitness(member, archive):
    return archive.policy_space_distance(member)

def sort_members_in_place(member_list, reverse):
    member_list.sort(key=operator.attrgetter('score'))
    if reverse:
        member_list.reverse()

def get_init_population(n, input_size, output_size, n_layers, size, activation, output_activation, discrete):
    member_list=[]
    for i in range(n):
        member_list.append(build_mlp(input_size, output_size, n_layers, size, activation, output_activation, discrete))
    return member_list

def train_ga(env_name='HalfCheetah-v2', n_exp=1, prob_save=0.05, n_gen=100, gamma=0.5, sigma=0.05, pop_size=50, fitness_eval_episodes=40, max_steps=150, n_elite=20, seed=1, n_layers=1, size=32, network_activation='tanh', output_activation=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env=gym.make(env_name)
    env.seed(seed)

    discrete=isinstance(env.action_space, gym.spaces.Discrete)
    max_steps=max_steps or env.spec.max_episode_steps

    input_size=env.observation_space.shape[0]
    output_size=env.action_space.n if discrete else env.action_space.shape[0]

    if network_activation=='relu':
        activation=torch.nn.functional.relu
    elif network_activation=='leaky_relu':
        activation=torch.nn.functional.leaky_relu
    else:
        activation=torch.nn.functional.tanh

    if output_activation=='relu':
        output_a=torch.nn.functional.relu
    elif output_activation=='leaky_relu':
        output_a=torch.nn.functional.leaky_relu
    elif output_activation=='tanh':
        output_a=torch.nn.functional.tanh
    else:
        output_a=None

    center_return_all=[]
    for i_experiment in range(n_exp):
        member_archive=Archive(prob_save)
        population=get_init_population(pop_size, input_size, output_size, n_layers, size, activation, output_a, discrete)
        
        for member in population:
            member.setScore(compute_fitness(env, member, member_archive, fitness_eval_episodes, gamma, max_steps, discrete))
        
        sort_members_in_place(population, reverse=True)
        
        #save in archive
        for member in population:
            member_archive.save(member)
        
        population=population[:n_elite]
        center_return_list=[]
        current_best_fitness_score=float(population[0].score)
        current_best_reward_score=float(population[0].reward_score)
        print('generation 0 current best', current_best_fitness_score, current_best_reward_score)
        center_return_list.append(current_best_reward_score)
        for i_gen in range(n_gen):
            offsprings=[]

            for i in range(int((pop_size-n_elite)/2)):
                parent_index=random.randint(0,n_elite-1)
                parent=population[parent_index]
                offspring1, offspring2=perturb_member(parent, sigma, input_size, output_size, n_layers, size, activation, output_a, discrete)
                offsprings.append(offspring1)
                offsprings.append(offspring2)

            for member in offsprings:
                member.setScore((compute_fitness(env, member, member_archive, fitness_eval_episodes, gamma, max_steps, discrete)))

            population=population+offsprings
            sort_members_in_place(population, reverse=True)
            for member in offsprings:
                member_archive.save(member)
            population=population[:n_elite]
            current_best_fitness_score=float(population[0].score)
            current_best_reward_score=float(population[0].reward_score)
            print('generation', i_gen+1, 'current best', current_best_fitness_score, current_best_reward_score)
            center_return_list.append(current_best_reward_score)

        center_return_all.append(center_return_list)

        import seaborn as sns
        import matplotlib.pyplot as plt
        for i in range(n_exp):
            ax=sns.tsplot(data=np.array(center_return_all), color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Return')
        plt.tight_layout()
        plt.show()

train_ga()


