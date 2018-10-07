import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym.spaces
import sys
import numpy as np
import argparse
import os
import random
import operator

def get_action(obs,theta):
    decision_value = np.dot(obs,theta)

    if is_continuous:
        if decision_value > 0:
            return 1
        else:
            return -1

    if decision_value > 0:
        return 1
    else:
        return 0

class Member:
    def __init__(self, zero_init=False):
        self.theta = np.random.normal(0,0.01,(4,))
        if zero_init:
            self.theta = np.zeros((4,))
        self.score = -99999
    def get_mutated_offspring(self,sigma,init_chance=0.1):
        ## get a mutated offspring of this member
        offspring = Member()

        if random.random() < init_chance: # get randomly initialized offspring
            return offspring

        # or get offspring by adding noise to parent param
        offspring.theta = self.theta + np.random.normal(0,sigma)
        return offspring
    def setScore(self,score):
        self.score = score
    def getTheta(self):
        return self.theta

def compute_fitness(env, member, n_episode):
    # run monte carlo roll out to get fitness estimate
    fitness = 0
    theta = member.getTheta()
    for i_episode in range(n_episode):
        observation = env.reset()
        for t in range(200):
            # env.render()
            action = get_action(observation,theta)
            observation, reward, done, info = env.step(action)
            fitness += reward
            if done:
                break
        ## now we have finished one episode, we now assign reward (all the data points in
        ## the same trajectory have the same reward)
    return fitness/n_episode

def sort_members_in_place(member_list,reverse):
    ## given a list of members, return the sorted list, small to large
    member_list.sort(key=operator.attrgetter('score'))
    if reverse:
        member_list.reverse()

def get_init_population(n):
    # get the initial population
    member_list = []
    for i in range(n):
        member_list.append(Member(False))
    return member_list

sigma = 0.01
## theta here is the center theta

POP_SIZE = 50
fitness_eval_episodes = 10
N_GEN = 100

T = 20 # number of TOP

env_name = 'CartPole-v0'
## uncomment following one line to test on Inverted Pendulum.
# env_name = 'InvertedPendulum-v2'
if env_name == 'InvertedPendulum-v2':
    is_continuous = True
else:
    is_continuous = False

center_return_all = []
for i_experiment in range(5):
    env = gym.make(env_name)

    population = get_init_population(POP_SIZE)
    for member in population:
        member.setScore(compute_fitness(env, member, fitness_eval_episodes))

    center_return_list = []
    for i_gen in range(N_GEN):
        offsprings = []
        # get all offsprings

        for i in range(POP_SIZE-T):
            # pick random parent
            parent_index = random.randint(0,T-1)
            parent = population[parent_index]
            # get offspring from this parent
            offspring = parent.get_mutated_offspring(sigma)
            offsprings.append(offspring)

        for member in offsprings:
            member.setScore(compute_fitness(env, member, fitness_eval_episodes))

        population = population + offsprings
        # sort parents and offsprings
        sort_members_in_place(population, reverse=True)
        # only keep the best ones
        population = population[:T]

        current_best_score = population[0].score
        print('gen',i_gen,'current best',current_best_score)
        center_return_list.append(current_best_score)

    center_return_all.append(center_return_list)
#
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# for i in range(5):
#     ax = sns.tsplot(data=np.array(center_return_all),color='blue')
#
# plt.xlabel('Epoch')
# plt.ylabel('Return')
# plt.tight_layout()
# plt.show()
#
#

