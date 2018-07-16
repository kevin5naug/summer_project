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


def get_action_continuous(obs, theta):
    decision_value = np.dot(obs,theta)

    if decision_value > 3:
        return  3
    if decision_value < -3:
        return -3

    return decision_value

def compute_fitness(env, theta, n_episode=1):
    fitness = 0
    for i_episode in range(n_episode):
        observation = env.reset()
        for t in range(1000):
            # env.render()
            action = get_action_continuous(observation, theta)
            observation, reward, done, info = env.step(action)
            fitness += reward
            if done:
                break
        ## now we have finished one episode, we now assign reward (all the data points in
        ## the same trajectory have the same reward)
    return fitness/n_episode

sigma = 0.1
## theta here is the center theta
theta_center = np.zeros((1,4))

POP_SIZE = 200
fitness_eval_episodes = 1
N_GEN = 60

epsilon_all = np.random.normal(0,sigma,(POP_SIZE,4))
theta_all = epsilon_all + theta_center

ALPHA = 0.05
# for i_gen in range(N_GENERATIONS):

fitness_list = [0 for _ in range(POP_SIZE)]

def reshape_fitness(fitness_list):
    order_list = np.argsort(fitness_list)
    for i in range(len(fitness_list)):
        rank = order_list[i]
        fitness_list[i] = rank/POP_SIZE
    return fitness_list

env = gym.make('InvertedPendulum-v2')
print(env.action_space)
print(env.observation_space)

center_return_all = []
for i_experiment in range(5):
    env = gym.make('InvertedPendulum-v2')
    theta_center = np.zeros((1, 4))
    epsilon_all = np.random.normal(0, sigma, (POP_SIZE, 4))
    theta_all = epsilon_all + theta_center

    center_return_list = []
    center_return_list.append(compute_fitness(env,theta_center.reshape(-1),10))
    for i_gen in range(N_GEN):
        for i_pop in range(POP_SIZE):
            theta = theta_all[i_pop]
            fitness = compute_fitness(env,theta,fitness_eval_episodes)
            fitness_list[i_pop] = fitness
        ave_fit = np.sum(fitness_list)/POP_SIZE
        for i_pop in range(POP_SIZE):
            fitness = fitness_list[i_pop]
            theta_center += ALPHA * fitness * epsilon_all[i_pop]/POP_SIZE ## update theta center
        ## now we perturb
        epsilon_all = np.random.normal(0, sigma, (POP_SIZE, 4))
        theta_all = epsilon_all + theta_center
        center_return = compute_fitness(env,theta_center.reshape(-1),10)
        print('gen',i_gen,'center',center_return,'ave',ave_fit)
        center_return_list.append(center_return)
        print(theta_center)
        if center_return > 999:
            break
    currentLen = len(center_return_list)
    fulllen = N_GEN+1
    center_return_list += [1000 for _ in range(fulllen-currentLen)]
    center_return_all.append(center_return_list)

import seaborn as sns
import matplotlib.pyplot as plt

for i in range(5):
    ax = sns.tsplot(data=np.array(center_return_all),color='blue')

plt.xlabel('Epoch')
plt.ylabel('Return')
plt.tight_layout()
plt.show()



