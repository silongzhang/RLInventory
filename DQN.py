# -*- coding: utf-8 -*-

import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import DP

T = 7
N = 4

class Environment(object):
    
    def __init__(self, parameterDP, pdf, rf, demand):
        self.parameterDP = parameterDP
        self.pdf = pdf
        self.rf = rf
        self.demand = demand
        
    def setofaction(self, t, current_state):
        return DP.setofaction(t, self.parameterDP, current_state)
    
    def step(self, t, current_state, action):
        try:
            if (current_state.period != t - 1) or (action.period != t) or (t > T + 1):
                raise Exception
            if t == T + 1:
                next_state = utils.State(T + 1, np.zeros(N))
                return self.rf.rewardfunction(t, self.parameterDP, current_state, action, next_state)
            inventory = current_state.inventory + action.order - self.demand.demand[t - 1]
            inventory = utils.maxElementwise(inventory, np.zeros(len(inventory)))
            next_state = utils.State(t, inventory)
            reward = self.rf.rewardfunction(t, self.parameterDP, current_state, action, next_state)
            done = True if t == T + 1 else False
            return next_state, reward, done
        except:
            utils.printErrorAndExit('step')

nets = [
        nn.Sequential(nn.Linear(N * 2, N * 2), nn.ReLU(), nn.Linear(N * 2, 1)),
        nn.Sequential(nn.Linear(N * 2, N * 2), nn.ReLU(), nn.Linear(N * 2, 1)),
        nn.Sequential(nn.Linear(N * 2, N * 2), nn.ReLU(), nn.Linear(N * 2, 1)),
        nn.Sequential(nn.Linear(N * 2, N * 2), nn.ReLU(), nn.Linear(N * 2, 1)),
        nn.Sequential(nn.Linear(N * 2, N * 2), nn.ReLU(), nn.Linear(N * 2, 1)),
        nn.Sequential(nn.Linear(N * 2, N * 2), nn.ReLU(), nn.Linear(N * 2, 1)),
        nn.Sequential(nn.Linear(N * 2, N * 2), nn.ReLU(), nn.Linear(N * 2, 1)),
        nn.Sequential(nn.Linear(N * 2, N * 2), nn.ReLU(), nn.Linear(N * 2, 1))
        ]

def actionValue(nets, t, current_state, action):
    try:
        if (current_state.period != t - 1) or (action.period != t):
            raise Exception
        input = list(current_state.inventory)
        input.extend(action.order)
        input = torch.Tensor(input)
        return nets[t - 1](input)
    except:
        utils.printErrorAndExit('actionValue')

def stateValue(env, nets, t, current_state):
    try:
        if current_state.period != t - 1:
            raise Exception
        if t == T + 1:
            action = utils.Action(T + 1, np.zeros(N))
            return actionValue(nets, t, current_state, action)
        actions = env.setofaction(t, current_state)

        values = []
        for action in actions:
            values.append(actionValue(nets, t, current_state, action))
        idx = values.index(max(values))

        return actionValue(nets, t, current_state, actions[idx])
    except:
        utils.printErrorAndExit('stateValue')

def act(env, nets, t, current_state, epsilon):
    '''given the current state, return the action'''
    
    try:
        if current_state.period != t - 1:
            raise Exception
        if t == T + 1:
            return utils.Action(T + 1, np.zeros(N))
        actions = env.setofaction(t, current_state)
        if random.random() < epsilon:
            return actions[random.randrange(len(actions))]
        else:
            values = []
            for action in actions:
                values.append(actionValue(nets, t, current_state, action))
            pos = values.index(max(values))
            return actions[pos]
    except:
        utils.printErrorAndExit('act')

class ReplayBuffer(object):
    
    def __init__(self, capacity):
        self.buffers = []
        for t in range(T + 1):
            self.buffers.append(deque(maxlen = capacity))
    
    def push(self, t, current_state, action, next_state, reward, done):
        try:
            if (current_state.period != t - 1) or (action.period != t) or (next_state.period != t):
                raise Exception
            self.buffers[t - 1].append((current_state, action, next_state, reward, done))
        except:
            utils.printErrorAndExit('push')
    
    def sample(self, t, batch_size):
        current_states, actions, next_states, rewards, dones = \
            zip(*random.sample(self.buffers[t - 1], batch_size))
        return current_states, actions, next_states, rewards, dones
    
    def len(self, t):
        return len(self.buffers[t - 1])

def train(env, nets, replayBuffer, batch_size, episodes_train, episodes_test, startTime):
    try:
        profits = []
        epsilon = 1.0
        decayEpsilon = 0.99
        
        optimizers = []
        for net in nets:
            optimizers.append(optim.Adam(net.parameters()))
        
        for episode in range(episodes_train + 1):
            epsilon *= decayEpsilon
            profit = 0
            episode_loss = 0
            current_state = utils.State(0, np.zeros(N))
            criterion = torch.nn.MSELoss()
            
            for t in range(1, T + 2):
                if t == T + 1:
                    qOld = stateValue(env, nets, t, current_state)
                    action = utils.Action(T + 1, np.zeros(N))
                    reward = env.step(t, current_state, action)
                    profit += reward
                    qNew = torch.Tensor([reward])
                    loss = criterion(qOld, qNew)
                    episode_loss += loss
                    optimizers[t - 1].zero_grad()
                    loss.backward()
                    optimizers[t - 1].step()
                else:
                    '''
                    action = act(env, nets, t, current_state, epsilon)
                    next_state, reward, done = env.step(t, current_state, action)
                    profit += reward
                    qOld = actionValue(nets, t, current_state, action)
                    qNew = torch.Tensor([reward]) + stateValue(env, nets, t + 1, next_state)
                    loss = criterion(qOld, qNew)
                    episode_loss += loss
                    optimizers[t - 1].zero_grad()
                    loss.backward()
                    optimizers[t - 1].step()

                    current_state = next_state
                    '''
                    action = act(env, nets, t, current_state, epsilon)
                    next_state, reward, done = env.step(t, current_state, action)
                    profit += reward
                    replayBuffer.push(t, current_state, action, next_state, reward, done)
                    current_state = next_state

                    if replayBuffer.len(t) > batch_size:
                        current_states, actions, next_states, rewards, dones = replayBuffer.sample(t, batch_size)
                        qOld = []
                        qNew = []
                        for i in range(batch_size):
                            qOld.append(actionValue(nets, t, current_states[i], actions[i]))
                            qNew.append(torch.Tensor([rewards[i]]) + stateValue(env, nets, t + 1, next_states[i]))
                        qOld = torch.stack(qOld, 0)
                        qNew = torch.stack(qNew, 0)
                        loss = criterion(qOld, qNew)
                        episode_loss += loss
                        optimizers[t - 1].zero_grad()
                        loss.backward()
                        optimizers[t - 1].step()
            profits.append(profit)
            if episode % 100 == 0:
                print('episode = {} \t time = {:.2f} \t loss = {:.2f} \t average training profit = {} \t average testing profit = {}'\
                          .format(episode, utils.runTime(startTime), episode_loss, \
                                  np.mean(profits), test(env, nets, episodes_test, 0)))
                profits = []
                epsilon = 1.0
    except:
        utils.printErrorAndExit('train')

def test(env, nets, episodes, epsilon):
    profits = []
    for episode in range(1, episodes + 1):
        profit = 0
        current_state = utils.State(0, np.zeros(N))
        
        for t in range(1, T + 2):
            if t == T + 1:
                action = utils.Action(T + 1, np.zeros(N))
                profit += env.step(t, current_state, action)
            else:
                action = act(env, nets, t, current_state, epsilon)
                next_state, reward, done = env.step(t, current_state, action)
                current_state = next_state
                profit += reward
        profits.append(profit)
    return np.mean(profits)

def DQNMethod(InFile, batch_size, buffer_size, episodes_train, episodes_test, startTime):
    parameterDP, pdf, rf, demand = DP.readData(InFile)
    global T, N
    T = parameterDP.T
    N = parameterDP.N
    
    env = Environment(parameterDP, pdf, rf, demand)
    replayBuffer = ReplayBuffer(buffer_size)
    
    print('Training ...')
    train(env, nets, replayBuffer, batch_size, episodes_train, episodes_test, startTime)
    
    print('Testing ...')
    return test(env, nets, episodes_test, 0)

