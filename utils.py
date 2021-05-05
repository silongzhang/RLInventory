# -*- coding: utf-8 -*-

import numpy as np
import sys
import time
import copy

PPM = 1e-6

def printErrorAndExit(info):
    print('Error: ' + info)
    sys.exit(1)

def runTime(start):
    return time.process_time() - start

def realNumEqual(lhs, rhs, precision):
    return abs(lhs - rhs) < precision

def realSequenceEqual(lhs, rhs, precision):
    for i in range(0, len(lhs)):
        if not realNumEqual(lhs[i], rhs[i], precision):
            return False

    return True

def minElementwise(lhs, rhs):
    try:
        if len(lhs) != len(rhs):
            raise Exception
    except:
        printErrorAndExit('minElementwise')
    result = []
    for i in range(0, len(lhs)):
        result.append(lhs[i]) if lhs[i] < rhs[i] else result.append(rhs[i])
    return result

def maxElementwise(lhs, rhs):
    try:
        if len(lhs) != len(rhs):
            raise Exception
    except:
        printErrorAndExit('maxElementwise')
    result = []
    for i in range(0, len(lhs)):
        result.append(lhs[i]) if lhs[i] > rhs[i] else result.append(rhs[i])
    return result

def enumerate(result, limit, partial, pos):
    if pos == len(limit):
        result.append(copy.deepcopy(partial))
        return
    for elem in range(0, int(limit[pos] + 1)):
        partial[pos] += elem
        enumerate(result, limit, partial, pos + 1)
        partial[pos] -= elem

class ParameterDP(object):
    
    def __init__(self, T, N, MaxInventory, MaxOrder):
        self.T = T
        self.N = N
        self.MaxInventory = MaxInventory
        self.MaxOrder = MaxOrder

class State(object):
    
    def __init__(self, period, inventory):
        self.period = period
        self.inventory = inventory
        
    def hash(self):
        s = [int(self.period)]
        for elem in self.inventory:
            s.append(int(elem))
        return hash(str(s))

class Action(object):
    
    def __init__(self, period, order):
        self.period = period
        self.order = order

class PDF(object):
    
    def __init__(self, name):
        self.name = name

    def statetransprob(self, t, current_state, action, demand, next_state):
        '''state transition probabilities'''
        '''given the current state, action, demand, return the probability 
        of choosing next_state as the next state'''
        
        try:
            if self.name == 'general':
                inventory = current_state.inventory + action.order - demand.demand[t - 1]
                inventory = maxElementwise(inventory, np.zeros(len(inventory)))
                if realSequenceEqual(inventory, next_state.inventory, PPM):
                    return 1
                else:
                    return 0
            else:
                raise Exception
        except:
            printErrorAndExit('statetransprob')

class RF(object):
    
    def __init__(self, name, fixed, variable, price, hold, salvage):
        self.name = name
        self.fixed = fixed
        self.variable = variable
        self.price = price
        self.hold = hold
        self.salvage = salvage

    def rewardfunction(self, t, parameterDP, current_state, action, next_state):
        '''given the current state, action, next state, return the reward'''
        
        try:
            if (t != current_state.period + 1) or (t != next_state.period):
                raise Exception
            if self.name == 'linear':
                if t == parameterDP.T + 1:
                    return np.dot(current_state.inventory, self.salvage)
                elif t <= parameterDP.T:
                    r = 0
                    r -= np.dot(action.order, self.variable)
                    if not realNumEqual(r, 0, PPM):
                        r -= self.fixed
                    r += np.dot(current_state.inventory + action.order - next_state.inventory, self.price)
                    r -= np.dot(next_state.inventory, self.hold)
                    return r
                else:
                    raise Exception
            else:
                raise Exception
        except:
            printErrorAndExit('rewardfunction')

class Demand(object):
    
    def __init__(self, name, demand):
        self.name = name
        self.demand = demand

