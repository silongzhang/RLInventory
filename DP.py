# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import utils

def setofaction(t, parameterDP, current_state):
    '''return the set of actions which can be adopted according to 
    capacity and the current state current_state'''
    
    try:
        if current_state.period != t - 1:
            raise Exception
    except:
        utils.printErrorAndExit('setofaction')
    
    temp = np.array(parameterDP.MaxInventory) - np.array(current_state.inventory)
    limit = utils.minElementwise(parameterDP.MaxOrder, temp.tolist())
    order = []
    partial = np.zeros(len(limit))
    utils.enumerate(order, limit, partial, 0)
    
    result = []
    for elem in order:
        result.append(utils.Action(t, elem))
    return result

def setofstate(t, pdf, current_state, action, demand):
    '''given the current state current_state and the action, return the 
    set of next states'''

    try:
        if pdf.name == 'general':
            inventory = current_state.inventory + action.order - demand.demand[t - 1]
            inventory = utils.maxElementwise(inventory, np.zeros(len(inventory)))
            result = []
            result.append(utils.State(t, inventory))
            return result
        else:
            raise Exception
    except:
        utils.printErrorAndExit('setofstate')

def statespace(t, parameterDP):
    '''the set of states'''
    
    inventory = []
    limit = parameterDP.MaxInventory
    partial = np.zeros(parameterDP.N)
    utils.enumerate(inventory, limit, partial, 0)
    
    result = []
    for elem in inventory:
        result.append(utils.State(t, elem))
    return result

def actionvalue(t, parameterDP, pdf, rf, current_state, action, value_pre, demand):
    '''action value function'''
    
    value = 0
    states = setofstate(t, pdf, current_state, action, demand)
    for i in range(0, len(states)):
        next_state = states[i]
        value += pdf.statetransprob(t, current_state, action, demand, next_state) * \
        (rf.rewardfunction(t, parameterDP, current_state, action, next_state) + \
         value_pre[next_state.hash()])

    return value

def DPAlgorithmCore(parameterDP, pdf, rf, demand):
    # the optimal state value function
    value_pre = {}
    value_suc = {}

    # initialization
    print('initialization')
    for state in statespace(parameterDP.T, parameterDP):
        value_pre[state.hash()] = rf.rewardfunction(parameterDP.T + 1, parameterDP, \
        state, utils.Action(parameterDP.T + 1, np.zeros(parameterDP.N)), \
        utils.State(parameterDP.T + 1, np.zeros(parameterDP.N)))

    for t in range(parameterDP.T, 0, -1):
        print('t = {}'.format(t))
        for current_state in statespace(t - 1, parameterDP):
            value_suc[current_state.hash()] = 0
            for action in setofaction(t, parameterDP, current_state):
                value_suc[current_state.hash()] = max(value_suc[current_state.hash()], \
                actionvalue(t, parameterDP, pdf, rf, current_state, action, value_pre, demand))
        value_pre = value_suc
        value_suc = {}

    return value_pre[utils.State(0, np.zeros(parameterDP.N)).hash()]

def readData(InFile):
    df = pd.read_excel(InFile)
    T = int(df['T'][0])
    N = int(df['N'][0])
    MaxInventory = df['MaxInventory'].astype(np.int64).tolist()
    MaxOrder = df['MaxOrder'].astype(np.int64).tolist()
    parameterDP = utils.ParameterDP(T, N, MaxInventory, MaxOrder)
    
    namePDF = df['pdf'][0]
    pdf = utils.PDF(namePDF)
    
    nameRF = df['rf'][0]
    fixed = df['fixed'][0]
    variable = df['variable'].astype(np.float64).tolist()
    price = df['price'].astype(np.float64).tolist()
    hold = df['hold'].astype(np.float64).tolist()
    salvage = df['salvage'].astype(np.float64).tolist()
    rf = utils.RF(nameRF, fixed, variable, price, hold, salvage)
    
    nameDemand = df['demand'][0]
    demands = []
    demands.append(df['demand_1'].astype(np.int64).tolist())
    demands.append(df['demand_2'].astype(np.int64).tolist())
    demands.append(df['demand_3'].astype(np.int64).tolist())
    demands.append(df['demand_4'].astype(np.int64).tolist())
    demands.append(df['demand_5'].astype(np.int64).tolist())
    demands.append(df['demand_6'].astype(np.int64).tolist())
    demands.append(df['demand_7'].astype(np.int64).tolist())
    demand = utils.Demand(nameDemand, demands)
    
    return parameterDP, pdf, rf, demand

def DPMethod(InFile):
    parameterDP, pdf, rf, demand = readData(InFile)
    return DPAlgorithmCore(parameterDP, pdf, rf, demand)

