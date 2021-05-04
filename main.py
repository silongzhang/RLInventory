# -*- coding: utf-8 -*-

import utils
import DP
import DQN

def run(InFile, method):
    try:
        if method == 'DP':
            value = DP.DPMethod(InFile)
        elif method == 'DQN':
            value = DQN.DQNMethod(InFile, 1, 1024, int(1e4), int(1e2))
        else:
            raise Exception
        print('value = {}'.format(value))
    except:
        utils.printErrorAndExit('run')

InFile = 'data//instance_1.xlsx'
method = 'DQN'
run(InFile, method)

