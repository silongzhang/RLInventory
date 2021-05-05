# -*- coding: utf-8 -*-

import sys
import time
import utils
import DP
import DQN

def main(argv):
    try:
        if argv[1] == 'DP':
            value = DP.DPMethod(argv[2])
        elif argv[1] == 'DQN':
            # InFile, batch_size, buffer_size, episodes_train, episodes_test, startTime
            value = DQN.DQNMethod(argv[2], int(argv[3]), int(argv[4]), int(argv[5]), int(argv[6]), time.process_time())
        else:
            raise Exception
        print('value = {}'.format(value))
    except:
        utils.printErrorAndExit('main')

if __name__ == '__main__':
    main(sys.argv)

# example: python main.py DQN data//instance_1.xlsx 8 1024 10000 100
