""" 
    Function:   Define some optional arguments and configurations
"""

import argparse
import time
import os

class opt():
    def __init__(self):
        time_stamp = time.time()
        local_time = time.localtime(time_stamp)
        self.time = time.strftime('%m%d%H%M', local_time)
        
    def dir(self):
        """ Function: Get directories of code, data and experimental results
        """ 
        work_dir = r'.'
        work_dir = os.path.abspath(os.path.expanduser(work_dir))
        dirs = {}

        dirs['data'] = work_dir + '/data'
        dirs['exp'] = work_dir + '/exp'

        # source data
        dirs['sousig_train'] = dirs['data'] + '/LibriSpeech/train-clean-100'
        dirs['sousig_test'] = dirs['data'] + '/LibriSpeech/test-clean'
        dirs['sousig_dev'] = dirs['data'] + '/LibriSpeech/dev-clean'
        # noise data
        dirs['noisig_train'] = dirs['data'] + '/NoiSig/Noise92'
        dirs['noisig_test'] = dirs['data'] + '/NoiSig/Noise92'
        dirs['noisig_dev'] = dirs['data'] + '/NoiSig/Noise92'
        # experimental data
        dirs['sensig_train'] = dirs['data'] + '/train'
        dirs['sensig_test'] = dirs['data'] + '/test'
        dirs['sensig_dev'] = dirs['data'] + '/dev'
        dirs['sensig_locata'] = dirs['data'] + '/LOCATA'
        dirs['log'] = dirs['exp'] + '/' + self.time

        return dirs

if __name__ == '__main__':
    opts = opt()
    args = opts().parse()
    dirs = opts().dir()
    print('gpu-id: ' + str(args.gpu_id))
    print('code path:' + dirs['code'])