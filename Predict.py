import os
from Opt import opt

opts = opt()
args = opts.parse()
dirs = opts.dir()
 
os.environ["OMP_NUM_THREADS"] = str(8) # limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import numpy as np
import torch
import time
import scipy.io
import sys
from tensorboardX import SummaryWriter
import Dataset as at_dataset
import Learner as at_learner
import Model as at_model
import Module as at_module
from Dataset import Parameter
from utils import set_seed, set_random_seed
import math
from torch.utils.data import DataLoader
if __name__ == "__main__":
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	set_seed(args.seed)
	speed = 343.0
	fs = 16000
	T = 4.79 # Trajectory length (s) 
	array_setup = at_dataset.dualch_array_setup
	array_locata_name = 'dicit'
	win_len = 512
	nfft = 512
	win_shift_ratio = 0.5
	fre_used_ratio = 1
	
	seg_fra_ratio = 12 # one estimate per segment (namely seg_fra_ratio frames) 
	seg_len = int(win_len*win_shift_ratio*(seg_fra_ratio+1))
	seg_shift = int(win_len*win_shift_ratio*seg_fra_ratio)

	segmenting = at_dataset.Segmenting_SRPDNN(K=seg_len, step=seg_shift, window=None)

	# %% Network declaration, learner declaration
	tar_useVAD = True
	ch_mode = 'MM' 
	res_the = 37 # Maps resolution (elevation) 
	res_phi = 73 # Maps resolution (azimuth) 

	net = at_model.LSTM_19()
	# from torchsummary import summary
	# summary(net,input_size=(4,256,100),batch_size=55,device="cpu")
	print('# Parameters:', sum(param.numel() for param in net.parameters())/1000000, 'M')

	learner = at_learner.SourceTrackingFromSTFTLearner(net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft, fre_used_ratio=fre_used_ratio,
				nele=res_the, nazi=res_phi, rn=array_setup.mic_pos, fs=fs, ch_mode = ch_mode, tar_useVAD = tar_useVAD, localize_mode = args.localize_mode)
	test_path = ["Your_test_data_path"]
	bsize = 50
	for path in test_path:
				#print(path)
		dataset_test = at_dataset.FixTrajectoryDataset(
			data_dir=path,
            dataset_sz =5000,
            transforms=[segmenting]
            )
		dataloader_test = DataLoader(dataset=dataset_test, # 传入的数据集, 必须参数
            batch_size=bsize,       # 输出的batch大小
            shuffle=False,       # 数据是否打乱
            num_workers=8)                
		print(path)
		print('Test Stage!')
            # Mode selection
		dataset_mode = 'simulate' # use cuda, use amp
                # dataset_mode = 'locata' # use no-cuda
		method_mode = args.localize_mode[0]
		source_num_mode = args.localize_mode[1]

		nmetric = 2
		if len(args.gpu_id)>1:
			learner.mul_gpu()
		if use_cuda:
			learner.cuda()
		else:
			learner.cpu()
		if args.use_amp:
			learner.amp()
		learner.resume_checkpoint(checkpoints_dir='Your_checkpoint_path', from_latest=False)
				#learner.resume_checkpoint(checkpoints_dir='/data/home/wangyabo/ICASSP23/0730_LR_single_metric/exp/08021303', from_latest=False)

		loss_test, metric_test = learner.test_epoch(dataloader_test, return_metric=True)
		print(loss_test, metric_test)




	 