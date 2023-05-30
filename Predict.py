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
from utils import set_seed, set_random_seed,locata_plot
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
	res_phi = 73 # Maps resolution (azimuth) 73/5°；

	net = at_model.LSTM_19()

	print('# Parameters:', sum(param.numel() for param in net.parameters())/1000000, 'M')

	learner = at_learner.SourceTrackingFromSTFTLearner(net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft, fre_used_ratio=fre_used_ratio,
				nele=res_the, nazi=res_phi, rn=array_setup.mic_pos, fs=fs, ch_mode = ch_mode, tar_useVAD = tar_useVAD, localize_mode = args.localize_mode)
	
	
	if len(args.gpu_id)>1:
		learner.mul_gpu()
	if use_cuda:
		learner.cuda()
	else:
		learner.cpu()
	if args.use_amp:
		learner.amp()
	learner.resume_checkpoint(checkpoints_dir='/exp/04231627/', from_latest=False)
	
	dataset_mode = 'simulate'
	if dataset_mode == 'simulate':
		test_path = [dirs['sensig_test']]
		bsize = 50
		for path in test_path:
			dataset_test = at_dataset.FixTrajectoryDataset(
				data_dir=path,
				dataset_sz =5000,
				transforms=[segmenting]
				)
			dataloader_test = DataLoader(dataset=dataset_test, 
				batch_size=bsize,       
				shuffle=False,       
				num_workers=8)                
			print('Test Stage!')
				# Mode selection
			loss_test, metric_test = learner.test_epoch(dataloader_test, return_metric=True)
			print(loss_test, metric_test)

	elif dataset_mode == 'locata':
		array_locata_name = 'dicit'
		tasks = ((3,5), )
		path_locata = (dirs['sensig_locata'] + '/eval',dirs['sensig_locata'] + '/dev')
		ntask = len(tasks)
		dataset_locata = at_dataset.LocataDataset(path_locata, array_locata_name, fs, dev=True, tasks=tasks[0], transforms=[segmenting])
		nins = len(dataset_locata)
		nmetric = 2
		metrics = np.zeros((nmetric, nins, ntask))
		metric_setting = {'ae_mode':['azi'], 'ae_TH':30, 'useVAD':True, 'vad_TH':[2/3, 0.2], 'metric_unfold':False}
		save_file = True
		for task in tasks:
			task_idx = tasks.index(task)
			t_start = time.time()
			dataset_locata = at_dataset.LocataDataset(path_locata, array_locata_name, fs, dev=True, tasks=task, transforms=[segmenting])
			dataloader = DataLoader(dataset=dataset_locata, batch_size=1, shuffle=False)			
			pred, gt, _,metric = learner.predict(dataloader, return_predgt=True, metric_setting=metric_setting, wDNN=True,save_file=save_file)
			print(torch.mean(metric['MAE']),torch.mean(metric['ACC']))
		locata_plot(result_path='./locata_result/', save_fig_path='./locata_result/')
