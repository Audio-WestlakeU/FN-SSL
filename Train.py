"""
	Function: Run training and test processes for source source localization on simulated dataset

    Reference: Bing Yang, Hong Liu, and Xiaofei Li, “SRP-DNN: Learning Direct-Path Phase Difference for Multiple Moving Sound Source Localization,” 
	IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 721–725.
	Author:    Bing Yang
    History:   2022-07-01 - Initial version
    Copyright Bing Yang
"""
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

from tensorboardX import SummaryWriter
import Dataset as at_dataset
import Learner as at_learner
import Model as at_model
import Module as at_module
from Dataset import Parameter
from utils import set_seed, set_random_seed
import math
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

	# Room acoustics
	dataset_train = at_dataset.FixTrajectoryDataset(
		data_dir=dirs['sensig_train'],
		dataset_sz = 166816,
		transforms=[segmenting]
	)	
	dataset_dev = at_dataset.FixTrajectoryDataset(
		data_dir=dirs['sensig_dev'],
		dataset_sz = 992,
		transforms=[segmenting]
	)

	# %% Network declaration, learner declaration
	tar_useVAD = True
	ch_mode = 'MM' 
	res_the = 37 # Maps resolution (elevation) 
	res_phi = 73 # Maps resolution (azimuth) 

	net = at_model.FN_SSL()
	# from torchsummary import summary
	# summary(net,input_size=(4,256,100),batch_size=55,device="cpu")
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
	kwargs = {'num_workers': args.workers, 'pin_memory': True}  if use_cuda else {}

	gamma = 0.8988
	print('Training Stage!')

	if args.checkpoint_start:
		learner.resume_checkpoint(checkpoints_dir=dirs['log'], from_latest=True) # Train from latest checkpoints

		# %% TensorboardX
	train_writer = SummaryWriter(dirs['log'] + '/train/', 'train')
	val_writer = SummaryWriter(dirs['log'] + '/val/', 'val')

		# %% Network training
		# trajectories_per_batch = args.train_bz
	lr = args.lr
	nepoch = args.epochs
	dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.bz[0], shuffle=True, num_workers=8)
	dataloader_val = torch.utils.data.DataLoader(dataset=dataset_dev, batch_size=args.bz[1], shuffle=False, num_workers=8)

	for epoch in range(learner.start_epoch, nepoch+1, 1):
		print('\nEpoch {}/{}:'.format(epoch, nepoch))
		print(lr)
		loss_train = learner.train_epoch(dataloader_train, lr=lr, epoch=epoch, return_metric=False)

		loss_val, metric_val = learner.test_epoch(dataloader_val, return_metric=True)
		print('Test loss: {:.4f}, Test ACC: {:.2f}%, Test MAE: {:.2f}deg'.\
				format(loss_val,metric_val['ACC']*100, metric_val['MAE']) )

			# %% Save model
		is_best_epoch = learner.is_best_epoch(current_score=loss_val*(-1))
		learner.save_checkpoint(epoch=epoch, checkpoints_dir=dirs['log'], is_best_epoch=is_best_epoch)

			# %% Visualize parameters with tensorboardX
		train_writer.add_scalar('loss', loss_train, epoch)
		val_writer.add_scalar('loss', loss_val, epoch)
		val_writer.add_scalar('metric-ACC', metric_val['ACC'], epoch)
		val_writer.add_scalar('metric-MAE', metric_val['MAE'], epoch)
			# sys.stdout.flush()
		lr = 0.001 * math.pow(gamma,epoch)
	print('\nTraining finished\n')
