import os
import numpy as np
import torch
import torch.optim as optim
import webrtcvad
from copy import deepcopy
from abc import ABC, abstractmethod
from tqdm import tqdm, trange

from utils import sph2cart, cart2sph,forgetting_norm
import Module as at_module


class Learner(ABC):
	""" Abstract class to the routines to train the one source tracking models and perform inferences.
	"""
	def __init__(self, model):
		self.model = model
		# self.cuda_activated = False
		self.max_score = -np.inf
		self.use_amp = False
		self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
		self.start_epoch = 1
		#self.device = device
		super().__init__()

	def mul_gpu(self):
		self.model = torch.nn.DataParallel(self.model) 
		# When multiple gpus are used, 'module.' is added to the name of model parameters. 
		# So whether using one gpu or multiple gpus should be consistent for model traning and checkpoints loading.

	def cuda(self):
		""" Move the model to the GPU and perform the training and inference there.
		"""
		self.model.cuda()
		self.device = "cuda"
		# self.cuda_activated = True

	def cpu(self):
		""" Move the model back to the CPU and perform the training and inference here.
		"""
		self.model.cpu()
		self.device = "cpu"
		# self.cuda_activated = False

	def amp(self):
		""" Use Automatic Mixed Precision to train network.
		"""
		self.use_amp = True
		self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

	@abstractmethod
	def data_preprocess(self, mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None):
		""" To be implemented in each learner according to input of their models
		"""
		pass

	@abstractmethod
	def predgt2DOA(self, pred_batch=None, gt_batch=None):
		"""
	    """
		pass

	def ce_loss(self, pred_batch, gt_batch):
		""" To be implemented in each learner according to output of their models
        """
		pass

	@abstractmethod
	def mse_loss(self, pred_batch, gt_batch):
		""" To be implemented in each learner according to output of their models
        """
		pass
	
	@abstractmethod
	def evaluate(self, pred, gt):
		""" To be implemented in each learner according to output of their models
        """
		pass

	def train_epoch(self, dataset, lr=0.0001, epoch=None, return_metric=False):
		""" Train the model with an epoch of the dataset.
		"""

		avg_loss = 0
		avg_beta = 0.99

		self.model.train() 
		optimizer = optim.Adam(self.model.parameters(), lr=lr)

		loss = 0
		if return_metric: 
			metric = {}

		optimizer.zero_grad()
		pbar = tqdm(enumerate(dataset), total=len(dataset), leave=False)

		for batch_idx, (mic_sig_batch, gt_batch) in pbar:
			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch))

			in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)
			in_batch.requires_grad_()

			with torch.cuda.amp.autocast(enabled=self.use_amp):
				pred_batch = self.model(in_batch)
				loss_batch = self.loss(pred_batch = pred_batch, gt_batch = gt_batch)

			# add up gradients until optimizer.zero_grad(), multiply a scale to gurantee the gradients equal to that when trajectories_per_gpu_call = trajectories_per_batch
			if self.use_amp:
				self.scaler.scale(loss_batch).backward()
				self.scaler.step(optimizer)
				self.scaler.update()
			else:
				loss_batch.backward()
				optimizer.step()

			optimizer.zero_grad()

			avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss_batch.item()
			pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (batch_idx + 1)))
			# pbar.set_postfix(loss=loss.item())
			pbar.update()

			loss += loss_batch.item()

			if return_metric: 
				pred_batch, gt_batch = self.predgt2DOA(pred_batch = pred_batch, gt_batch = gt_batch)
				metric_batch = self.evaluate(pred=pred_batch, gt=gt_batch)
				if batch_idx==0:
					for m in metric_batch.keys():
						metric[m] = 0
				for m in metric_batch.keys():
					metric[m] += metric_batch[m].item()

		loss /= len(pbar)
		if return_metric: 
			for m in metric_batch.keys():
				metric[m] /= len(pbar)

		if return_metric: 
			return loss, metric
		else:
			return loss
	
	def test_epoch(self, dataset, return_metric=False):
		""" Test the model with an epoch of the dataset.
		"""
		self.model.eval()
		with torch.no_grad():
			loss = 0
			idx = 0
			if return_metric: 
				metric = {}


			for mic_sig_batch, gt_batch in dataset:
				print('-----------------------')
				in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)

				with torch.cuda.amp.autocast(enabled=self.use_amp):
					pred_batch = self.model(in_batch)
					loss_batch = self.loss(pred_batch=pred_batch, gt_batch=gt_batch)

				loss += loss_batch.item()

				if return_metric: 
					pred_batch, gt_batch = self.predgt2DOA(pred_batch=pred_batch, gt_batch=gt_batch)
					metric_batch = self.evaluate(pred=pred_batch, gt=gt_batch)
					if idx==0:
						for m in metric_batch.keys():
							metric[m] = 0
					for m in metric_batch.keys():
						metric[m] += metric_batch[m].item()
					idx = idx+1

			loss /= len(dataset)
			if return_metric: 
				for m in metric_batch.keys():
					metric[m] /= len(dataset)

			if return_metric: 
				return loss, metric
			else:
				return loss

	def predict_batch(self, gt_batch, mic_sig_batch, wDNN=True):
		""" 
		Function: Predict 
		Args:
			mic_sig_batch
			gt_batch
		Returns:
			pred_batch		- [DOA, VAD] / [DOA]
			gt_batch		- [DOA, IPD, VAD] / [DOA, VAD]
			mic_sig_batch	- (nb, nsample, nch)
		"""
		self.model.eval()
		with torch.no_grad():
			
			mic_sig_batch = mic_sig_batch.to(self.device)
			in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)

			if wDNN:
				with torch.cuda.amp.autocast(enabled=self.use_amp):
					pred_batch = self.model(in_batch)
				pred_batch, gt_batch = self.predgt2DOA(pred_batch=pred_batch, gt_batch=gt_batch)
			else:
				nt_ori = in_batch.shape[-1]
				nt_pool = gt_batch['doa'].shape[1]
				time_pool_size = int(nt_ori/nt_pool)
				phase = in_batch[:, int(in_batch.shape[1]/2):, :, :].detach() # (nb*nmic_pair, 2, nf, nt)
				phased = phase[:,0,:,:] - phase[:,1,:,:]
				pred_batch = torch.cat((torch.cos(phased), torch.sin(phased)), dim=1).permute(0, 2, 1) # (nb*nmic_pair, nt, 2nf)
				pred_batch, gt_batch = self.predgt2DOA(pred_batch=pred_batch, gt_batch=gt_batch, time_pool_size=time_pool_size)

			return pred_batch, gt_batch, mic_sig_batch


	def predict(self, dataset, wDNN=True, return_predgt=False, metric_setting=None):
		""" 
		Function: Predict 
		Args:
			metric_setting: ae_mode=ae_mode, ae_TH=ae_TH, useVAD=useVAD, vad_TH=vad_TH
		Returns:
			pred		- [DOA, VAD] / [DOA]
			gt			- [DOA, IPD, VAD] / [DOA, VAD]
			mic_sig		- (nb, nsample, nch)
			metric		- [ACC, MDR, FAR, MAE, RMSE]
		"""
		data = []
			
		self.model.eval()
		with torch.no_grad():
			idx = 0
			if return_predgt:
				pred = []
				gt = []
				mic_sig = []
			if metric_setting is not None:
				metric = {}

			for mic_sig_batch, gt_batch in dataset:
				print('Dataloading: ' + str(idx+1))
				# print(mic_sig_batch.shape)
				mic_sig_batch = torch.cat((mic_sig_batch[:,:,8:9], mic_sig_batch[:,:,5:6]), axis=-1)
				pred_batch, gt_batch, mic_sig_batch = self.predict_batch(gt_batch, mic_sig_batch, wDNN)
				# print(mic_sig_batch.shape)

				if (metric_setting is not None):
					metric_batch = self.evaluate(pred=pred_batch, gt=gt_batch)
				if return_predgt:
					pred += [pred_batch]
					gt += [gt_batch]
					mic_sig += [mic_sig_batch]
				if metric_setting is not None:
					for m in metric_batch.keys():
						if idx==0:
							metric[m] = deepcopy(metric_batch[m])
						else:
							metric[m] = torch.cat((metric[m], metric_batch[m]), axis=0)

				idx = idx+1
				
			if return_predgt:
				data += [pred, gt]
				data += [mic_sig]
			if metric_setting is not None:
				data += [metric]
			return data

	def is_best_epoch(self, current_score):
		""" Check if the current model got the best metric score
        """
		if current_score >= self.max_score:
			self.max_score = current_score
			is_best_epoch = True
		else:
			is_best_epoch = False

		return is_best_epoch

	def save_checkpoint(self, epoch, checkpoints_dir, is_best_epoch = False):
		""" Save checkpoint to "checkpoints_dir" directory, which consists of:
            - the epoch number
            - the best metric score in history
            - the optimizer parameters
            - the model parameters
        """
        
		print(f"\t Saving {epoch} epoch model checkpoint...")
		if self.use_amp:
			state_dict = {
				"epoch": epoch,
				"max_score": self.max_score,
				# "optimizer": self.optimizer.state_dict(),
				"scalar": self.scaler.state_dict(), 
				"model": self.model.state_dict()
			}
		else:
			state_dict = {
				"epoch": epoch,
				"max_score": self.max_score,
				# "optimizer": self.optimizer.state_dict(),
				"model": self.model.state_dict()
			}

		torch.save(state_dict, checkpoints_dir + "/latest_model.tar")
		torch.save(state_dict, checkpoints_dir + "/model"+str(epoch)+".tar")

		if is_best_epoch:
			print(f"\t Found a max score in the {epoch} epoch, saving...")
			torch.save(state_dict, checkpoints_dir + "/best_model.tar")


	def resume_checkpoint(self, checkpoints_dir, from_latest = True):
		"""Resume from the latest/best checkpoint.
		"""

		if from_latest:

			latest_model_path = checkpoints_dir + "/lightning.ckpt"

			assert os.path.exists(latest_model_path), f"{latest_model_path} does not exist, can not load latest checkpoint."

			# self.dist.barrier()  # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work

			# device = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
			checkpoint = torch.load(latest_model_path, map_location=self.device)

			#self.start_epoch = checkpoint["epoch"] + 1
			#self.max_score = checkpoint["max_score"]
			# self.optimizer.load_state_dict(checkpoint["optimizer"])
			if self.use_amp:
				self.scaler.load_state_dict(checkpoint["scalar"])
			self.model.load_state_dict(checkpoint["state_dict"])

			# if self.rank == 0:
			print(f"Model checkpoint loaded. Training will begin at {self.start_epoch} epoch.")

		else:
			best_model_path = checkpoints_dir + "/best_model.tar"

			assert os.path.exists(best_model_path), f"{best_model_path} does not exist, can not load best model."

			# self.dist.barrier()  # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work

			# device = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
			checkpoint = torch.load(best_model_path, map_location=self.device)

			self.model.load_state_dict(checkpoint["model"])




class SourceTrackingFromSTFTLearner(Learner):
	""" Learner for models which use STFTs of multiple channels as input
	"""
	def __init__(self, model, win_len, win_shift_ratio, nfft, fre_used_ratio, nele, nazi, rn, fs, ch_mode, tar_useVAD, localize_mode, c=343.0): #, arrayType='planar', cat_maxCoor=False, apply_vad=False):
		""" 
		fre_used_ratio - the ratio between used frequency and valid frequency
		"""
		super().__init__(model)

		self.nele = nele
		self.nazi = nazi

		self.nfft = nfft
		#self.nf_used = int(self.nfft/2*fre_used_ratio)
		if fre_used_ratio == 1:
			self.fre_range_used = range(1, int(self.nfft/2*fre_used_ratio)+1, 1)
		elif fre_used_ratio == 0.5:
			self.fre_range_used = range(0, int(self.nfft/2*fre_used_ratio), 1)
		else:
			raise Exception('Prameter fre_used_ratio unexpected')

		# self.nf_used = int((self.nfft / 2 +1)* fre_used_ratio)
		self.dostft = at_module.STFT(win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft)
		fre_max = fs / 2
		self.ch_mode = ch_mode
		self.gerdpipd = at_module.DPIPD(ndoa_candidate=[nele, nazi], mic_location=rn, nf=int(self.nfft/2) + 1, fre_max=fre_max, 
										ch_mode=self.ch_mode, speed=c)
		self.tar_useVAD = tar_useVAD
		self.addbatch = at_module.AddChToBatch(ch_mode=self.ch_mode)
		self.removebatch = at_module.RemoveChFromBatch(ch_mode=self.ch_mode)
		self.sourcelocalize = at_module.SourceDetectLocalize(max_num_sources=int(localize_mode[2]), source_num_mode=localize_mode[1], meth_mode=localize_mode[0])
		
		self.getmetric = at_module.getMetric(source_mode='single')

	def data_preprocess(self, mic_sig_batch=None, gt_batch=None, vad_batch=None, eps=1e-6, nor_flag=True):

		data = []
		if mic_sig_batch is not None:
			mic_sig_batch = mic_sig_batch.to(self.device)
			
			stft = self.dostft(signal=mic_sig_batch) # (nb,nf,nt,nch)
			stft = stft.permute(0, 3, 1, 2)  # (nb,nch,nf,nt)

			# change batch (nb,nch,nf,nt)→(nb*(nch-1),2,nf,nt)/(nb*(nch-1)*nch/2,2,nf,nt)
			stft_rebatch = self.addbatch(stft)
			if nor_flag:
				nb, nc, nf, nt = stft_rebatch.shape
				mag = torch.abs(stft_rebatch)
				mean_value = forgetting_norm(mag)
				stft_rebatch_real = torch.real(stft_rebatch) / (mean_value + eps)
				stft_rebatch_image = torch.imag(stft_rebatch) / (mean_value + eps)				
			else:
				stft_rebatch_real = torch.real(stft_rebatch)
				stft_rebatch_image = torch.imag(stft_rebatch)
			# prepare model input
			real_image_batch  =  torch.cat((stft_rebatch_real,stft_rebatch_image),dim=1)
			data += [real_image_batch[:,:,self.fre_range_used,:]]

		if gt_batch is not None:
			DOAw_batch = gt_batch['doa']
			vad_batch = gt_batch['vad_sources']

			source_doa = DOAw_batch.cpu().numpy()  
			
			if self.ch_mode == 'M':
				_, ipd_batch,_ = self.gerdpipd(source_doa=source_doa)
			elif self.ch_mode == 'MM':
				_, ipd_batch,_ = self.gerdpipd(source_doa=source_doa)
			ipd_batch = np.concatenate((ipd_batch.real[:,:,self.fre_range_used,:,:], ipd_batch.imag[:,:,self.fre_range_used,:,:]), axis=2).astype(np.float32) # (nb, ntime, 2nf, nmic-1, nsource)
			ipd_batch = torch.from_numpy(ipd_batch)

			vad_batch = vad_batch.mean(axis=2).float() # (nb,nseg,nsource) # s>2/3 

			# DOAw_batch = torch.from_numpy(source_doa).to(self.device) 
			DOAw_batch = DOAw_batch.to(self.device) # (nb,nseg,2,nsource)
			ipd_batch = ipd_batch.to(self.device)
			vad_batch = vad_batch.to(self.device)

			if self.tar_useVAD:
				nb, nt, nf, nmic, num_source = ipd_batch.shape
				th = 0
				vad_batch_copy = deepcopy(vad_batch)
				vad_batch_copy[vad_batch_copy<=th] = th
				vad_batch_copy[vad_batch_copy>0] = 1
				vad_batch_expand = vad_batch_copy[:, :, np.newaxis, np.newaxis, :].expand(nb, nt, nf, nmic, num_source)
				ipd_batch = ipd_batch * vad_batch_expand
			ipd_batch = torch.sum(ipd_batch, dim=-1)  # (nb,nseg,2nf,nmic-1)

			gt_batch['doa'] = DOAw_batch
			gt_batch['ipd'] = ipd_batch
			gt_batch['vad_sources'] = vad_batch
			
			data += [gt_batch]

		return data # [Input, DOA, IPD, VAD]

	def ce_loss(self, pred_batch=None, gt_batch=None):
		""" 
		Function: ce loss
		Args:
			pred_batch: doa
			gt_batch: dict{'doa'}
		Returns:
			loss
        """
		pred_doa = pred_batch
		gt_doa = gt_batch['doa'] * 180 / np.pi
		gt_doa = gt_doa[:,:,1,:].type(torch.LongTensor).to(self.device)
		nb,nt,_ = pred_doa.shape
		pred_doa = pred_doa.to(self.device)
		loss = torch.nn.functional.cross_entropy(pred_doa.reshape(nb*nt,-1),gt_doa.reshape(nb*nt))
		return loss
	def mse_loss(self, pred_batch=None, gt_batch=None):
		""" 
		Function: mse loss
		Args:
			pred_batch: ipd
			gt_batch: dict{'ipd'}
		Returns:
			loss
        """
		pred_ipd = pred_batch
		gt_ipd = gt_batch['ipd']
		nb, _, _, _ = gt_ipd.shape # (nb, nt, nf, nmic)

		pred_ipd_rebatch = self.removebatch(pred_ipd, nb).permute(0, 2, 3, 1)

		loss = torch.nn.functional.mse_loss(pred_ipd_rebatch.contiguous(), gt_ipd.contiguous())

		return loss		
	def predgt2DOA_cls(self, pred_batch=None, gt_batch=None):
		""" 
		Function: pred to doa of classification
		Args:
			pred_batch: doa classification
		Returns:
			loss
        """		
		if pred_batch is not None:
			pred_batch = pred_batch.detach()
			DOA_batch_pred = torch.argmax(pred_batch,dim=-1) # distance = 1 (nb, nt, 2)    
			pred_batch = [DOA_batch_pred[:, :, np.newaxis, np.newaxis]]  # !! only for single source
		return pred_batch, gt_batch

	def predgt2DOA(self, pred_batch=None, gt_batch=None, time_pool_size=None):
		"""
		Function: Conert IPD vector to DOA
		Args:
			pred_batch: ipd
			gt_batch: dict{'doa', 'vad_sources', 'ipd'}
		Returns:
			pred_batch: dict{'doa', 'spatial_spectrum'}
			gt_batch: dict{'doa', 'vad_sources', 'ipd'}
	    """

		if pred_batch is not None:
			
			pred_ipd = pred_batch.detach()
			dpipd_template, _, doa_candidate = self.gerdpipd( ) # (nele, nazi, nf, nmic)

			_, _, _, nmic = dpipd_template.shape
			nbnmic, nt, nf = pred_ipd.shape
			nb = int(nbnmic/nmic)

			dpipd_template = np.concatenate((dpipd_template.real[:,:,self.fre_range_used,:], dpipd_template.imag[:,:,self.fre_range_used,:]), axis=2).astype(np.float32) # (nele, nazi, 2nf, nmic-1)
			dpipd_template = torch.from_numpy(dpipd_template).to(self.device) # (nele, nazi, 2nf, nmic)

			# !!!
			nele, nazi, _, _ = dpipd_template.shape
			dpipd_template = dpipd_template[int((nele-1)/2):int((nele-1)/2)+1, int((nazi-1)/2):nazi, :, :]
			doa_candidate[0] = np.linspace(np.pi/2, np.pi/2, 1)
			doa_candidate[1] = np.linspace(0, np.pi, 37)
			# doa_candidate[0] = doa_candidate[0][int((nele-1)/2):int((nele-1)/2)+1]
			# doa_candidate[1] = doa_candidate[1][int((nazi-1)/2):nazi]

			# rebatch from (nb*nmic, nt, 2nf) to (nb, nt, 2nf, nmic)
			pred_ipd_rebatch = self.removebatch(pred_ipd, nb).permute(0, 2, 3, 1) # (nb, nt, 2nf, nmic)
			if time_pool_size is not None:
				nt_pool = int(nt / time_pool_size)
				ipd_pool_rebatch = torch.zeros((nb, nt_pool, nf, nmic), dtype=torch.float32, requires_grad=False).to(self.device)  # (nb, nt_pool, 2nf, nmic-1)
				for t_idx in range(nt_pool):
					ipd_pool_rebatch[:, t_idx, :, :]  = torch.mean(
					pred_ipd_rebatch[:, t_idx*time_pool_size: (t_idx+1)*time_pool_size, :, :], dim=1)
				pred_ipd_rebatch = deepcopy(ipd_pool_rebatch)
				nt = deepcopy(nt_pool)
			
			pred_DOAs, pred_VADs, pred_ss = self.sourcelocalize(pred_ipd=pred_ipd_rebatch, dpipd_template=dpipd_template, doa_candidate=doa_candidate)
			pred_batch = {}
			pred_batch['doa'] = pred_DOAs
			pred_batch['vad_sources'] = pred_VADs
			pred_batch['spatial_spectrum'] = pred_ss

		if gt_batch is not None: 
			for key in gt_batch.keys():
				gt_batch[key] = gt_batch[key].detach()

		return pred_batch, gt_batch 

	def evaluate(self, pred, gt, metric_setting={'ae_mode':['azi'], 'ae_TH':5, 'useVAD':True, 'vad_TH':[2/3, 2/3], 'metric_unfold':False} ):
		""" 
		Function: Evaluate DOA estimation results
		Args:
			pred 	- dict{'doa', 'vad_sources'}
			gt 		- dict{'doa', 'vad_sources'}
							doa (nb, nt, 2, nsources) in radians
							vad (nb, nt, nsources) binary values
		Returns:
			metric
        """
		doa_gt = gt['doa'] * 180 / np.pi 
		doa_pred = pred['doa'] * 180 / np.pi 
		vad_gt = gt['vad_sources']  
		vad_pred = pred['vad_sources'] 

		# single source 
		# metric = self.getmetric(doa_gt, vad_gt, doa_pred, vad_pred, ae_mode = ae_mode, ae_TH=ae_TH, useVAD=False, vad_TH=vad_TH, metric_unfold=Falsemetric_unfold)

		# multiple source
		metric = \
			self.getmetric(doa_gt, vad_gt, doa_pred, vad_pred, 
				ae_mode = metric_setting['ae_mode'], ae_TH=metric_setting['ae_TH'], 
				useVAD=metric_setting['useVAD'], vad_TH=metric_setting['vad_TH'], 
				metric_unfold=metric_setting['metric_unfold'])
		return metric
