import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import permutations
from scipy.optimize import linear_sum_assignment

# %% Complex number operations

def complex_multiplication(x, y):
	return torch.stack([ x[...,0]*y[...,0] - x[...,1]*y[...,1],   x[...,0]*y[...,1] + x[...,1]*y[...,0]  ], dim=-1)


def complex_conjugate_multiplication(x, y):
	return torch.stack([ x[...,0]*y[...,0] + x[...,1]*y[...,1],   x[...,1]*y[...,0] - x[...,0]*y[...,1]  ], dim=-1)


def complex_cart2polar(x):
	mod = torch.sqrt( complex_conjugate_multiplication(x, x)[..., 0] )
	phase = torch.atan2(x[..., 1], x[..., 0])
	return torch.stack((mod, phase), dim=-1)


# %% Signal processing and DOA estimation layers

class STFT(nn.Module):
	""" Function: Get STFT coefficients of microphone signals (batch processing by pytorch)
        Args:       win_len         - the length of frame / window
                    win_shift_ratio - the ratio between frame shift and frame length
                    nfft            - the number of fft points
                    win             - window type 
                                    'boxcar': a rectangular window (equivalent to no window at all)
                                    'hann': a Hann window
					signal          - the microphone signals in time domain (nbatch, nsample, nch)
        Returns:    stft            - STFT coefficients (nbatch, nf, nt, nch)
    """

	def __init__(self, win_len, win_shift_ratio, nfft, win='hann'):
		super(STFT, self).__init__()

		self.win_len = win_len
		self.win_shift_ratio = win_shift_ratio
		self.nfft = nfft
		self.win = win

	def forward(self, signal):

		nsample = signal.shape[-2]
		nch = signal.shape[-1]
		win_shift = int(self.win_len * self.win_shift_ratio)
		nf = int(self.nfft / 2) + 1

		nb = signal.shape[0]
		nt = np.floor((nsample - self.win_len) / win_shift + 1).astype(int)
		# nt = int((nsample) / win_shift) + 1  # for iSTFT
		stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64).to(signal.device)

		if self.win == 'hann':
			window = torch.hann_window(window_length=self.win_len, device=signal.device)
		for ch_idx in range(0, nch, 1):
			stft[:, :, :, ch_idx] = torch.stft(signal[:, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, win_length=self.win_len,
								   window=window, center=False, normalized=False, return_complex=True)
			# stft[:, :, :, ch_idx] = torch.stft(signal[:, :, ch_idx], n_fft = nfft, hop_length = win_shift, win_length = win_len,
                                #    window = window, center = True, normalized = False, return_complex = True)  # for iSTFT

		return stft

class ISTFT(nn.Module):
	""" Function: Get inverse STFT (batch processing by pytorch) 
		Args:		stft            - STFT coefficients (nbatch, nf, nt, nch)
					win_len         - the length of frame / window
					win_shift_ratio - the ratio between frame shift and frame length
					nfft            - the number of fft points
		Returns:	signal          - time-domain microphone signals (nbatch, nsample, nch)
	"""
	def __init__(self, win_len, win_shift_ratio, nfft):
		super(ISTFT, self).__init__()
		
		self.win_len = win_len
		self.win_shift_ratio = win_shift_ratio
		self.nfft = nfft

	def forward(self, stft):

		nf = stft.shape[-3]
		nt = stft.shape[-2]
		nch = stft.shape[-1]
		nb = stft.shape[0]
		win_shift = int(self.win_len * self.win_shift_ratio)
		nsample = (nt - 1) * win_shift
		signal = torch.zeros((nb, nsample, nch)).to(stft.device)
		for ch_idx in range(0, nch, 1):
			signal_temp = torch.istft(stft[:, :, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, win_length=self.win_len,
                                        center=True, normalized=False, return_complex=False)
			signal[:, :, ch_idx] = signal_temp[:, 0:nsample]

		return signal

class getMetric(nn.Module):
	"""  
	Call: 
	# single source 
	getmetric = at_module.getMetric(source_mode='single', metric_unfold=True)
	metric = self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode=['azi,'ele'], ae_TH=30, useVAD=False, vad_TH=vad_TH)
	# multiple source
	self.getmetric = getMetric(source_mode='multiple', metric_unfold=True)
	metric = self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode=['azi,'ele'], ae_TH=30, useVAD=False, vad_TH=[2/3, 0.2]])
	"""
	def __init__(self, source_mode='multiple', large_number=10000, invalid_source_idx=10, eps=+1e-5):
		"""
		Args:
			source_mode	- 'single', 'multiple'
		"""
		super(getMetric, self).__init__()

		# self.ae_mode = ae_mode
		# self.ae_TH = ae_TH
		# self.useVAD = useVAD
		self.source_mode = source_mode
		self.inf = large_number
		self.invlid_sidx = invalid_source_idx
		self.eps = eps

	def forward(self, doa_gt, vad_gt, doa_est, vad_est, ae_mode, ae_TH=30, useVAD=True, vad_TH=[2/3,2/3], metric_unfold=False):
		"""
		Args:
			doa_gt, doa_est - (nb, nt, 2, ns) in degrees
			vad_gt, vad_est - (nb, nt, ns) 
			ae_mode 		- angle error mode, [*, *, *], * - 'azi', 'ele', 'aziele' 
			ae_TH			- angle error threshold, namely azimuth error threshold in degrees
			vad_TH 			- VAD threshold, [gtVAD_TH, estVAD_TH] 
		Returns:
			ACC, MAE or ACC, MD, FA, MAE, RMSE - [*, *, *]
		"""
		device = doa_gt.device
		# doa_gt = doa_gt * 180 / np.pi
		# doa_est = doa_est * 180 / np.pi
		if self.source_mode == 'single':

			nbatch, nt, naziele, nsources = doa_est.shape
			if useVAD == False:
				vad_gt = torch.ones((nbatch, nt, nsources)).to(device)
				vad_est = torch.ones((nbatch,nt, nsources)).to(device)
			else:
				vad_gt = vad_gt > vad_TH[0]
				vad_est = vad_est > vad_TH[1]
			vad_est = vad_est * vad_gt

			azi_error = self.angular_error(doa_est[:,:,1,:], doa_gt[:,:,1,:], 'azi')            
			ele_error = self.angular_error(doa_est[:,:,0,:], doa_gt[:,:,0,:], 'ele')
			aziele_error = self.angular_error(doa_est.permute(2,0,1,3), doa_gt.permute(2,0,1,3), 'aziele')
			
			corr_flag = ((azi_error < ae_TH)+0.0) * vad_est # Accorrding to azimuth error
			act_flag = 1*vad_gt
			K_corr = torch.sum(corr_flag) 
			print(torch.sum(corr_flag),torch.sum(act_flag)) 
			ACC = torch.sum(corr_flag) / torch.sum(act_flag)
			MAE = []
			if 'ele' in ae_mode:
				MAE += [torch.sum(vad_gt * ele_error) / torch.sum(act_flag)]
			if 'azi' in ae_mode:
				MAE += [ torch.sum(vad_gt * azi_error) / torch.sum(act_flag)]
				# MAE += [torch.sum(corr_flag * azi_error) / torch.sum(act_flag)]
			if 'aziele' in ae_mode:
				MAE += [torch.sum(vad_gt * aziele_error) / torch.sum(act_flag)]

			MAE = torch.tensor(MAE)
			metric = {}
			metric['ACC'] = ACC
			metric['MAE'] = MAE
			# metric = [ACC, MAE]

			if metric_unfold:
				metric, key_list = self.unfold_metric(metric)
				return metric, key_list
			else:
				return metric

		elif self.source_mode == 'multiple':
			nbatch = doa_est.shape[0]
			nmode = len(ae_mode)
			acc = torch.zeros(nbatch, 1)
			mdr = torch.zeros(nbatch, 1)
			far = torch.zeros(nbatch, 1)
			mae = torch.zeros(nbatch, nmode)
			rmse = torch.zeros(nbatch, nmode)
			for b_idx in range(nbatch):
				doa_gt_one = doa_gt[b_idx, ...]
				doa_est_one = doa_est[b_idx, ...]
				
				nt = doa_gt_one.shape[0]
				num_sources_gt = doa_gt_one.shape[2]
				num_sources_est = doa_est_one.shape[2]

				if useVAD == False:
					vad_gt_one = torch.ones((nt, num_sources_gt)).to(device)
					vad_est_one = torch.ones((nt, num_sources_est)).to(device)
				else:
					vad_gt_one = vad_gt[b_idx, ...]
					vad_est_one = vad_est[b_idx, ...]
					vad_gt_one = vad_gt_one > vad_TH[0]
					vad_est_one = vad_est_one > vad_TH[1]

				corr_flag = torch.zeros((nt, num_sources_gt)).to(device)
				azi_error = torch.zeros((nt, num_sources_gt)).to(device)
				ele_error = torch.zeros((nt, num_sources_gt)).to(device)
				aziele_error = torch.zeros((nt, num_sources_gt)).to(device)
				K_gt = vad_gt_one.sum(axis=1)
				vad_gt_sum = torch.reshape(vad_gt_one.sum(axis=1)>0, (nt, 1)).repeat((1, num_sources_est))
				vad_est_one = vad_est_one * vad_gt_sum
				K_est = vad_est_one.sum(axis=1)
				for t_idx in range(nt):
					num_gt = int(K_gt[t_idx].item())
					num_est = int(K_est[t_idx].item())
					if num_gt>0 and num_est>0:
						est = doa_est_one[t_idx, :, vad_est_one[t_idx,:]>0]
						gt = doa_gt_one[t_idx, :, vad_gt_one[t_idx,:]>0]
						dist_mat_az = torch.zeros((num_gt, num_est))
						dist_mat_el = torch.zeros((num_gt, num_est))
						dist_mat_azel = torch.zeros((num_gt, num_est))
						for gt_idx in range(num_gt):
							for est_idx in range(num_est):
								dist_mat_az[gt_idx, est_idx] = self.angular_error(est[1,est_idx], gt[1,gt_idx], 'azi')
								dist_mat_el[gt_idx, est_idx] = self.angular_error(est[0,est_idx], gt[0,gt_idx], 'ele')
								dist_mat_azel[gt_idx, est_idx] = self.angular_error(est[:,est_idx], gt[:,gt_idx], 'aziele')
						
						invalid_assigns = dist_mat_az > ae_TH  # Accorrding to azimuth error
						# 	invalid_assigns = dist_mat_el > ae_TH
						# 	invalid_assigns = dist_mat_azel > ae_TH
						
						dist_mat_az_bak = dist_mat_az.clone()
						dist_mat_az_bak[invalid_assigns] = self.inf
						assignment = list(linear_sum_assignment(dist_mat_az_bak))
						assignment = self.judge_assignment(dist_mat_az_bak, assignment)
						for src_idx in range(num_gt):
							if assignment[src_idx] != self.invlid_sidx:
								corr_flag[t_idx, src_idx] = 1
								azi_error[t_idx, src_idx] = dist_mat_az[src_idx, assignment[src_idx]]
								ele_error[t_idx, src_idx] = dist_mat_el[src_idx, assignment[src_idx]]
								aziele_error[t_idx, src_idx] = dist_mat_azel[src_idx, assignment[src_idx]]

				K_corr = corr_flag.sum(axis=1)
				acc[b_idx, :] = K_corr.sum(axis=0) / K_gt.sum(axis=0)
				mdr[b_idx, :] = (K_gt.sum(axis=0) - K_corr.sum(axis=0)) / K_gt.sum(axis=0)
				far[b_idx, :] = (K_est.sum(axis=0) - K_corr.sum(axis=0)) / K_gt.sum(axis=0)

				mae_temp = []
				rmse_temp = []
				if 'ele' in ae_mode:
					mae_temp += [((ele_error*corr_flag).sum(axis=0)).sum() / (K_corr.sum(axis=0)+self.eps)]
					rmse_temp += [torch.sqrt(((ele_error*ele_error*corr_flag).sum(axis=0)).sum() / (K_corr.sum(axis=0)+self.eps))]
				if 'azi' in ae_mode:
					mae_temp += [((azi_error*corr_flag).sum(axis=0)).sum() / (K_corr.sum(axis=0)+self.eps)]
					rmse_temp += [torch.sqrt(((azi_error*azi_error*corr_flag).sum(axis=0)).sum() / (K_corr.sum(axis=0)+self.eps))]
				if 'aziele' in ae_mode:
					mae_temp += [((aziele_error*corr_flag).sum(axis=0)).sum() / (K_corr.sum(axis=0)+self.eps)]
					rmse_temp += [torch.sqrt(((aziele_error*aziele_error*corr_flag).sum(axis=0)).sum() / (K_corr.sum(axis=0)+self.eps))]

				mae[b_idx, :] = torch.tensor(mae_temp)
				rmse[b_idx, :] = torch.tensor(rmse_temp)
			
			metric = {}
			metric['ACC'] = torch.mean(acc, dim=0)
			metric['MDR'] = torch.mean(mdr, dim=0)
			metric['FAR'] = torch.mean(far, dim=0)
			metric['MAE'] = torch.mean(mae, dim=0)
			metric['RMSE'] = torch.mean(rmse, dim=0)

			if metric_unfold:
				metric, key_list  = self.unfold_metric(metric)
				return metric
			else:
				return metric

	def judge_assignment(self, dist_mat, assignment):
		final_assignment = torch.tensor([self.invlid_sidx for i in range(dist_mat.shape[0])])
		for i in range(min(dist_mat.shape[0],dist_mat.shape[1])):
			if dist_mat[assignment[0][i], assignment[1][i]] != self.inf:
				final_assignment[assignment[0][i]] = assignment[1][i]
			else:
				final_assignment[i] = self.invlid_sidx
		return final_assignment

	def angular_error(self, est, gt, ae_mode):
		"""
		Function: return angular error in degrees
		"""
		if ae_mode == 'azi':
			ae = torch.abs((est-gt+180)%360 - 180)
		elif ae_mode == 'ele':
			ae = torch.abs(est-gt)
		elif ae_mode == 'aziele':
			ele_gt = gt[0, ...].float() / 180 * np.pi
			azi_gt = gt[1, ...].float() / 180 * np.pi
			ele_est = est[0, ...].float() / 180 * np.pi
			azi_est = est[1, ...].float() / 180 * np.pi
			aux = torch.cos(ele_gt) * torch.cos(ele_est) + torch.sin(ele_gt) * torch.sin(ele_est) * torch.cos(azi_gt - azi_est)
			aux[aux.gt(0.99999)] = 0.99999
			aux[aux.lt(-0.99999)] = -0.99999
			ae = torch.abs(torch.acos(aux)) * 180 / np.pi
		else:
			raise Exception('Angle error mode unrecognized')
		
		return ae

	def unfold_metric(self, metric):
		metric_unfold = []
		for m in metric.keys():
			if metric[m].numel() !=1:
				for n in range(metric[m].numel()):
					metric_unfold += [metric[m][n].item()]
			else:
				metric_unfold += [metric[m].item()]
		key_list = [i for i in metric.keys()]
		return metric_unfold, key_list
 
class visDOA(nn.Module):
	""" Function: Visualize localization results
	"""
	def __init__(self, ):
		super(visDOA, self).__init__()

	def forward(self, doa_gt, vad_gt, doa_est, vad_est, vad_TH, time_stamp, doa_invalid=200):
		""" Args:
				doa_gt, doa_est - (nt, 2, ns) in degrees
				vad_gt, vad_est - (nt, ns)  
				vad_TH 			- VAD threshold, [gtVAD_TH, estVAD_TH] 
			Returns: plt
		"""
		plt.switch_backend('agg')
		doa_mode = ['Elevation [º]', 'Azimuth [º]']
		range_mode = [[0, 180], [0, 180]]

		num_sources_gt = doa_gt.shape[-1]
		num_sources_pred = doa_est.shape[-1]
		ndoa_mode = 1
		for doa_mode_idx in [1]:
			valid_flag_all = np.sum(vad_gt, axis=-1)>0
			valid_flag_all = valid_flag_all[:,np.newaxis,np.newaxis].repeat(doa_gt.shape[1], axis=1).repeat(doa_gt.shape[2], axis=2)

			valid_flag_gt = vad_gt>vad_TH[0]
			valid_flag_gt = valid_flag_gt[:,np.newaxis,:].repeat(doa_gt.shape[1], axis=1)
			doa_gt_v = np.where(valid_flag_gt, doa_gt, doa_invalid)
			doa_gt_silence_v = np.where(valid_flag_gt==0, doa_gt, doa_invalid)

			valid_flag_pred = vad_est>vad_TH[1]
			valid_flag_pred = valid_flag_pred[:,np.newaxis,:].repeat(doa_est.shape[1], axis=1)
			doa_pred_v = np.where(valid_flag_pred & valid_flag_all, doa_est, doa_invalid)
			
			plt.subplot(ndoa_mode, 1, 1)
			plt.grid(linestyle=":", color="silver")
			for source_idx in range(num_sources_gt):
				# plt.plot(time_stamp, doa_gt[:, doa_mode_idx, source_idx], label='GT',
				# 		color='lightgray', linewidth=3, linestyle=style[0])
				plt_gt_silence = plt.scatter(time_stamp, doa_gt_silence_v[:, doa_mode_idx, source_idx],
						label='GT_silence', c='whitesmoke', marker='.', linewidth=1)

				plt_gt = plt.scatter(time_stamp, doa_gt_v[:, doa_mode_idx, source_idx],
						label='GT', c='lightgray', marker='o', linewidth=1.5)

			for source_idx in range(num_sources_pred):
				plt_est = plt.scatter(time_stamp, doa_pred_v[:, doa_mode_idx, source_idx],
						label='EST', c='firebrick', marker='.', linewidth=0.8)

			plt.gca().set_prop_cycle(None)
			plt.legend(handles = [plt_gt_silence, plt_gt, plt_est])
			plt.xlabel('Time [s]')
			plt.ylabel(doa_mode[doa_mode_idx])
			plt.ylim(range_mode[doa_mode_idx][0],range_mode[doa_mode_idx][1])

		return plt
		
		 
class AddChToBatch(nn.Module):
	""" Change dimension from  (nb, nch, ...) to (nb*(nch-1), ...) 
	"""
	def __init__(self, ch_mode):
		super(AddChToBatch, self).__init__()
		self.ch_mode = ch_mode

	def forward(self, data):
		nb = data.shape[0]
		nch = data.shape[1]

		if self.ch_mode == 'M':
			data_adjust = torch.zeros((nb*(nch-1),2)+data.shape[2:], dtype=torch.complex64).to(data.device) # (nb*(nch-1),2,nf,nt)
			for b_idx in range(nb):
				st = b_idx*(nch-1)
				ed = (b_idx+1)*(nch-1)
				data_adjust[st:ed, 0, ...] = data[b_idx, 0 : 1, ...].expand((nch-1,)+data.shape[2:])
				data_adjust[st:ed, 1, ...] = data[b_idx, 1 : nch, ...]

		elif self.ch_mode == 'MM':
			data_adjust = torch.zeros((nb*int((nch-1)*nch/2),2)+data.shape[2:], dtype=torch.complex64).to(data.device) # (nb*(nch-1)*nch/2,2,nf,nt)
			for b_idx in range(nb):
				for ch_idx in range(nch-1):
					st = b_idx*int((nch-1)*nch/2) + int((2*nch-2-ch_idx+1)*ch_idx/2)
					ed = b_idx*int((nch-1)*nch/2) + int((2*nch-2-ch_idx)*(ch_idx+1)/2)
					data_adjust[st:ed, 0, ...] = data[b_idx, ch_idx:ch_idx+1, ...].expand((nch-ch_idx-1,)+data.shape[2:])
					data_adjust[st:ed, 1, ...] = data[b_idx, ch_idx+1:, ...]
			
		return data_adjust.contiguous()

class RemoveChFromBatch(nn.Module):
	""" Change dimension from (nb*nmic, nt, nf) to (nb, nmic, nt, nf)
	"""
	def __init__(self, ch_mode):
		super(RemoveChFromBatch, self).__init__()
		self.ch_mode = ch_mode

	def forward(self, data, nb):
		nmic = int(data.shape[0]/nb)
		data_adjust = torch.zeros((nb, nmic)+data.shape[1:], dtype=torch.float32).to(data.device)
		for b_idx in range(nb):
			st = b_idx * nmic
			ed = (b_idx + 1) * nmic
			data_adjust[b_idx, ...] = data[st:ed, ...]
			
		return data_adjust.contiguous()


class DPIPD(nn.Module):
	""" Complex-valued Direct-path inter-channel phase difference	
	"""

	def __init__(self, ndoa_candidate, mic_location, nf=257, fre_max=8000, ch_mode='M', speed=343.0):
		super(DPIPD, self).__init__()

		self.ndoa_candidate = ndoa_candidate
		self.mic_location = mic_location
		self.nf = nf
		self.fre_max = fre_max
		self.speed = speed
		self.ch_mode = ch_mode

		nmic = mic_location.shape[-2]
		nele = ndoa_candidate[0]
		nazi = ndoa_candidate[1]
		ele_candidate = np.linspace(0, np.pi, nele)
		azi_candidate = np.linspace(-np.pi, np.pi, nazi)
		ITD = np.empty((nele, nazi, nmic, nmic))  # Time differences, floats
		IPD = np.empty((nele, nazi, nf, nmic, nmic))  # Phase differences
		fre_range = np.linspace(0.0, fre_max, nf)
		for m1 in range(nmic):
			for m2 in range(nmic):
				r = np.stack([np.outer(np.sin(ele_candidate), np.cos(azi_candidate)),
							  np.outer(np.sin(ele_candidate), np.sin(azi_candidate)),
							  np.tile(np.cos(ele_candidate), [nazi, 1]).transpose()], axis=2)
				ITD[:, :, m1, m2] = np.dot(r, mic_location[m2, :] - mic_location[m1, :]) / speed
				IPD[:, :, :, m1, m2] = -2 * np.pi * np.tile(fre_range[np.newaxis, np.newaxis, :], [nele, nazi, 1]) * \
									   np.tile(ITD[:, :, np.newaxis, m1, m2], [1, 1, nf])
		dpipd_template_ori = np.exp(1j * IPD)
		self.dpipd_template = self.data_adjust(dpipd_template_ori) # (nele, nazi, nf, nmic-1) / (nele, nazi, nf, nmic*(nmic-1)/2)
		self.doa_candidate = [ele_candidate, azi_candidate]

		# 	# import scipy.io
		# 	# scipy.io.savemat('dpipd_template_nele_nazi_2nf_nmic-1.mat',{'dpipd_template': self.dpipd_template})
		# 	# print(a)

		del ITD, IPD

	def forward(self, source_doa=None):
		# source_doa: (nb, ntimestep, 2, nsource)
		mic_location = self.mic_location
		nf = self.nf
		fre_max = self.fre_max
		speed = self.speed

		if source_doa is not None:
			source_doa = source_doa.transpose(0, 1, 3, 2) # (nb, ntimestep, nsource, 2)
			nmic = mic_location.shape[-2]
			nb = source_doa.shape[0]
			nsource = source_doa.shape[-2]
			ntime = source_doa.shape[-3]
			ITD = np.empty((nb, ntime, nsource, nmic, nmic))  # Time differences, floats
			IPD = np.empty((nb, ntime, nsource, nf, nmic, nmic))  # Phase differences
			fre_range = np.linspace(0.0, fre_max, nf)

			for m1 in range(nmic):
				for m2 in range(nmic):
					r = np.stack([np.sin(source_doa[:, :, :, 0]) * np.cos(source_doa[:, :, :, 1]),
								  np.sin(source_doa[:, :, :, 0]) * np.sin(source_doa[:, :, :, 1]),
								  np.cos(source_doa[:, :, :, 0])], axis=3)
					ITD[:, :, :, m1, m2] = np.dot(r, mic_location[m1, :] - mic_location[m2, :]) / speed # t2- t1
					IPD[:, :, :, :, m1, m2] = -2 * np.pi * np.tile(fre_range[np.newaxis, np.newaxis, np.newaxis, :],
										[nb, ntime, nsource, 1]) * np.tile(ITD[:, :, :, np.newaxis, m1, m2], [1, 1, 1, nf])*(-1)  # !!!! delete -1

			dpipd_ori = np.exp(1j * IPD)
			dpipd = self.data_adjust(dpipd_ori) # (nb, ntime, nsource, nf, nmic-1) / (nb, ntime, nsource, nf, nmic*(nmic-1)/2)

			dpipd = dpipd.transpose(0, 1, 3, 4, 2) # (nb, ntime, nf, nmic-1, nsource)

		else:
			dpipd = None

		return self.dpipd_template, dpipd, self.doa_candidate
	
	def data_adjust(self, data):
		# change dimension from (..., nmic-1) to (..., nmic*(nmic-1)/2)
		if self.ch_mode == 'M':
			data_adjust = data[..., 0, 1:] # (..., nmic-1)
		elif self.ch_mode == 'MM':
			nmic = data.shape[-1]
			data_adjust = np.empty(data.shape[:-2] + (int(nmic*(nmic-1)/2),), dtype=np.complex64)
			for mic_idx in range(nmic - 1):
				st = int((2 * nmic - 2 - mic_idx + 1) * mic_idx / 2)
				ed = int((2 * nmic - 2 - mic_idx) * (mic_idx + 1) / 2)
				data_adjust[..., st:ed] = data[..., mic_idx, (mic_idx+1):] # (..., nmic*(nmic-1)/2)
		else:
			raise Exception('Microphone channel mode unrecognised')

		return data_adjust

class SourceDetectLocalize(nn.Module):
	# Function: Iterative localization and voice-activity dectection
	def __init__(self, max_num_sources, source_num_mode='unkNum', meth_mode='IDL'):
		super(SourceDetectLocalize, self).__init__()
		self.max_num_sources = max_num_sources
		self.source_num_mode = source_num_mode
		self.meth_mode = meth_mode
		
	def forward(self, pred_ipd, dpipd_template, doa_candidate):
		# pred_ipd: (nb, nt, 2nf, nmic_pair)  2nf-[cos, sin]
		# dpipd_template: (nele, nazi, 2nf, nmic_pair)
		device = pred_ipd.device
		pred_ipd = pred_ipd.detach()
		nb, nt, nf, nmic = pred_ipd.shape
		nele, nazi, _, _ = dpipd_template.shape
		dpipd_template = dpipd_template[np.newaxis, ...].repeat(nb, 1, 1, 1, 1)
		ele_candidate = doa_candidate[0]
		azi_candidate = doa_candidate[1]

		pred_ss = torch.bmm(pred_ipd.contiguous().view(nb, nt, -1), dpipd_template.contiguous().view(nb, nele, nazi, -1)
					.permute(0, 3, 1, 2).view(nb, nmic * nf, -1))/(nmic*nf/2)  # (nb, nt, nele*nazi)
		pred_ss = pred_ss.view(nb, nt, nele, nazi)

		# 'KNum', ns = 1
		# 'UnkNum', ns = 2
		# max_num_sources = 2  # !!!!! maximum number of sources
		pred_DOAs = torch.zeros((nb, nt, 2, self.max_num_sources), dtype=torch.float32, requires_grad=False).to(device)
		pred_VADs = torch.zeros((nb, nt, self.max_num_sources), dtype=torch.float32, requires_grad=False).to(device)

		if self.meth_mode == 'IDL': # iterative source detection and localization
			
			for source_idx in range(self.max_num_sources):
				map = torch.bmm(pred_ipd.contiguous().view(nb, nt, -1),
								dpipd_template.contiguous().view(nb, nele, nazi, -1).permute(0, 3, 1, 2).view(nb, nmic * nf, -1)) / (
									nmic * nf / 2)  # (nb, nt, nele*nazi)
				map = map.view(nb, nt, nele, nazi)

				max_flat_idx = map.reshape((nb, nt, -1)).argmax(2)
				ele_max_idx, azi_max_idx = np.unravel_index(max_flat_idx.cpu().numpy(), map.shape[2:])  # (nb, nt)

				# ele_candidate = np.linspace(0, np.pi, nele)
				# azi_candidate = np.linspace(0, 0, nazi)
				pred_DOA = np.stack((ele_candidate[ele_max_idx], azi_candidate[azi_max_idx]),
									axis=-1)  # (nb, nt, 2)
				pred_DOA = torch.from_numpy(pred_DOA).to(device)
				pred_DOAs[:, :, :, source_idx] = pred_DOA

				max_dpipd_template = torch.zeros((nb, nt, nf, nmic), dtype=torch.float32, requires_grad=False).to(device)
				for b_idx in range(nb):
					for t_idx in range(nt):
						max_dpipd_template[b_idx, t_idx, :, :] = \
							dpipd_template[b_idx, ele_max_idx[b_idx, t_idx], azi_max_idx[b_idx, t_idx], :,
							:] * 1.0  # (nb, nt, 2nf, nmic-1)
						ratio = torch.sum(
							max_dpipd_template[b_idx, t_idx, :, :] * pred_ipd[b_idx, t_idx, :, :]) / \
								torch.sum(
									max_dpipd_template[b_idx, t_idx, :, :] * max_dpipd_template[b_idx, t_idx, :, :])
						max_dpipd_template[b_idx, t_idx, :, :] = ratio * max_dpipd_template[b_idx, t_idx, :, :]
						if self.source_num_mode == 'kNum':
							pred_VADs[b_idx, t_idx, source_idx] = 1
						elif self.source_num_mode == 'unkNum':
							pred_VADs[b_idx, t_idx, source_idx] = ratio * 1
				pred_ipd = pred_ipd - max_dpipd_template

		elif self.meth_mode =='PD': # peak detection
			ss = deepcopy(pred_ss[:,:,:,0:-1]) # redundant azi
			# Find peaks: compare values with their neighbours
			ss_top = torch.cat((ss[:, :, 0:1, :],ss[:, :, 0:-1, :]), dim=2)
			ss_bottom = torch.cat((ss[:, :, 1:, :],ss[:, :, -1:, :]), dim=2)
			ss_left = torch.cat((ss[:, :, :, -1:],ss[:, :, :, 0:-1]), dim=3)
			ss_right = torch.cat((ss[:, :, :, 1:],ss[:, :, :, 0:1]), dim=3)
			ss_top_left = torch.cat((torch.cat((ss[:, :, 0:1, -1:],ss[:, :, 0:1, 0:-1]), dim=3),
										torch.cat((ss[:, :, 0:-1, -1:],ss[:, :, 0:-1, 0:-1]), dim=3)), dim=2)
			ss_top_right = torch.cat((torch.cat((ss[:, :, 0:1, 1:],ss[:, :, 0:1, 0:1]), dim=3),
										torch.cat((ss[:, :, 0:-1, 1:],ss[:, :, 0:-1, 0:1]), dim=3)), dim=2)
			ss_bottom_left = torch.cat((torch.cat((ss[:, :, 1:, -1:],ss[:, :, 1:, 0:-1]), dim=3),
										torch.cat((ss[:, :, -1:, -1:],ss[:, :, -1:, 0:-1]), dim=3)), dim=2)
			ss_bottom_right = torch.cat((torch.cat((ss[:, :, 1:, 1:],ss[:, :, 1:, 0:1]), dim=3),
										torch.cat((ss[:, :, -1:, 1:],ss[:, :, -1:, 0:1]), dim=3)), dim=2)
			peaks = (ss>ss_top)&(ss>ss_bottom)&(ss>ss_left)&(ss>ss_right) &\
					(ss>ss_top_left)&(ss>ss_top_right)&(ss>ss_bottom_left)&(ss>ss_bottom_right)
			peaks = torch.cat((peaks, torch.zeros_like(peaks[:,:,:,0:1])), dim=3)
			peaks_reshape = peaks.reshape((nb, nt, -1))
			ss_reshape = pred_ss.reshape((nb, nt, -1))

			# ele_candidate = np.linspace(0, np.pi, nele)
			# azi_candidate = np.linspace(-np.pi, np.pi, nazi)
			for b_idx in range(nb):
				for t_idx in range(nt):
					peaks_idxs = torch.nonzero(peaks_reshape[b_idx, t_idx, :]==1)# ???
					max_flat_idx = sorted(peaks_idxs,
											key=lambda k: ss_reshape[b_idx, t_idx, k], reverse=True)
					max_flat_idx = max_flat_idx[0:self.max_num_sources]
					max_flat_peakvalue = ss_reshape[b_idx, t_idx, max_flat_idx]
					max_flat_idx = [i.cpu() for i in max_flat_idx]
					ele_max_idx, azi_max_idx = np.unravel_index(max_flat_idx, peaks.shape[2:])  # (ns)
					pred_DOA = np.stack((ele_candidate[ele_max_idx], azi_candidate[azi_max_idx]), axis=-1)  # (ns,2)
					pred_DOA = torch.from_numpy(pred_DOA).to(device)
					pred_DOAs[b_idx, t_idx, :, :] = pred_DOA.transpose(1, 0) * 1
					if self.source_num_mode == 'kNum':
						pred_VADs[b_idx, t_idx, :] = 1
					elif self.source_num_mode == 'unkNum':
						pred_VADs[b_idx, t_idx, :] = max_flat_peakvalue * 1
		else:
			raise Exception('Localizion method is unrecognized')


		# data association - for tracking !!! vad needs to adjust with doa adjustment
		track_enable = False
		if track_enable == True:
			for b_idx in range(nb):
				for t_idx in range(nt-1):
					temp = []
					for source_idx in range(self.max_num_sources):
						temp += [pred_DOAs[b_idx, t_idx+1, :, source_idx]]
					pair_permute = list(permutations(temp, self.max_num_sources))

					diff = torch.zeros((len(pair_permute))).to(device)
					for pair_idx in range(len(pair_permute)):
						pair = torch.stack(pair_permute[pair_idx]).permute(1,0)
						abs_diff1 = torch.abs(pair - pred_DOAs[b_idx, t_idx, :, :])
						abs_diff2 = deepcopy(abs_diff1)
						abs_diff2[1,:] = np.pi*2-abs_diff1[1,:]
						abs_diff = torch.min(abs_diff1, abs_diff2)
						diff[pair_idx] = torch.sum(abs_diff)

					pair_idx_sim = torch.argmin(diff)
					pred_DOAs[b_idx, t_idx + 1, :, :] = torch.stack(pair_permute[pair_idx_sim]).permute(1,0)


		return pred_DOAs, pred_VADs, pred_ss


class GCC(nn.Module):
	""" Compute the Generalized Cross Correlation of the inputs.
	In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K).
	You can use tau_max to output only the central part of the GCCs and transform='PHAT' to use the PHAT transform.
	"""

	def __init__(self, N, K, tau_max=None, transform=None):
		assert transform is None or transform == 'PHAT', 'Only the \'PHAT\' transform is implemented'
		assert tau_max is None or tau_max <= K // 2
		super(GCC, self).__init__()

		self.K = K
		self.N = N
		self.tau_max = tau_max if tau_max is not None else K // 2
		self.transform = transform

	def forward(self, x):
		x_fft_c = torch.fft.rfft(x)
		x_fft = torch.stack((x_fft_c.real, x_fft_c.imag), -1)  

		if self.transform == 'PHAT':
			mod = torch.sqrt(complex_conjugate_multiplication(x_fft, x_fft))[..., 0]
			mod += 1e-12  # To avoid numerical issues
			x_fft /= mod.reshape(tuple(x_fft.shape[:-1]) + (1,))

		gcc = torch.empty(list(x_fft.shape[0:-3]) + [self.N, self.N, 2 * self.tau_max + 1], device=x.device)
		for n in range(self.N):
			gcc_fft_batch = complex_conjugate_multiplication(x_fft[..., n, :, :].unsqueeze(-3), x_fft)
			gcc_fft_batch_c = torch.complex(gcc_fft_batch[..., 0], gcc_fft_batch[..., 1])
			gcc_batch = torch.fft.irfft(gcc_fft_batch_c)    

			gcc[..., n, :, 0:self.tau_max + 1] = gcc_batch[..., 0:self.tau_max + 1]
			gcc[..., n, :, -self.tau_max:] = gcc_batch[..., -self.tau_max:]

		return gcc


class SRP_map(nn.Module):
	""" Compute the SRP-PHAT maps from the GCCs taken as input.
	In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K), the
	desired resolution of the maps (resTheta and resPhi), the microphone positions relative to the center of the
	array (rn) and the sampling frequency (fs).
	With normalize=True (default) each map is normalized to ethe range [-1,1] approximately
	"""

	def __init__(self, N, K, resTheta, resPhi, rn, fs, c=343.0, normalize=True, thetaMax=np.pi / 2):
		super(SRP_map, self).__init__()

		self.N = N
		self.K = K
		self.resTheta = resTheta
		self.resPhi = resPhi
		self.fs = float(fs)
		self.normalize = normalize

		self.cross_idx = np.stack([np.kron(np.arange(N, dtype='int16'), np.ones((N), dtype='int16')),
								   np.kron(np.ones((N), dtype='int16'), np.arange(N, dtype='int16'))])

		self.theta = np.linspace(0, thetaMax, resTheta)
		self.phi = np.linspace(-np.pi, np.pi, resPhi + 1)
		self.phi = self.phi[0:-1]

		self.IMTDF = np.empty((resTheta, resPhi, self.N, self.N))  # Time differences, floats
		for k in range(self.N):
			for l in range(self.N):
				r = np.stack(
					[np.outer(np.sin(self.theta), np.cos(self.phi)), np.outer(np.sin(self.theta), np.sin(self.phi)),
					 np.tile(np.cos(self.theta), [resPhi, 1]).transpose()], axis=2)
				self.IMTDF[:, :, k, l] = np.dot(r, rn[l, :] - rn[k, :]) / c

		tau = np.concatenate([range(0, K // 2 + 1), range(-K // 2 + 1, 0)]) / float(fs)  # Valid discrete values
		self.tau0 = np.zeros_like(self.IMTDF, dtype=np.int)
		for k in range(self.N):
			for l in range(self.N):
				for i in range(resTheta):
					for j in range(resPhi):
						self.tau0[i, j, k, l] = int(np.argmin(np.abs(self.IMTDF[i, j, k, l] - tau)))
		self.tau0[self.tau0 > K // 2] -= K
		self.tau0 = self.tau0.transpose([2, 3, 0, 1])

	def forward(self, x):
		tau0 = self.tau0
		tau0[tau0 < 0] += x.shape[-1]
		maps = torch.zeros(list(x.shape[0:-3]) + [self.resTheta, self.resPhi], device=x.device).float()
		for n in range(self.N):
			for m in range(self.N):
				maps += x[..., n, m, tau0[n, m, :, :]]

		if self.normalize:
			maps -= torch.mean(torch.mean(maps, -1, keepdim=True), -2, keepdim=True)
			maps += 1e-12  # To avoid numerical issues
			maps /= torch.max(torch.max(maps, -1, keepdim=True)[0], -2, keepdim=True)[0]

		return maps


class SphericPad(nn.Module):
	""" Replication padding for time axis, reflect padding for the elevation and circular padding for the azimuth.
	The time padding is optional, do not use it with CausConv3d.
	"""
	def __init__(self, pad):
		super(SphericPad, self).__init__()
		
		if len(pad) == 4:
			self.padLeft, self.padRight, self.padTop, self.padBottom = pad
			self.padFront, self.padBack = 0, 0
		elif len(pad) == 6:
			self.padLeft, self.padRight, self.padTop, self.padBottom, self.padFront, self.padBack = pad
		else: 
			raise Exception('Expect 4 or 6 values for padding (padLeft, padRight, padTop, padBottom, [padFront, padBack])')
	
	def forward(self, x):
		assert x.shape[-1] >= self.padRight and x.shape[-1] >= self.padLeft, \
			'Padding size should be less than the corresponding input dimension for the azimuth axis'

		if self.padBack > 0 or self.padFront > 0:
			x = F.pad(x, (0, 0, 0, 0, self.padFront, self.padBack), 'replicate')

		input_shape = x.shape
		x = x.view((x.shape[0], -1, x.shape[-2], x.shape[-1]))

		x = F.pad(x, (0, 0, self.padTop, self.padBottom), 'reflect') # Actually, it should add a pi shift

		x = torch.cat((x[..., -self.padLeft:], x, x[..., :self.padRight]), dim=-1)

		return x.view((x.shape[0],) + input_shape[1:-2] + (x.shape[-2], x.shape[-1]))


# %% Neural Network layers

class CausConv3d(nn.Module):
	""" Causal 3D Convolution for SRP-PHAT maps sequences
	"""
	def __init__(self, in_channels, out_channels, kernel_size):
		super(CausConv3d, self).__init__()
		self.pad = kernel_size[0] - 1
		self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=(self.pad, 0, 0))

	def forward(self, x):
		return self.conv(x)[:, :, :-self.pad, :, :]


class CausConv2d(nn.Module):
	""" Causal 2D Convolution for spectrograms and GCCs sequences
	"""
	def __init__(self, in_channels, out_channels, kernel_size):
		super(CausConv2d, self).__init__()
		self.pad = kernel_size[0] - 1
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(self.pad, 0))

	def forward(self, x):
		return self.conv(x)[:, :, :-self.pad, :]


class CausConv1d(nn.Module):
	""" Causal 1D Convolution
	"""
	def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
		super(CausConv1d, self).__init__()
		self.pad = (kernel_size - 1) * dilation
		self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation)
		
	def forward(self, x):
		return self.conv(x)[:, :, :-self.pad]

class CausCnnBlock1x1(nn.Module):
	# expansion = 1
	def __init__(self, inplanes, planes, kernel=(1,1), stride=(1,1), padding=(0,0)):
		super(CausCnnBlock1x1, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)

	def forward(self, x):
		out = self.conv1(x)

		return out

class CausCnnBlock(nn.Module): 
	""" Function: Basic convolutional block
    """
	# expansion = 1
	def __init__(self, inplanes, planes, kernel=(3,3), stride=(1,1), padding=(1,2), use_res=True, downsample=None):
		super(CausCnnBlock, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.downsample = downsample
		self.stride = stride
		self.pad = padding
		self.use_res = use_res

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		if self.pad[1] !=0:
			out = out[:,:,:,:-self.pad[1]]

		out = self.conv2(out)
		out = self.bn2(out)
		if self.pad[1] != 0:
			out = out[:, :, :, :-self.pad[1]]

		if self.use_res == True:
			if self.downsample is not None:
				residual = self.downsample(x)
			out += residual

		out = self.relu(out)

		return out
