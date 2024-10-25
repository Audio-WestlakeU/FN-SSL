"""
	Utils functions to deal with spherical coordinates in Pytorch.
"""

from math import pi
import torch
import soundfile
import pickle
from matplotlib import pyplot as plt
import numpy as np
import random
def forgetting_norm(input, sample_length=298):
        """
        Using the mean value of the near frames to normalization
        Args:
            input: feature
            sample_length: length of the training sample, used for calculating smooth factor
        Returns:
            normed feature
        Shapes:
            input: [B, C, F, T]
            sample_length_in_training: 192
        """
        assert input.ndim == 4
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size, num_channels * num_freqs, num_frames)

        eps = 1e-10
        mu = 0
        alpha = (sample_length - 1) / (sample_length + 1)

        mu_list = []
        for frame_idx in range(num_frames):
            if frame_idx < sample_length:
                alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(
                    input[:, :, frame_idx], dim=1
                ).reshape(
                    batch_size, 1
                )  # [B, 1]
            else:
                current_frame_mu = torch.mean(input[:, :, frame_idx], dim=1).reshape(
                    batch_size, 1
                )  # [B, 1]
                mu = alpha * mu + (1 - alpha) * current_frame_mu

            mu_list.append(mu)

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]

        output = mu.reshape(batch_size, 1, 1, num_frames)
        return output
    
    
def cart2sph(cart, include_r=False):
	""" Cartesian coordinates to spherical coordinates conversion.
	Each row contains one point in format (x, y, x) or (elevation, azimuth, radius),
	where the radius is optional according to the include_r argument.
	"""
	r = torch.sqrt(torch.sum(torch.pow(cart, 2), dim=-1))
	theta = torch.acos(cart[..., 2] / r)
	phi = torch.atan2(cart[..., 1], cart[..., 0])
	if include_r:
		sph = torch.stack((theta, phi, r), dim=-1)
	else:
		sph = torch.stack((theta, phi), dim=-1)
	return sph


def sph2cart(sph):
	""" Spherical coordinates to cartesian coordinates conversion.
	Each row contains one point in format (x, y, x) or (elevation, azimuth, radius),
	where the radius is supposed to be 1 if it is not included.
	"""
	if sph.shape[-1] == 2: sph = torch.cat((sph, torch.ones_like(sph[..., 0]).unsqueeze(-1)), dim=-1)
	x = sph[..., 2] * torch.sin(sph[..., 0]) * torch.cos(sph[..., 1])
	y = sph[..., 2] * torch.sin(sph[..., 0]) * torch.sin(sph[..., 1])
	z = sph[..., 2] * torch.cos(sph[..., 0])
	return torch.stack((x, y, z), dim=-1)


## for room acoustic data saving and reading 
def save_file(mic_signal, acoustic_scene, sig_path, acous_path):
    
    if sig_path is not None:
        soundfile.write(sig_path, mic_signal, acoustic_scene.fs)

    if acous_path is not None:
        file = open(acous_path,'wb')
        file.write(pickle.dumps(acoustic_scene.__dict__))
        file.close()

def load_file(acoustic_scene, sig_path, acous_path):

    if sig_path is not None:
        mic_signal, fs = soundfile.read(sig_path)

    if acous_path is not None:
        file = open(acous_path,'rb')
        dataPickle = file.read()
        file.close()
        acoustic_scene.__dict__ = pickle.loads(dataPickle)

    if (sig_path is not None) & (acous_path is not None):
        return mic_signal, acoustic_scene
    elif (sig_path is not None) & (acous_path is None):
        return mic_signal
    elif (sig_path is None) & (acous_path is not None):
        return acoustic_scene

def locata_plot(result_path, save_fig_path, bias=4):
    plt.figure(figsize=(16,8),dpi=300)
    for k in range(12):   
        doa_gt = np.load(result_path+str(k)+'_gt.npy')
        doa_est = np.load(result_path+str(k)+'_est.npy')-bias
        vad_gt = np.load(result_path+str(k)+'_vadgt.npy')
        vad_gt[vad_gt<2/3] = -1
        vad_gt[vad_gt>2/3] = 1
        for i in range(1):
            plt.subplot(3,4,k+1)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.3, hspace=0.3)
            x = [j*4096/16000 for j in range(doa_gt.shape[1])]
            plt.scatter(x,doa_gt[i,:,1,0],s=5,c='grey',linewidth=0.8,label='GT')
            plt.scatter(x,doa_est[i,:,1,0]*vad_gt[i,:,0],s=3,c='firebrick',linewidth=0.8,label='EST')
            #plt.scatter(x,doa_est[i,:,1,0],s=3,c='firebrick',linewidth=0.8,label='EST')
            plt.xlabel('Time [s]')
            plt.ylabel('DOA[°]')
            plt.ylim((0,180))
            plt.grid()
            plt.legend(loc=0,prop={'size': 4})
    plt.savefig(save_fig_path + 'locata_fig.jpg')
    
def set_seed(seed):
	""" Function: fix random seed.
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False # avoid-CUDNN_STATUS_NOT_SUPPORTED #(commont if use cpu??)

	np.random.seed(seed)
	random.seed(seed)

def set_random_seed(seed):

    np.random.seed(seed)
    random.seed(seed)

def pad_segments(x, seg_len):
    """ Pad the input tensor x to ensure the t-dimension is divisible by seg_len """
    nb, nt, nf, nc = x.shape
    pad_len = (seg_len - (nt % seg_len)) % seg_len  
    if pad_len > 0:
        pad = torch.zeros(nb, pad_len, nf, nc, device=x.device, dtype=x.dtype) 
        x = torch.cat([x, pad], dim=1)  
    return x

def split_segments(x, seg_len):
    """ Split the input tensor x along the t-dimension into segments of length seg_len """
    nb, nt, nf, nc = x.shape
    x = pad_segments(x, seg_len)  # Pad the input to make it divisible by seg_len
    nt_padded = x.shape[1]  # New time dimension after padding
    x = x.reshape(nb, nt_padded // seg_len, seg_len, nf, nc)  # Reshape into segments
    return x
