import torch
import numpy as np
import torch
import random 
import pickle
import soundfile 

## for spherical coordinates
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

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        #print(mu.shape)
        #output = input / (mu + eps)

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


## for training process 

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

def get_learning_rate(optimizer):
    """ Function: get learning rates from optimizer
    """ 
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def set_learning_rate(epoch, lr_init, step, gamma):
    """ Function: adjust learning rates 
    """ 
    lr = lr_init*pow(gamma, int(epoch/step))
    return lr

## for data number

def detect_infnan(data, mode='torch'):
    """ Function: check whether there is inf/nan in the element of data or not
    """ 
    if mode == 'troch':
        inf_flag = torch.isinf(data)
        nan_flag = torch.isnan(data)
    elif mode == 'np':
        inf_flag = np.isinf(data)
        nan_flag = np.isnan(data)
    else:
        raise Exception('Detect infnan mode unrecognized')
    if (True in inf_flag):
        raise Exception('INF exists in data')
    if (True in nan_flag):
        raise Exception('NAN exists in data')


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




# def angular_error_2d(pred, true, doa_mode='azi'):
# 	""" 2D Angular distance between spherical coordinates.
# 	"""
# 	if doa_mode == 'azi':
# 		ae = torch.abs((pred-true+np.pi)%np.pi-np.pi)
# 	elif doa_mode == 'ele':
# 		ae = torch.abs(pred-true)

# 	return  ae

# def angular_error(the_pred, phi_pred, the_true, phi_true):
# 	""" Angular distance between spherical coordinates.
# 	"""
# 	aux = torch.cos(the_true) * torch.cos(the_pred) + \
# 		  torch.sin(the_true) * torch.sin(the_pred) * torch.cos(phi_true - phi_pred)

# 	return torch.acos(torch.clamp(aux, -0.99999, 0.99999))


# def mean_square_angular_error(y_pred, y_true):
# 	""" Mean square angular distance between spherical coordinates.
# 	Each row contains one point in format (elevation, azimuth).
# 	"""
# 	the_true = y_true[:, 0]
# 	phi_true = y_true[:, 1]
# 	the_pred = y_pred[:, 0]
# 	phi_pred = y_pred[:, 1]

# 	return torch.mean(torch.pow(angular_error(the_pred, phi_pred, the_true, phi_true), 2), -1)


# def rms_angular_error_deg(y_pred, y_true):
# 	""" Root mean square angular distance between spherical coordinates.
# 	Each input row contains one point in format (elevation, azimuth) in radians
# 	but the output is in degrees.
# 	"""

# 	return torch.sqrt(mean_square_angular_error(y_pred, y_true)) * 180 / pi
