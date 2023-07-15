import numpy as np
import os
import math
import scipy
import scipy.io
import scipy.signal
import random
# import librosa
import soundfile
import pandas
import warnings
from copy import deepcopy
from collections import namedtuple
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import animation
import webrtcvad
#import gpuRIR
from utils_ import load_file

# %% Util functions

def acoustic_power(s):
	""" Acoustic power of after removing the silences.
	"""
	w = 512  # Window size for silent detection
	o = 256  # Window step for silent detection

	# Window the input signal
	s = np.ascontiguousarray(s)
	sh = (s.size - w + 1, w)
	st = s.strides * 2
	S = np.lib.stride_tricks.as_strided(s, strides=st, shape=sh)[0::o]

	window_power = np.mean(S ** 2, axis=-1)
	th = 0.01 * window_power.max()  # Threshold for silent detection
	return np.mean(window_power[np.nonzero(window_power > th)])


def cart2sph(cart):
	xy2 = cart[:,0]**2 + cart[:,1]**2
	sph = np.zeros_like(cart)
	sph[:,0] = np.sqrt(xy2 + cart[:,2]**2)
	sph[:,1] = np.arctan2(np.sqrt(xy2), cart[:,2]) # Elevation angle defined from Z-axis down
	sph[:,2] = np.arctan2(cart[:,1], cart[:,0])
	return sph


# %% Util classes

class Parameter:
	""" Random parammeter class.
	"""
	def __init__(self, *args, discrete=False):
		self.discrete = discrete
		if discrete == False:
			if len(args) == 1:
				self.random = False
				self.value = np.array(args[0])
				self.min_value = None
				self.max_value = None
			elif len(args) == 2:
				self.random = True
				self.min_value = np.array(args[0])
				self.max_value = np.array(args[1])
				self.value = None
			else:
				raise Exception('Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
		else:
			self.value_range = args[0]

	def getValue(self):
		if self.discrete == False:
			if self.random:
				return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
			else:
				return self.value
		else:
			idx = np.random.randint(0, len(self.value_range))
			return self.value_range[idx]


# !!! mic_scale should be 1, due to the 'mic_pos' parameter given in learner
ArraySetup = namedtuple('ArraySetup', 'arrayType, orV, mic_scale, mic_pos, mic_orV, mic_pattern')

# Named tuple with the characteristics of a microphone array and definitions of dual-channel array
dualch_array_setup = ArraySetup(arrayType='planar', 
    orV = np.array([0.0, 1.0, 0.0]), # ??? put the source in oneside (indicated by orV) of the array
	mic_scale = Parameter(1), # !!! half of the mic_distance should be smaller than the minimum separation between the array and the walls defined by array_pos
	mic_pos = np.array(((
			(-0.04, 0.0, 0.0),
			(0.04, 0.0, 0.0),
	))), # Actural position is mic_scale*mic_pos
	mic_orV = None, # Invalid for omnidirectional microphones
	mic_pattern = 'omni'
)


class AcousticScene:
	""" Acoustic scene class.
	"""
	def __init__(self, room_sz, T60, beta, noise_signal, SNR, source_signal, fs, array_setup, mic_pos, timestamps, traj_pts, 
				 trajectory, t, DOA, c=343.0):
		self.room_sz = room_sz				# Room size
		self.T60 = T60						# Reverberation time of the simulated room
		self.beta = beta					# Reflection coefficients of the walls of the room (make sure it corresponds with T60)
		self.noise_signal = noise_signal	# Noise signal (nsample', 1)
		self.source_signal = source_signal  # Source signal (nsample,nsource)
		self.fs = fs						# Sampling frequency of the source signal and the simulations
		self.SNR = SNR						# Signal to (omnidirectional) noise ratio to simulate
		self.array_setup = array_setup		# Named tuple with the characteristics of the array
		self.mic_pos = mic_pos				# Position of microphones (nch,3)
		self.timestamps = timestamps		# Time of each simulation (it does not need to correspond with the DOA estimations) (npoint)
		self.traj_pts = traj_pts 			# Trajectory points to simulate (npoint,3,nsource)
		self.trajectory = trajectory		# Continuous trajectory (nsample,3,nsource)
		self.t = t							# Continuous time (nsample)
		self.DOA = DOA 						# Continuous DOA (nsample,3,nsource)
		self.c = c 							# Speed of sound 
 
	def simulate(self):
		""" Get the array recording using gpuRIR to perform the acoustic simulations.
		"""
		if self.T60 == 0:
			Tdiff = 0.1
			Tmax = 0.1
			nb_img = [1,1,1]
		else:
			Tdiff = gpuRIR.att2t_SabineEstimator(12, self.T60) # Use ISM until the RIRs decay 12dB
			Tmax = gpuRIR.att2t_SabineEstimator(40, self.T60)  # Use diffuse model until the RIRs decay 40dB
			if self.T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
			nb_img = gpuRIR.t2n( Tdiff, self.room_sz )

		# nb_mics  = len(self.mic_pos)
		# nb_traj_pts = len(self.traj_pts)
		# nb_gpu_calls = min(int(np.ceil( self.fs * Tdiff * nb_mics * nb_traj_pts * np.prod(nb_img) / 1e9 )), nb_traj_pts)
		# traj_pts_batch = np.ceil( nb_traj_pts / nb_gpu_calls * np.arange(0, nb_gpu_calls+1) ).astype(int)

		num_source = self.traj_pts.shape[-1]
		RIRs_sources = []
		mic_signals_sources = []
		dp_RIRs_sources = []
		dp_mic_signals_sources = []
		for source_idx in range(num_source):
			RIRs = gpuRIR.simulateRIR(self.room_sz, self.beta, self.traj_pts[:,:,source_idx], self.mic_pos,
						nb_img, Tmax, self.fs, Tdiff=Tdiff, orV_rcv=self.array_setup.mic_orV, 
						mic_pattern=self.array_setup.mic_pattern)
			mic_sig = gpuRIR.simulateTrajectory(self.source_signal[:,source_idx], RIRs, timestamps=self.timestamps, fs=self.fs)
			mic_sig = mic_sig[0:len(self.t),:]

			dp_RIRs = gpuRIR.simulateRIR(self.room_sz, self.beta, self.traj_pts[:,:,source_idx], self.mic_pos, [1,1,1],
							0.1, self.fs, orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern)
			dp_mic_sig = gpuRIR.simulateTrajectory(self.source_signal[:,source_idx], dp_RIRs, timestamps=self.timestamps, fs=self.fs)
			dp_mic_sig = dp_mic_sig[0:len(self.t),:]

			RIRs_sources += [RIRs]
			mic_signals_sources += [mic_sig]
			dp_RIRs_sources += [dp_RIRs]
			dp_mic_signals_sources += [dp_mic_sig]

		RIRs_sources = np.array(RIRs_sources).transpose(1,2,3,0) # (npoints,nch,nsamples,nsources)
		mic_signals_sources = np.array(mic_signals_sources).transpose(1,2,0) # (nsamples,nch,nsources)
		dp_RIRs_sources = np.array(dp_RIRs_sources).transpose(1,2,3,0)
		dp_mic_signals_sources = np.array(dp_mic_signals_sources).transpose(1,2,0)

		# Add Noise
		if self.noise_signal is None:
			self.noise_signal = np.random.standard_normal(mic_sig.shape)
		mic_signals = np.sum(mic_signals_sources, axis=2) # (nsamples, nch)
		dp_mic_signals = np.sum(dp_mic_signals_sources, axis=2)
		ac_pow = np.mean([acoustic_power(dp_mic_signals[:,i]) for i in range(dp_mic_signals_sources.shape[1])])
		ac_pow_noise = np.mean([acoustic_power(self.noise_signal[:,i]) for i in range(self.noise_signal.shape[1])])
		noise_signal = np.sqrt(ac_pow/10**(self.SNR/10)) / np.sqrt(ac_pow_noise) * self.noise_signal
		mic_signals += noise_signal[0:len(self.t), :]

		# Apply the propagation delay to the VAD information if it exists
		if hasattr(self, 'source_vad'):
			self.mic_vad_sources = [] # for vad of separate sensor signals of source
			for source_idx in range(num_source):
				vad = gpuRIR.simulateTrajectory(self.source_vad[:,source_idx], dp_RIRs_sources[:,:,:,source_idx],
												timestamps=self.timestamps, fs=self.fs)
				vad_sources = vad[0:len(self.t),:].mean(axis=1) > vad[0:len(self.t),:].max()*1e-3
				self.mic_vad_sources += [vad_sources] # binary value
			self.mic_vad_sources = np.array(self.mic_vad_sources).transpose(1,0) # (nsample, nsources)
			self.mic_vad = np.sum(self.mic_vad_sources, axis=1)>0.5 # for vad of mixed sensor signals of sources (nsample)


		return mic_signals

## %% Trajectory Datasets
class FixTrajectoryDataset(Dataset):
	"""
	"""
	def __init__(self, data_dir, dataset_sz, transforms=None, return_acoustic_scene=False):
		"""
		"""
		self.transforms = transforms
		self.data_paths = []
		data_names = os.listdir(data_dir)
		for fname in data_names:
			front, ext = os.path.splitext(fname)
			if ext == '.wav':
				self.data_paths.append((os.path.join(data_dir, fname)))
		self.dataset_sz = len(self.data_paths) if dataset_sz is None else dataset_sz
		self.return_acoustic_scene = return_acoustic_scene

	def __len__(self):
		return self.dataset_sz 

	def __getitem__(self, idx):
		if idx < 0: idx = len(self) + idx

		sig_path = self.data_paths[idx]
		acous_path = sig_path.replace('wav','npz')

		acoustic_scene = AcousticScene(
					room_sz = [],
					T60 = [],
					beta = [],
					noise_signal = [],
					SNR = [],
					array_setup = [],
					mic_pos = [],
					source_signal = [],
					fs = [],
					traj_pts = [],
					timestamps = [],
					trajectory = [],
					t = [],
					DOA = [],
					c = []
				)
		mic_signals, acoustic_scene = load_file(acoustic_scene, sig_path, acous_path)

		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

		if self.return_acoustic_scene:
			return mic_signals, acoustic_scene
		else: 
			# mic_signals = mic_signals.astype(np.float32)
			# gts = []
			# gts += [acoustic_scene.DOAw.astype(np.float32)]
			# gts += [acoustic_scene.mic_vad_sources]
			# gts += [acoustic_scene.mic_vad]
			gts = {}
			gts['doa'] = acoustic_scene.DOAw.astype(np.float32)
			gts['vad_sources'] = acoustic_scene.mic_vad_sources

			# gts['vad'] = acoustic_scene.mic_vad

			return mic_signals, gts


class LocataDataset(Dataset):
	""" Dataset with the LOCATA dataset recordings and its corresponding Acoustic Scenes.
	"""
	def __init__(self, paths, array, fs, tasks=(1,3,5), recording=None, dev=False, transforms = None, return_acoustic_scene=False):
		"""
		path: path to the root of the LOCATA dataset in your file system
		array: string with the desired array ('dummy', 'eigenmike', 'benchmark2' or 'dicit')
		fs: sampling frequency (you can use it to downsample the LOCATA recordings)
		tasks: LOCATA tasks to include in the dataset (only one-source tasks are supported)
		recording: recordings that you want to include in the dataset (only supported if you selected only one task)
		dev: True if the groundtruth source positions are available
		transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
		"""
		assert array in ('dummy', 'eigenmike', 'benchmark2', 'dicit'), 'Invalid array.'
		assert recording is None or len(tasks) == 1, 'Specific recordings can only be selected for dataset with only one task'
		#for task in tasks: assert task in (1,3,5), 'Invalid task ' + str(task) + '.'

		self.path = paths
		self.dev = dev
		self.array = array
		self.tasks = tasks
		self.transforms = transforms
		self.fs = fs

		self.vad = webrtcvad.Vad()
		self.vad.set_mode(1)

		if array == 'dummy':
			self.array_setup = dummy_array_setup
		elif array == 'eigenmike':
			self.array_setup = eigenmike_array_setup
		elif array == 'benchmark2':
			self.array_setup = benchmark2_array_setup
		elif array == 'dicit':
			self.array_setup = dicit_array_setup

		self.directories = []
		for path in paths:
			for task in tasks:
				task_path = os.path.join(path, 'task' + str(task))
				for recording in os.listdir( task_path ):
					arrays = os.listdir( os.path.join(task_path, recording) )
					if array in arrays:
						self.directories.append( os.path.join(task_path, recording, array) )
		self.directories.sort()

		self.return_acoustic_scene = return_acoustic_scene

	def __len__(self):
		return len(self.directories)

	def __getitem__(self, idx):
		# idx = 4
		directory0 = self.directories[idx]
		directory = directory0.replace('\\','/')
		mic_signals, fs = soundfile.read( os.path.join(directory, 'audio_array_' + self.array + '.wav') )

		if fs > self.fs:
			mic_signals = scipy.signal.decimate(mic_signals, int(fs/self.fs), axis=0)
			new_fs = fs / int(fs/self.fs)
			if new_fs != self.fs: warnings.warn('The actual fs is {}Hz'.format(new_fs))
			self.fs = new_fs
		elif fs < self.fs:
			raise Exception('The sampling rate of the file ({}Hz) was lower than self.fs ({}Hz'.format(fs, self.fs))

		mic_signals_ori = deepcopy(mic_signals)

		# Remove initial silence
		start = np.argmax(mic_signals[:,0] > mic_signals[:,0].max()*0.15)
		mic_signals = mic_signals[start:,:]
		t = (np.arange(len(mic_signals)) + start)/self.fs

		df = pandas.read_csv( os.path.join(directory, 'position_array_' + self.array + '.txt'), sep='\t' )
		array_pos = np.stack((df['x'].values, df['y'].values,df['z'].values), axis=-1)
		array_ref_vec = np.stack((df['ref_vec_x'].values, df['ref_vec_y'].values,df['ref_vec_z'].values), axis=-1)
		array_rotation = np.zeros((array_pos.shape[0],3,3))
		for i in range(3):
			for j in range(3):
				array_rotation[:,i,j] = df['rotation_' + str(i+1) + str(j+1)]

		df = pandas.read_csv( os.path.join(directory, 'required_time.txt'), sep='\t' )
		required_time = df['hour'].values*3600+df['minute'].values*60+df['second'].values
		timestamps = required_time - required_time[0]

		if self.dev:
			sources_name = [] #loudspeaker
			for file in os.listdir( directory ):
				if file.startswith('audio_source') and file.endswith('.wav'):
					ls = file[13:-4]
					sources_name.append(ls)

			sources_signal = []
			sources_pos = []
			trajectories = []
			sensor_vads = []

			for source_name in sources_name:
				file = 'audio_source_' + source_name + '.wav'
				source_signal, fs_src = soundfile.read(os.path.join(directory, file))
				if fs_src > self.fs:
					source_signal = scipy.signal.decimate(source_signal, int(fs_src / self.fs), axis=0)
				source_signal = source_signal[start:start + len(t)]
				sources_signal.append(source_signal)

			for source_name in sources_name:
				file = 'position_source_' + source_name + '.txt'
				df = pandas.read_csv(os.path.join(directory, file), sep='\t')
				source_pos = np.stack((df['x'].values, df['y'].values, df['z'].values), axis=-1)

				sources_pos.append(source_pos)
				trajectories.append(
					np.array([np.interp(t, timestamps, source_pos[:, i]) for i in range(3)]).transpose())

			for source_name in sources_name:
				array = directory.split('/')[-1]
				file = 'VAD_' + array + '_' + source_name + '.txt'
				df = pandas.read_csv(os.path.join(directory, file), sep='\t')
				sensor_vad_ori = df['VAD'].values

				# VAD values @48kHz matched with timestamps @16kHz
				L_audio = len(sensor_vad_ori)
				t_stamps_audio = np.linspace(0,L_audio-1,L_audio) / fs_src #48 kHz
				t_stamps_opti = t + 0.0 # 16 kHz
				sensor_vad = np.zeros(len(t_stamps_opti))
				cnt = 0
				for i in range(1,L_audio):
					if t_stamps_audio[i] >= t_stamps_opti[cnt]:
						sensor_vad[cnt] = sensor_vad_ori[i - 1]
						cnt = cnt + 1
					if cnt > len(sensor_vad)-1:
						break
				if cnt <= len(sensor_vad)-1:
					VAD[cnt: end] = sensor_vad_ori[end]
					if cnt < len(sensor_vad) - 2:
						print('Warning: VAD values do not match~')

				sensor_vads.append(sensor_vad)

			# !! the fist dimension is for sources
			sources_signal = np.stack(sources_signal)
			sources_pos = np.stack(sources_pos)
			trajectories = np.stack(trajectories)
			sensor_vads = np.stack(sensor_vads)

			DOA_pts = np.zeros(sources_pos.shape[0:2] + (2,))
			DOA = np.zeros(trajectories.shape[0:2] + (2,))
			for s in range(sources_pos.shape[0]):
				source_pos_local = np.matmul( np.expand_dims(sources_pos[s,...] - array_pos, axis=1), array_rotation ).squeeze() # np.matmul( array_rotation, np.expand_dims(sources_pos[s,...] - array_pos, axis=-1) ).squeeze()
				# DOA_pts[s,...] = cart2sph(source_pos_local) [:,1:3]
				# DOA[s,...] = np.array([np.interp(t, timestamps, DOA_pts[s,:,i]) for i in range(2)]).transpose()
				DOA_pts[s, ...] = cart2sph(source_pos_local)[:, 1:3]
				source_pos_local_interp = np.array([np.interp(t, timestamps, source_pos_local[ :, i]) for i in range(3)]).transpose()
				DOA[s, ...] = cart2sph(source_pos_local_interp)[:, 1:3]
			# DOA[DOA[...,1]<-np.pi, 1] += 2*np.pi

		else:
			sources_pos = None
			DOA = None
			source_signal = np.NaN * np.ones((len(mic_signals),1))

		acoustic_scene = AcousticScene(
			room_sz = np.NaN * np.ones((3,1)),
			T60 = np.NaN,
			beta = np.NaN * np.ones((6,1)),
			noise_signal = np.NaN, 
			SNR = np.NaN,
			source_signal = sources_signal.transpose(1,0),
			fs = self.fs,
			array_setup = self.array_setup,
			mic_pos = np.matmul( array_rotation[0,...], np.expand_dims(self.array_setup.mic_pos*self.array_setup.mic_scale.getValue(), axis=-1) ).squeeze() + array_pos[0,:], 
			timestamps = timestamps - start/self.fs, # time-stamp-wise
			traj_pts = sources_pos.transpose(1,2,0), #[0,...] # time-stamp-wise
			t = t - start/self.fs, # sample-wise
			trajectory = trajectories.transpose(1,2,0), # sample-wise
			DOA = DOA.transpose(1,2,0), #[0,...] # sample-wise
			c = np.NaN
		)

		# vad_from = 'alg'
		vad_from = 'dataset'
		if vad_from == 'alg':
			sources_signal = sources_signal.transpose(1, 0)
			vad = np.zeros_like(sources_signal)
			vad_frame_len = int(10e-3 * self.fs)
			n_vad_frames = len(source_signal) // vad_frame_len
			num_sources = sources_signal.shape[-1]
			for source_idx in range(num_sources):
				for frame_idx in range(n_vad_frames):
					frame = sources_signal[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len, source_idx]
					frame_bytes = (frame * 32767).astype('int16').tobytes()
					vad[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len, source_idx] = \
						self.vad.is_speech(frame_bytes, int(self.fs))
		elif vad_from == 'dataset':
			vad = sensor_vads.transpose(1,0)

		acoustic_scene.mic_vad_sources = deepcopy(vad) # (nsample, nsource)
		acoustic_scene.mic_vad = np.sum(vad, axis=1)>0.5 # for vad of mixed sensor signals of source
		# acoustic_scene.mic_signal = mic_signal*1
		# acoustic_scene.mic_signal_start = start*1
		
		mic_signals.transpose()
		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)
		
		if self.return_acoustic_scene:
			return mic_signals.copy(), acoustic_scene
		else:
			gts = {}
			gts['doa'] = acoustic_scene.DOAw.astype(np.float32)
			gts['vad_sources'] = acoustic_scene.mic_vad_sources
			# print(mic_signal.shape, mic_signals.shape, start, gts['doa'].shape, gts['vad_sources'].shape)
			
			return mic_signals.copy(), gts # np.ascontiguousarray(mic_signals)


# %% Transform classes
class Segmenting_SRPDNN(object):
	""" Segmenting transform.
	"""
	def __init__(self, K, step, window=None):
		self.K = K
		self.step = step
		if window is None:
			self.w = np.ones(K)
		elif callable(window):
			try: self.w = window(K)
			except: raise Exception('window must be a NumPy window function or a Numpy vector with length K')
		elif len(window) == K:
			self.w = window
		else:
			raise Exception('window must be a NumPy window function or a Numpy vector with length K')

	def __call__(self, x, acoustic_scene):
		# N_mics = x.shape[1]
		N_dims = acoustic_scene.DOA.shape[1]
		num_source = acoustic_scene.DOA.shape[2]
		L = x.shape[0]
		N_w = np.floor(L/self.step - self.K/self.step + 1).astype(int)

		if self.K > L:
			raise Exception('The window size can not be larger than the signal length ({})'.format(L))
		elif self.step > L:
			raise Exception('The window step can not be larger than the signal length ({})'.format(L))

		DOA = []
		for source_idx in range(num_source):
			DOA += [np.append(acoustic_scene.DOA[:,:,source_idx], np.tile(acoustic_scene.DOA[-1,:,source_idx].reshape((1,2)),
				[N_w*self.step+self.K-L, 1]), axis=0)] # Replicate the last known DOA
		DOA = np.array(DOA).transpose(1,2,0) 

		shape_DOAw = (N_w, self.K, N_dims) # (nwindow, win_len, naziele)
		strides_DOAw = [self.step*N_dims, N_dims, 1]
		strides_DOAw = [strides_DOAw[i] * DOA.itemsize for i in range(3)]
		acoustic_scene.DOAw = [] 
		for source_idx in range(num_source):
			DOAw = np.lib.stride_tricks.as_strided(DOA[:,:,source_idx], shape=shape_DOAw, strides=strides_DOAw)
			DOAw = np.ascontiguousarray(DOAw)
			for i in np.flatnonzero(np.abs(np.diff(DOAw[..., 1], axis=1)).max(axis=1) > np.pi):
				DOAw[i, DOAw[i,:,1]<0, 1] += 2*np.pi # Avoid jumping from -pi to pi in a window
			DOAw = np.mean(DOAw, axis=1)
			DOAw[DOAw[:,1]>np.pi, 1] -= 2*np.pi
			acoustic_scene.DOAw += [DOAw]
		acoustic_scene.DOAw = np.array(acoustic_scene.DOAw).transpose(1, 2, 0) # (nsegment,naziele,nsource)

		# Pad and window the VAD if it exists
		if hasattr(acoustic_scene, 'mic_vad'): # (nsample,1)
			vad = acoustic_scene.mic_vad[:, np.newaxis] 
			vad = np.append(vad, np.zeros((L - vad.shape[0], 1)), axis=0)

			shape_vadw = (N_w, self.K, 1)
			strides_vadw = [self.step * 1, 1, 1]
			strides_vadw = [strides_vadw[i] * vad.itemsize for i in range(3)]

			acoustic_scene.mic_vad = np.lib.stride_tricks.as_strided(vad, shape=shape_vadw, strides=strides_vadw)[..., 0]

		# Pad and window the VAD if it exists
		if hasattr(acoustic_scene, 'mic_vad_sources'): # (nsample,nsource)
			shape_vadw = (N_w, self.K, 1)
			
			num_sources = acoustic_scene.mic_vad_sources.shape[1]
			vad_sources = []
			for source_idx in range(num_sources):
				vad = acoustic_scene.mic_vad_sources[:, source_idx:source_idx+1]  
				vad = np.append(vad, np.zeros((L - vad.shape[0], 1)), axis=0)

				strides_vadw = [self.step * 1, 1, 1]
				strides_vadw = [strides_vadw[i] * vad.itemsize for i in range(3)]
				vad_sources += [np.lib.stride_tricks.as_strided(vad, shape=shape_vadw, strides=strides_vadw)[..., 0]]

			acoustic_scene.mic_vad_sources = np.array(vad_sources).transpose(1,2,0) # (nsegment, nsample, nsource)

		# Timestamp for each window
		acoustic_scene.tw = np.arange(0, (L-self.K), self.step) / acoustic_scene.fs

		return x, acoustic_scene

