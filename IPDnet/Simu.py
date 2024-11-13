import os
from Opt import opt
from utils_ import set_seed,save_file,load_file
opts = opt()
args = opts.parse()
dirs = opts.dir()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
from Dataset import Parameter
import Dataset as at_dataset
import tqdm
if args.train:
    data_num = 300000
    stage = 'train'
    snr_range = Parameter(-5,15)
    rt_range = Parameter(0.2,1.3)
    set_seed(100)

if args.test:
    data_num = 4000
    stage = 'test'
    snr_range = Parameter(0,15)
    rt_range = Parameter(0.2,1)
    set_seed(101)

if args.dev:
    data_num = 4000
    stage = 'dev'
    snr_range = Parameter(0,15)
    rt_range = Parameter(0.2,1)
    set_seed(102)

speed = 343.0	
fs = 16000
T = 4.5 # Trajectory length (s) 
traj_points = 50 # number of RIRs per trajectory
array_setup = at_dataset.dualch_array_setup
# Source signal
sourceDataset = at_dataset.LibriSpeechDataset(
	path = dirs['sousig_'+stage], 
	T = T, 
	fs = fs,
	num_source = max(args.sources), 
	return_vad = True, 
	clean_silence = True,
	stage = stage,)

# Noise signal
noiseDataset = at_dataset.NoiseDataset(
	T = T, 
	fs = fs, 
	nmic = array_setup.mic_pos.shape[0], 
	noise_type = Parameter(['diffuse'], discrete=True), 
	noise_path = dirs['noisig_'+stage], 
	c = speed)

dataset = at_dataset.RandomTrajectoryDataset(
	sourceDataset = sourceDataset,
	num_source = Parameter(args.sources, discrete=True), # Random number of sources from list-args.sources
	source_state = args.source_state,
	room_sz = Parameter([6,6,2.5], [10,8,6]),  	# Random room sizes from 6x6x2.5 to 10x8x6 meters
	T60 = rt_range,					# Random reverberation times
	abs_weights = Parameter([0.5]*6, [1.0]*6),  # Random absorption weights ratios between walls
	array_setup = array_setup,
	array_pos = Parameter([0.35,0.35,0.3], [0.65,0.65,0.5]), # Ensure a minimum separation between the array and the walls
	noiseDataset = noiseDataset,
	SNR = snr_range, 	# Start the simulation with a low level of omnidirectional noise
	nb_points = traj_points,	# Simulate RIRs per trajectory
	min_dis = Parameter(0.5),
	c = speed, 
	transforms = []
	)
	# Data generation
save_dir = 'data/'+stage+'/'
exist_temp = os.path.exists(save_dir)
if exist_temp==False:
	os.makedirs(save_dir)
	print('make dir: ' + save_dir)
print(data_num)
for idx in tqdm.tqdm(range(data_num)):
	mic_signals, acoustic_scene = dataset[idx]    
	sig_path = save_dir + '/' + str(idx) + '.wav'
	acous_path = save_dir + '/' + str(idx) + '.npz'
	save_file(mic_signals, acoustic_scene, sig_path, acous_path)
