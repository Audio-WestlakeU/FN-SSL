from Opt import opt
from typing import Callable
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import numpy as np
import Dataset as at_dataset
import Module as at_module
import IPDnet.FixedAarryIPDnet as at_model
from utils.flops import write_FLOPs
from utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback
from utils import tag_and_log_git_status
from utils import MyLogger as TensorBoardLogger
from utils import MyRichProgressBar as RichProgressBar
from packaging.version import Version
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning import LightningDataModule, LightningModule
from jsonargparse import lazy_instance
from torch import Tensor
import torch
from typing import Tuple
import os
from copy import deepcopy
from torchmetrics.functional import permutation_invariant_training
from torchmetrics.functional.audio.pit import pit_permutate
from scipy.special import jn
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ["OMP_NUM_THREADS"] = str(8)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


opts = opt()
dirs = opts.dir()

win_len = 512
win_shift_ratio = 0.5
seg_fra_ratio = 12
seg_len = int(win_len*win_shift_ratio*(seg_fra_ratio+1))
seg_shift = int(win_len*win_shift_ratio*seg_fra_ratio)
segmenting = at_dataset.Segmenting_SRPDNN(
    K=seg_len, step=seg_shift, window=None)

dataset_train = at_dataset.FixTrajectoryDataset(
    data_dir=dirs['sensig_train'],
    dataset_sz=300000,
    transforms=[segmenting]
)
dataset_dev = at_dataset.FixTrajectoryDataset(
    data_dir=dirs['sensig_dev'],
    dataset_sz=4000,
    transforms=[segmenting]
)
dataset_test = at_dataset.FixTrajectoryDataset(
    data_dir=dirs['sensig_test'],
    dataset_sz=4000,
    transforms=[segmenting]
)

class MyDataModule(LightningDataModule):
    
    def __init__(self, num_workers: int = 5, batch_size: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.num_workers = num_workers
        # train: batch_size[0]; test: batch_size[1]
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        return super().prepare_data()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset_train, batch_size=self.batch_size[0], num_workers=self.num_workers,shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset_dev, batch_size=self.batch_size[1], num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset_test, batch_size=self.batch_size[1], num_workers=self.num_workers)
    
class MyModel(LightningModule):
    def __init__(
            self,
            tar_useVAD: bool = True,
            ch_mode: str = 'M',
            res_the: int = 1,
            res_phi: int = 180,
            fs: int = 16000,
            win_len: int = 512,
            nfft: int = 512,
            win_shift_ratio: float = 0.5,
            max_source: int = 2,
            device: str = 'cuda',
            mic_pos: Tensor = torch.tensor((((-0.04, 0.0, 0.0),(0.04, 0.0, 0.0),))),
            compile: bool = False,
            is_linear_array: bool = True,
            is_planar_array: bool = True, # both lienar and planar only affect the mapping from IPD to DOA
            exp_name: str = 'exp',
    ):
        super().__init__()
        # for 2-mic IPDnet
        self.arch = at_model.IPDnet()
        if compile:
            assert Version(torch.__version__) >= Version(
                '2.0.0'), torch.__version__
            self.arch = torch.compile(self.arch)

        self.save_hyperparameters(ignore=['arch'])
        self.tar_useVAD = tar_useVAD
        self.ch_mode = ch_mode
        self.nfft = nfft
        self.fre_max = fs / 2
        self.max_source = max_source
        self.mic_pos =   mic_pos.cpu().numpy()
        self.is_linear_array = is_linear_array
        self.dostft = at_module.STFT(
            win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft)
        self.gerdpipd = at_module.DPIPD(ndoa_candidate=[res_the, res_phi],
                                        mic_location=self.mic_pos,
                                        nf=int(self.nfft/2) + 1,
                                        fre_max=self.fre_max,
                                        ch_mode=self.ch_mode,
                                        speed=340)
        self.addbatch = at_module.AddChToBatch(ch_mode=self.ch_mode)
        self.removebatch = at_module.RemoveChFromBatch(ch_mode=self.ch_mode)
        self.fre_range_used = range(1, int(self.nfft/2)+1, 1)
        # Mapping IPD to DOA and calculate metrics
        self.get_metric = at_module.PredDOA(mic_location=self.mic_pos,is_linear_array=is_linear_array,is_planar_array=is_planar_array)
        self.dev = device
    def forward(self, x):
        return self.arch(x)

    def on_train_start(self):
        if self.current_epoch == 0:
            if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir') and 'notag' not in self.hparams.exp_name:
                tag_and_log_git_status(self.logger.log_dir + '/git.out', self.logger.version,
                                       self.hparams.exp_name, model_name=type(self).__name__)

            if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir'):
                with open(self.logger.log_dir + '/model.txt', 'a') as f:
                    f.write(str(self))
                    f.write('\n\n\n')


    def training_step(self, batch, batch_idx: int):
        
        mic_sig_batch = batch[0]
        acoustic_scene_batch = batch[1]
        data_batch = self.data_preprocess(mic_sig_batch, acoustic_scene_batch)
        in_batch = data_batch[0]
        gt_batch = data_batch[1:]
        pred_batch = self(in_batch)
        loss = self.cal_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        self.log("train/loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        acoustic_scene_batch = batch[1]
        data_batch = self.data_preprocess(mic_sig_batch, acoustic_scene_batch)
        in_batch = data_batch[0]
        gt_batch = data_batch[1:]
        pred_batch = self(in_batch)
        loss = self.cal_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        self.log("valid/loss", loss, sync_dist=True)
        metric = self.get_metric(pred_batch=pred_batch,gt_batch=gt_batch,idx=None)
        for m in metric:
            self.log('valid/'+m, metric[m].item(), sync_dist=True)       

    def test_step(self, batch: Tensor, batch_idx: int):
        mic_sig_batch = batch[0]
        acoustic_scene_batch = batch[1]
        data_batch = self.data_preprocess(mic_sig_batch, acoustic_scene_batch)
        in_batch = data_batch[0]
        gt_batch = data_batch[1:]
        pred_batch = self(in_batch,offline_inference=True)
        loss = self.cal_loss(pred_batch=pred_batch, gt_batch=gt_batch,batch_idx=batch_idx)
        self.log("test/loss", loss, sync_dist=True)
        metric = self.get_metric(pred_batch=pred_batch,gt_batch=gt_batch,idx=batch_idx)
        for m in metric:
            self.log('test/'+m, metric[m].item(), sync_dist=True)

    def predict_step(self, batch, batch_idx: int):
        data_batch = self.data_preprocess(mic_sig_batch=batch.permute(0,2,1))
        in_batch = data_batch[0]        
        preds = self.forward(in_batch)
        return preds[0]

    def MSE_loss(self, preds, targets):
        nbatch = preds.shape[0]
        sum_loss = torch.nn.functional.mse_loss(preds, targets, reduction='none').contiguous().view(nbatch,-1)
        item_num = sum_loss.shape[1]
        return sum_loss.sum(axis=1) / item_num


    #Using Frame-level PIT for training
    def cal_loss(self, pred_batch=None, gt_batch=None,batch_idx=None):
        ipd_gt_batch = gt_batch[1]
        nb, nt, _, nmic, nsrc = pred_batch.shape
        pred_batch = pred_batch.reshape(nb*nt, -1, nsrc).permute(0,2,1)
        ipd_gt_batch = ipd_gt_batch.reshape(nb*nt,-1,nsrc).permute(0,2,1)
        pred_batch = pred_batch.to(self.dev)
        best_metric, best_perm = permutation_invariant_training(pred_batch, ipd_gt_batch, self.MSE_loss, 'min')
        pred_batch= pit_permutate(pred_batch, best_perm)
        loss = torch.nn.functional.mse_loss(
            pred_batch.contiguous(), ipd_gt_batch.contiguous())
        return loss
    
    # non-source target 
    def euclidean_distances_to_bessel(self, data,fre_use, order=0):
        reference = data[0, :]
        distances = np.sqrt(np.sum((data[1:] - reference) ** 2, axis=1))
        frequencies = 2 * np.pi * np.linspace(0, 8000, 257) / 340
        frequencies = frequencies[fre_use]
        bessel_values_extended = []
        for distance in distances:
            bessel_value = jn(order, frequencies * distance)
            zero_vector = np.zeros(256)
            extended_value = np.concatenate((bessel_value, zero_vector))
            bessel_values_extended.append(extended_value)
        bessel_values_final = np.array(bessel_values_extended)
        return bessel_values_final.T
    
    # dp-vad 
    def cal_vad(self,dp_mic_sig_batch,stft):
        nb,nf,nt,nc = stft.shape
        dp_vad = torch.zeros(nb,nt,self.max_source)
        for source_idx in range(self.max_source):
            dp_temp = self.dostft(signal=dp_mic_sig_batch[:,:,:,source_idx])
            dp_temp_mag = torch.abs(dp_temp)
            vad_temp = dp_temp_mag[:,:,:,0] / torch.abs(stft[:,:,:,0])
            vad_temp = torch.mean(vad_temp,dim=1)
            dp_vad[:,:,source_idx] = vad_temp
        pooling = torch.nn.AvgPool2d(kernel_size=(12, 1))
        dp_vad = pooling(dp_vad)
        return dp_vad
    
    def data_preprocess(self, mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None, eps=1e-6):
        data = []
        #Do STFT for noisy multi-channel signals             
        stft = self.dostft(signal=mic_sig_batch)
        nb,nf,nt,nc = stft.shape
        #Using DP-signal to calculate VAD
        dp_mic_sig_batch = acoustic_scene_batch['dp_signal'] 
        dp_vad = self.cal_vad(dp_mic_sig_batch=dp_mic_sig_batch,stft=stft)
        
        stft_rebatch = stft.permute(0, 3, 1, 2)
        stft_rebatch = stft_rebatch.to(self.dev)
        mag = torch.abs(stft_rebatch)
        # offline normalization
        mean_value = torch.mean(mag.reshape(mag.shape[0],-1), dim = 1)
        mean_value = mean_value[:,np.newaxis,np.newaxis,np.newaxis].expand(mag.shape)
            #mean_value = forgetting_norm(mag,sample_length=280)
        stft_rebatch_real = torch.real(stft_rebatch) / (mean_value + eps)
        stft_rebatch_image = torch.imag(stft_rebatch) / (mean_value + eps)
        real_image_batch = torch.cat(
            (stft_rebatch_real, stft_rebatch_image), dim=1)
        data += [real_image_batch[:, :, self.fre_range_used, :]]

        DOAw_batch = acoustic_scene_batch['doa'].to(self.dev)
        source_doa = DOAw_batch.cpu().numpy()
        
        if self.ch_mode == 'M':
            _, ipd_batch = self.gerdpipd(source_doa=source_doa)
        elif self.ch_mode == 'MM':
            _, ipd_batch = self.gerdpipd(source_doa=source_doa)

        non_source_tar = self.euclidean_distances_to_bessel(self.mic_pos,fre_use=self.fre_range_used)                
        non_source_tar = torch.from_numpy(non_source_tar)                          

        ipd_batch = np.concatenate((ipd_batch.real[:, :, self.fre_range_used, :, :], ipd_batch.imag[:, :, self.fre_range_used, :, :]), axis=2).astype(np.float32)  # (nb, ntime, 2nf, nmic-1, nsource)
        ipd_batch = torch.from_numpy(ipd_batch).to(self.dev)
        nb, nt, nf, nmic, nsrc = ipd_batch.shape
        
        vad_batch_copy = deepcopy(dp_vad).to(self.dev)
        th = 0.001
        vad_batch_copy[vad_batch_copy <= th] = 0
        vad_batch_copy[vad_batch_copy > th] = 1
        vad_batch_expand_ipd = vad_batch_copy[:, :, np.newaxis, np.newaxis,:].expand(nb, nt, nf, nmic, nsrc)
        
        # set silence frame to non-source target
        ipd_batch = ipd_batch * vad_batch_expand_ipd
        for i in range(nb):
            for j in range(nt):
                for k in range(nsrc):
                    if (ipd_batch[i,j,:,:,k] == 0).all():
                        ipd_batch[i,j,:,:,k] = non_source_tar.to(ipd_batch)

        ipd_batch = ipd_batch.view(nb*nt, nf, nmic, nsrc)
        data += [DOAw_batch]
        data += [ipd_batch]
        if self.tar_useVAD:
            data += [dp_vad]
        return data 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.arch.parameters(), lr=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975, last_epoch=-1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'valid/loss',
            }
        }

class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

        parser.set_defaults(
            {"trainer.strategy": "ddp"})
        parser.set_defaults({"trainer.accelerator": "gpu"})

                
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({
            "early_stopping.monitor": "valid/loss",
            "early_stopping.min_delta": 0.01,
            "early_stopping.patience": 100,
            "early_stopping.mode": "min",
        })

        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_valid_loss{valid/loss:.4f}",
            "model_checkpoint.monitor": "valid/loss",
            "model_checkpoint.mode": "min",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": 5,
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True
        }
        parser.set_defaults(model_checkpoint_defaults)

        # RichProgressBar
        parser.add_lightning_class_args(
            RichProgressBar, nested_key='progress_bar')
        parser.set_defaults({
            "progress_bar.console_kwargs": {
                "force_terminal": True,
                "no_color": True,  
                "width": 200,  
            }
        })

        # LearningRateMonitor
        parser.add_lightning_class_args(
            LearningRateMonitor, "learning_rate_monitor")
        learning_rate_monitor_defaults = {
            "learning_rate_monitor.logging_interval": "epoch",
        }
        parser.set_defaults(learning_rate_monitor_defaults)


    def before_fit(self):
        resume_from_checkpoint: str = self.config['fit']['ckpt_path']
        if resume_from_checkpoint is not None and resume_from_checkpoint.endswith('last.ckpt'):
            resume_from_checkpoint = os.path.normpath(resume_from_checkpoint)
            splits = resume_from_checkpoint.split(os.path.sep)
            version = int(splits[-3].replace('version_', ''))
            save_dir = os.path.sep.join(splits[:-3])
            self.trainer.logger = TensorBoardLogger(
                save_dir=save_dir, name="", version=version, default_hp_metric=False)
        else:
            model_name = type(self.model).__name__
            self.trainer.logger = TensorBoardLogger(
                'logs/', name=model_name, default_hp_metric=False)

    def before_test(self):
        torch.set_num_interop_threads(5)
        torch.set_num_threads(5)
        if self.config['test']['ckpt_path'] != None:
            ckpt_path = self.config['test']['ckpt_path']
        else:
            raise Exception('You should give --ckpt_path if you want to test')
        epoch = os.path.basename(ckpt_path).split('_')[0]
        write_dir = os.path.dirname(os.path.dirname(ckpt_path))
        exp_save_path = os.path.normpath(write_dir + '/' + epoch)

        import time
        # add 10 seconds for threads to simultaneously detect the next version
        self.trainer.logger = TensorBoardLogger(
            exp_save_path, name='', default_hp_metric=False)
        time.sleep(10)

    def after_test(self):
        if not self.trainer.is_global_zero:
            return
        import fnmatch
        files = fnmatch.filter(os.listdir(
            self.trainer.log_dir), 'events.out.tfevents.*')
        for f in files:
            os.remove(self.trainer.log_dir + '/' + f)
            print('tensorboard log file for test is removed: ' +
                  self.trainer.log_dir + '/' + f)


if __name__ == '__main__':
    cli = MyCLI(
        MyModel,
        MyDataModule,
        seed_everything_default=2, 
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        #parser_kwargs={"parser_mode": "omegaconf"},
    )
