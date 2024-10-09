from Opt import opt
from typing import Callable
from utils_ import forgetting_norm
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import numpy as np
import Dataset as at_dataset
import Module as at_module
import Model as at_model
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
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ["OMP_NUM_THREADS"] = str(8)
# limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
#torch.backends.cuda.matmul.allow_tf32 = True
#The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
#torch.backends.cudnn.allow_tf32 = True
# torch.set_float32_matmul_precision('medium')


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
    dataset_sz=166816,
    transforms=[segmenting]
)
dataset_dev = at_dataset.FixTrajectoryDataset(
    data_dir=dirs['sensig_dev'],
    dataset_sz=992,
    transforms=[segmenting]
)
dataset_test = at_dataset.FixTrajectoryDataset(
    data_dir=dirs['sensig_test'],
    dataset_sz=5000,
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
        # return DataLoader(self.dpz, batch_size=None)
        return DataLoader(dataset_train, batch_size=self.batch_size[0], num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset_dev, batch_size=self.batch_size[1], num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset_test, batch_size=self.batch_size[1], num_workers=self.num_workers)

class MyModel(LightningModule):

    def __init__(
            self,
            tar_useVAD: bool = True,
            ch_mode: str = 'MM',
            res_the: int = 37,
            res_phi: int = 73,
            fs: int = 16000,
            win_len: int = 512,
            nfft: int = 512,
            win_shift_ratio: float = 0.5,
            method_mode: str = 'IDL',
            source_num_mode: str = 'KNum',
            max_num_sources: int = 1,
            return_metric: bool = True,
            exp_name: str = 'exp',
            compile: bool = False,
            device: str = "cuda",
    ):
        super().__init__()
        self.arch = at_model.FN_SSL()
        if compile:
            assert Version(torch.__version__) >= Version(
                '2.0.0'), torch.__version__
            self.arch = torch.compile(self.arch)

        # save all the parameters to self.hparams
        self.save_hyperparameters(ignore=['arch'])
        self.tar_useVAD = tar_useVAD
        self.method_mode = method_mode
        self.dev = device
        self.source_num_mode = source_num_mode
        self.max_num_sources = max_num_sources
        self.ch_mode = ch_mode
        self.nfft = nfft
        self.fre_max = fs / 2
        self.return_metric = return_metric
        self.dostft = at_module.STFT(
            win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft)
        self.gerdpipd = at_module.DPIPD(ndoa_candidate=[res_the, res_phi],
                                        mic_location=np.array(
                                            (((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0),))),
                                        nf=int(self.nfft/2) + 1,
                                        fre_max=self.fre_max,
                                        ch_mode=self.ch_mode,
                                        speed=340)
        self.addbatch = at_module.AddChToBatch(ch_mode=self.ch_mode)
        self.removebatch = at_module.RemoveChFromBatch(ch_mode=self.ch_mode)
        self.fre_range_used = range(1, int(self.nfft/2)+1, 1)
        self.get_metric = at_module.PredDOA()

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
                write_FLOPs(model=self, save_dir=self.logger.log_dir,
                            num_chns=2, fs=16000, audio_time_len=4.79, model_file=__file__)

    def training_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        gt_batch = batch[1]
        in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)

        pred_batch = self(in_batch)
        loss = self.cal_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        self.log("train/loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        gt_batch = batch[1]
        in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)
        pred_batch = self(in_batch)
        loss = self.cal_loss(pred_batch=pred_batch, gt_batch=gt_batch)

        self.log("valid/loss", loss, sync_dist=True)
        metric = self.get_metric(pred_batch=pred_batch, gt_batch=gt_batch)
        for m in metric:
            self.log('valid/'+m, metric[m].item(), sync_dist=True)
    
    def test_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        gt_batch = batch[1]
        in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)
        pred_batch = self(in_batch)
        loss = self.cal_loss(pred_batch=pred_batch, gt_batch=gt_batch)

        self.log("test/loss", loss, sync_dist=True)
        metric = self.get_metric(pred_batch=pred_batch, gt_batch=gt_batch)
        #print(metric)
        for m in metric:
            self.log('test/'+m, metric[m].item(), sync_dist=True)
    
    def predict_step(self, batch, batch_idx: int):
        data_batch = self.data_preprocess(mic_sig_batch=batch.permute(0, 2, 1))
        in_batch = data_batch[0]
        # print(in_batch.device)
        preds = self.forward(in_batch)
        return preds

    def cal_loss(self, pred_batch=None, gt_batch=None):
        pred_ipd = pred_batch
        gt_ipd = gt_batch['ipd']
        nb, _, _, _ = gt_ipd.shape  # (nb, nt, nf, nmic)
        pred_ipd_rebatch = self.removebatch(pred_ipd, nb).permute(0, 2, 3, 1)
        loss = torch.nn.functional.mse_loss(
            pred_ipd_rebatch.contiguous(), gt_ipd.contiguous())
        return loss

    def data_preprocess(self, mic_sig_batch=None, gt_batch=None, vad_batch=None, eps=1e-6, nor_flag=True):

        data = []
        if mic_sig_batch is not None:
            mic_sig_batch = mic_sig_batch.to(self.dev)

            stft = self.dostft(signal=mic_sig_batch)  # (nb,nf,nt,nch)
            stft = stft.permute(0, 3, 1, 2)  # (nb,nch,nf,nt)

            # change batch (nb,nch,nf,nt)→(nb*(nch-1),2,nf,nt)/(nb*(nch-1)*nch/2,2,nf,nt)
            stft_rebatch = self.addbatch(stft)
            if nor_flag:
                nb, nc, nf, nt = stft_rebatch.shape
                mag = torch.abs(stft_rebatch)
                mean_value = forgetting_norm(mag)
                stft_rebatch_real = torch.real(
                    stft_rebatch) / (mean_value + eps)
                stft_rebatch_image = torch.imag(
                    stft_rebatch) / (mean_value + eps)
            else:
                stft_rebatch_real = torch.real(stft_rebatch)
                stft_rebatch_image = torch.imag(stft_rebatch)
            # prepare model input
            real_image_batch = torch.cat(
                (stft_rebatch_real, stft_rebatch_image), dim=1)
            data += [real_image_batch[:, :, self.fre_range_used, :]]

        if gt_batch is not None:
            DOAw_batch = gt_batch['doa']
            vad_batch = gt_batch['vad_sources']

            source_doa = DOAw_batch.cpu().numpy()

            if self.ch_mode == 'M':
                _, ipd_batch, _ = self.gerdpipd(source_doa=source_doa)
            elif self.ch_mode == 'MM':
                _, ipd_batch, _ = self.gerdpipd(source_doa=source_doa)
            ipd_batch = np.concatenate((ipd_batch.real[:, :, self.fre_range_used, :, :], ipd_batch.imag[:, :, self.fre_range_used, :, :]), axis=2).astype(
                np.float32)  # (nb, ntime, 2nf, nmic-1, nsource)
            ipd_batch = torch.from_numpy(ipd_batch)

            # (nb,nseg,nsource) # s>2/3
            vad_batch = vad_batch.mean(axis=2).float()

            # DOAw_batch = torch.from_numpy(source_doa).to(self.device)
            DOAw_batch = DOAw_batch.to(self.dev)  # (nb,nseg,2,nsource)
            ipd_batch = ipd_batch.to(self.dev)
            vad_batch = vad_batch.to(self.dev)

            if self.tar_useVAD:
                nb, nt, nf, nmic, num_source = ipd_batch.shape
                th = 0
                vad_batch_copy = deepcopy(vad_batch)
                vad_batch_copy[vad_batch_copy <= th] = th
                vad_batch_copy[vad_batch_copy > 0] = 1
                vad_batch_expand = vad_batch_copy[:, :, np.newaxis, np.newaxis, :].expand(
                    nb, nt, nf, nmic, num_source)
                ipd_batch = ipd_batch * vad_batch_expand
            # (nb,nseg,2nf,nmic-1)
            ipd_batch = torch.sum(ipd_batch, dim=-1)

            gt_batch['doa'] = DOAw_batch
            gt_batch['ipd'] = ipd_batch
            gt_batch['vad_sources'] = vad_batch

            data += [gt_batch]

        return data  # [Input, DOA, IPD, VAD]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.arch.parameters(), lr=0.001)

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8988, last_epoch=-1)
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
        # parser.set_defaults({"trainer.gradient_clip_val": 5, "trainer.gradient_clip_algorithm":"norm"})
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({
            "early_stopping.monitor": "valid/loss",
            "early_stopping.min_delta": 0.01,
            "early_stopping.patience": 10,
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
        # parser_kwargs={"parser_mode": "omegaconf"},
    )
