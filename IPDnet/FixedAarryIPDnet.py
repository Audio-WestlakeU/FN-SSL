import torch
import torch.nn as nn
import numpy as np
import Module as at_module
import math
from utils_ import pad_segments,split_segments
class FNblock(nn.Module):
    """
    The implementation of the full-band and narrow-band fusion block
    """
    def __init__(self, input_size, hidden_size=128, dropout=0.2,add_skip_dim=4, is_online=False, is_first=False):
        super(FNblock, self).__init__()
        self.input_size = input_size
        self.full_hidden_size =  hidden_size // 2
        self.is_first = is_first
        self.is_online = is_online
        if self.is_online:
            self.narr_hidden_size = hidden_size
        else:
            self.narr_hidden_size = hidden_size  // 2
        self.dropout = dropout
        self.dropout_full =  nn.Dropout(p=self.dropout)
        self.dropout_narr = nn.Dropout(p=self.dropout)
        if is_first:
            self.fullLstm = nn.LSTM(input_size=self.input_size, hidden_size=self.full_hidden_size, batch_first=True, bidirectional=True)
        else:
             self.fullLstm = nn.LSTM(input_size=self.input_size+add_skip_dim, hidden_size=self.full_hidden_size, batch_first=True, bidirectional=True)
        self.narrLstm = nn.LSTM(input_size=2*self.full_hidden_size+add_skip_dim, hidden_size=self.narr_hidden_size, batch_first=True, bidirectional=not self.is_online)       
    def forward(self, x, fb_skip, nb_skip):
        nb,nt,nf,nc = x.shape
        x = x.reshape(nb*nt,nf,-1)
        x, _ = self.fullLstm(x)
        x = self.dropout_full(x)
        x = torch.cat((x,fb_skip),dim=-1)
        x = x.view(nb,nt,nf,-1).permute(0,2,1,3).reshape(nb*nf,nt,-1) 
        x, _ = self.narrLstm(x)
        x = self.dropout_narr(x)
        x = torch.cat((x,nb_skip),dim=-1)
        x = x.view(nb,nf,nt,-1).permute(0,2,1,3)
        return x
    
class CausCnnBlock(nn.Module): 
	""" 
    Function: Basic causal convolutional block
    """
	# expansion = 1
	def __init__(self, inp_dim,out_dim,cnn_hidden_dim=128, kernel=(3,3), stride=(1,1), padding=(1,2)):
		super(CausCnnBlock, self).__init__()

        # convlutional layers
		self.conv1 = nn.Conv2d(inp_dim, cnn_hidden_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.conv2 = nn.Conv2d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.conv3 = nn.Conv2d(cnn_hidden_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		
        # Time compression
		self.pooling1 = nn.AvgPool2d(kernel_size=(1, 3))
		self.pooling2 = nn.AvgPool2d(kernel_size=(1, 4))                
		self.pad = padding
		self.relu = nn.ReLU(inplace=True)
		self.tanh = nn.Tanh()
	def forward(self, x):
		out = self.conv1(x)
		out = self.relu(out)
		out = out[:, :, :, :-self.pad[1]]
		out = self.pooling1(out)
		out = self.conv2(out)
		out = self.relu(out)
		out = out[:, :, :, :-self.pad[1]]
		out = self.pooling2(out)
		out = self.conv3(out)
		out = out[:, :, :, :-self.pad[1]]
		out = self.tanh(out)
		return out


class IPDnet(nn.Module):
    '''
    The implementation of the IPDnet
    '''
    def __init__(self,input_size=4,hidden_size=128,max_track=2,is_online=True,n_seg=312):
        super(IPDnet, self).__init__()
        self.is_online = is_online
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.block_1 = FNblock(input_size=self.input_size,hidden_size=self.hidden_size,add_skip_dim=self.input_size, is_online=self.is_online, is_first=True)
        self.block_2 = FNblock(input_size=self.hidden_size,hidden_size=self.hidden_size,add_skip_dim=self.input_size,is_online=self.is_online, is_first=False)
        self.cnn_out_dim = 2 * ((input_size // 2) - 1) * max_track
        self.cnn_inp_dim = hidden_size + input_size
        self.conv = CausCnnBlock(inp_dim = self.cnn_inp_dim, out_dim = self.cnn_out_dim)   
        self.n = n_seg
    def forward(self,x,offline_inference=False):
        
        x = x.permute(0,3,2,1)
        nb,nt,nf,nc = x.shape
        ou_frame = nt//12
        # chunk-wise inference for offline 
        if not self.is_online and offline_inference:
            x = split_segments(x, self.n)  # Split into segments of length n
            nb,nseg,seg_nt,nf,nc = x.shape
            x = x.reshape(nb*nseg,seg_nt,nf,nc)
            nb,nt,nf,nc = x.shape        
        
        # reshaping the input
        fb_skip = x.reshape(nb*nt,nf,nc)
        nb_skip = x.permute(0,2,1,3).reshape(nb*nf,nt,nc)
        
        # FN blocks
        x = self.block_1(x,fb_skip=fb_skip,nb_skip=nb_skip)
        x = self.block_2(x,fb_skip=fb_skip, nb_skip=nb_skip)
        nb,nt,nf,nc = x.shape
        x = x.permute(0,3,2,1)
        
        nt2 = nt//12
        x = self.conv(x).permute(0,3,2,1).reshape(nb,nt2,nf,2,-1).permute(0,1,3,2,4)
        if not self.is_online and offline_inference: 
            x = x.reshape(nb//nseg,nt2*nseg,2,nf*2,-1).permute(0,1,3,4,2)
            output = x[:,:ou_frame,:,:,:]
        else:
            output = x.reshape(nb,nt2,2,nf*2,-1).permute(0,1,3,4,2)
        return output
  

if __name__ == "__main__":
    x = torch.randn((1,8,257,280)) 
    # for 2-mic IPDnet
    model = IPDnet()

    # for >2-mic IPDnet, for example, 4-mic IPDnet
    #model = IPDnet(input_size=8,hidden_size=256,max_track=2,is_online=True)
    import time
    ts = time.time()
    y = model(x)
    te = time.time()
    print(model)
    print(y.shape)
    print(te - ts)
    model = model.to('meta')
    x = x.to('meta')
    from torch.utils.flop_counter import FlopCounterMode # requires torch>=2.1.0
    with FlopCounterMode(model, display=False) as fcm:
        y = model(x)
        flops_forward_eval = (fcm.get_total_flops()) / 4.5
        res = y.sum()
        res.backward()
        flops_backward_eval = (fcm.get_total_flops() - flops_forward_eval) / 4.5
    params_eval = sum(param.numel() for param in model.parameters())
    print(f"flops_forward={flops_forward_eval/1e9:.2f}G, flops_back={flops_backward_eval/1e9:.2f}G, params={params_eval/1e6:.2f} M")