import torch
import torch.nn as nn
import numpy as np
import Module as at_module
import math
class FNblock_mean(nn.Module):
    """
    The implementation of the full-band and narrow-band fusion block
    """
    def __init__(self, input_size, hidden_size=128, dropout=0.2,add_skip_dim=4, is_online=False, is_first=False):
        super(FNblock_mean, self).__init__()
        self.input_size = input_size
        self.full_hidden_size =  hidden_size // 2
        self.is_first = is_first
        self.is_online = is_online
        self.linear1 = nn.Linear(2*hidden_size+add_skip_dim,hidden_size)
        self.linear2 = nn.Linear(2*hidden_size+add_skip_dim,hidden_size)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        
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
             self.fullLstm = nn.LSTM(input_size=self.input_size, hidden_size=self.full_hidden_size, batch_first=True, bidirectional=True)
        self.narrLstm = nn.LSTM(input_size=2*self.full_hidden_size+add_skip_dim, hidden_size=self.narr_hidden_size, batch_first=True, bidirectional=not self.is_online)       
    def forward(self, x, skip):
            #shape of x: nb,nv,nf,nt
        nb,nt,nf,nc = x.shape
        x = x.reshape(nb*nt,nf,-1)
        x, _ = self.fullLstm(x)
        x = self.dropout_full(x)
        x = x.view(nb,nt,nf,-1)
        #mean embedding
        x_mean = torch.mean(x.reshape(1,nb//1,nt,nf,-1),dim=1).reshape(1,1,nt,nf,-1)
        x_mean = x_mean.repeat(1,nb//1,1,1,1).reshape(nb,nt,nf,-1)
        x = torch.cat((x,x_mean,skip),dim=-1).permute(0,2,1,3).reshape(nb*nf,nt,-1)
        x = self.relu1(self.linear1(x))        
        x, _ = self.narrLstm(x)
        x = self.dropout_narr(x)
        x = x.view(nb,nf,nt,-1).permute(0,2,1,3)
        #mean embedding
        x_mean = torch.mean(x.reshape(1,nb//1,nt,nf,-1),dim=1).reshape(1,1,nt,nf,-1)
        x_mean = x_mean.repeat(1,nb//1,1,1,1).reshape(nb,nt,nf,-1)
        x = torch.cat((x,x_mean,skip),dim=-1)#.permute(0,2,1,3).reshape(nb*nf,nt,-1)
        x = self.relu2(self.linear2(x)) 
        # print(x.shape)
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


class VariableIPDnet(nn.Module):
    '''
    The implementation of the IPDnet
    '''
    def __init__(self,input_size=4,hidden_size=128,is_online=True):
        super(VariableIPDnet, self).__init__()
        self.is_online = is_online
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.block_1 = FNblock_mean(input_size=self.input_size,hidden_size=self.hidden_size,add_skip_dim=self.input_size, is_online=self.is_online, is_first=True)
        self.block_2 = FNblock_mean(input_size=self.hidden_size,hidden_size=self.hidden_size,add_skip_dim=self.input_size,is_online=self.is_online, is_first=False)
        self.cnn_out_dim = 4 # 2(speaker number)×2(IPD)
        self.cnn_inp_dim = hidden_size
        self.conv = CausCnnBlock(inp_dim = self.cnn_inp_dim, out_dim = self.cnn_out_dim)   
    def forward(self,x):
        x = x.permute(0,3,2,1)
        nb,nt,nf,nc = x.shape
        skip = x
        # FN blocks
        x = self.block_1(x,skip=skip)
        x = self.block_2(x,skip=skip)
        nb,nt,nf,nc = x.shape
        x = x.permute(0,3,2,1)
        
        nt2 = nt//12
        x = self.conv(x).permute(0,3,2,1).reshape(nb,nt2,nf,2,-1).permute(0,1,3,2,4)

        output = x.reshape(1,nb//1,nt2,-1,nf*2).permute(0,2,4,1,3)
        return output
  

if __name__ == "__main__":
    x = torch.randn((3,4,256,280)) #micrphone pair/2-channel[imag/real]/fre/time
    # for variable-array IPDnet
    model = VariableIPDnet()

    import time
    ts = time.time()
    y = model(x)
    print(y.shape)
    te = time.time()
