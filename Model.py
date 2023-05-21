import torch
import torch.nn as nn

import Module as at_module

class FNblock(nn.Module):
    """ 
    """
    def __init__(self, input_size, hidden_size=256, dropout=0.2, is_online=False, is_first=False):
        """the block of full-band and narrow-band fusion
        """
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
        self.fullLstm = nn.LSTM(input_size=self.input_size, hidden_size=self.full_hidden_size, batch_first=True, bidirectional=True)
        if self.is_first:
              self.narrLstm = nn.LSTM(input_size=2*self.full_hidden_size+self.input_size, hidden_size=self.narr_hidden_size, batch_first=True, bidirectional=not self.is_online)
        else:
            self.narrLstm = nn.LSTM(input_size=2*self.full_hidden_size, hidden_size=self.narr_hidden_size, batch_first=True, bidirectional=not self.is_online)
        
    def forward(self, x, nb_skip=None, fb_skip=None):
            #shape of x: nb,nv,nf,nt
        nb,nt,nf,nc = x.shape
        nb_skip = x.permute(0,2,1,3).reshape(nb*nf,nt,-1)
        x = x.reshape(nb*nt,nf,-1)
        if not self.is_first:
            x = x + fb_skip
        x, _ = self.fullLstm(x)
        fb_skip = x
        x = self.dropout_full(x)
        x = x.view(nb,nt,nf,-1).permute(0,2,1,3).reshape(nb*nf,nt,-1)
        if self.is_first:  
            x = torch.cat((x,nb_skip),dim=-1)
        else:
            x = x + nb_skip
        x, _ = self.narrLstm(x)
        nb_skip = x
        x = self.dropout_narr(x)
        x = x.view(nb,nf,nt,-1).permute(0,2,1,3)
        return x, fb_skip, nb_skip

       
class FN_SSL(nn.Module):
    """ 
    """
    def __init__(self,input_size=4,hidden_size=256,is_online=True):
        """the block of full-band and narrow-band fusion
        """
        super(FN_SSL, self).__init__()
        self.is_online = is_online
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.block_1 = FNblock(input_size=self.input_size,is_online=self.is_online, is_first=True)
        self.block_2 = FNblock(input_size=self.hidden_size,is_online=self.is_online, is_first=False)
        self.block_3 = FNblock(input_size=self.hidden_size,is_online=self.is_online, is_first=False)        
        self.emb2ipd = nn.Linear(256,2)
        self.pooling = nn.AvgPool2d(kernel_size=(12, 1))
        self.tanh = nn.Tanh()
    def forward(self,x):
        x = x.permute(0,3,2,1)
        nb,nt,nf,nc = x.shape       
        x, fb_skip, nb_skip = self.block_1(x)
        x, fb_skip, nb_skip = self.block_2(x,fb_skip=fb_skip, nb_skip=nb_skip)
        x, fb_skip, nb_skip = self.block_3(x,fb_skip=fb_skip, nb_skip=nb_skip)  
        #nb nt nf nc
        x = x.permute(0,2,1,3).reshape(nb*nf,nt,-1)   
        ipd = self.pooling(x)
        ipd = self.tanh(self.emb2ipd(ipd))
        _, nt2, _ = ipd.shape
        ipd = ipd.view(nb,nf,nt2,-1)
        ipd = ipd.permute(0,2,1,3)
        ipd_real = ipd[:,:,:,0]
        ipd_image = ipd[:,:,:,1]
        ipd_final = torch.cat((ipd_real,ipd_image),dim=2)
        return ipd_final

class FN(nn.Module):
    """ 
    """
    def __init__(self,input_size=4,hidden_size=256,is_online=True):
        """the block of full-band and narrow-band fusion
        """
        super(FN, self).__init__()
        self.arch = FN_SSL()
    def forward(self,x):
        return self.arch(x)


class LSTM_19(nn.Module):
    """ 19 LSTM model
    """
    def __init__(self,):
        """Two layers of BiLSTMs & one fully connected layer
        Args:
            input_size: the input size for the features of the first BiLSTM layer
            output_size: the output size for the features of the last BiLSTM layer
            hidden_size: the hidden size of each BiLSTM layer. Defaults to (256, 128).
        """
        super(LSTM_19, self).__init__()
        self.input_size = 4
        self.output_size = 2 
        self.hidden_size = [256,256]
        self.dropout = 0.2


        self.fullbandLstm1 = nn.LSTM(input_size=4, hidden_size=128, batch_first=True, bidirectional=True)
        self.nbbandLstm1 = nn.LSTM(input_size=256+4, hidden_size=256, batch_first=True, bidirectional=False)
        self.fullbandLstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.nbbandLstm2 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True, bidirectional=False)
        self.fullbandLstm3 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.nbbandLstm3 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True, bidirectional=False)

        self.emb2ipd = nn.Linear(256,2)
        self.pooling = nn.AvgPool2d(kernel_size=(12, 1))
        #self.ipdLstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True, bidirectional=False)
        #self.ipdlinear = nn.Linear(512, 512)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        #self.linear1_3 = nn.Linear(256, 256)
        #self.lstm1_3 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True, bidirectional=False)  # type:ignore                     
        
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.dropout3 = nn.Dropout(p=self.dropout)
        self.dropout4 = nn.Dropout(p=self.dropout)
        self.dropout5 = nn.Dropout(p=self.dropout)
        self.dropout6 = nn.Dropout(p=self.dropout)
        # self.dropout7 = nn.Dropout(p=self.dropout)
        # self.dropout8 = nn.Dropout(p=self.dropout)
        # self.dropout3_x = nn.Dropout(p=self.dropout)
        # self.dropout4_x = nn.Dropout(p=self.dropout)

    def forward(self, x):
        """forward
        Args:
            x: shape [batch, seq, input_size]
        Returns:
            Tensor: shape [batch, seq, output_size]
        """
        x = x.permute(0,3,2,1)
        nb,nt,nf,nc = x.shape
        nb_skip = x.permute(0,2,1,3).reshape(nb*nf,nt,-1)
        print(x.shape)
		#--------- 1 fullband-subband layer----------------
        x = x.reshape(nb*nt,nf,-1)
        x, _ = self.fullbandLstm1(x)
		#skip 
        skip_emb_fb = x
        x = self.dropout1(x)
        x = x.view(nb,nt,nf,-1).permute(0,2,1,3)
        x = x.reshape(nb*nf,nt,-1)
        x = torch.cat((x,nb_skip),dim=-1)
        x, _ = self.nbbandLstm1(x)
        skip_emb_nb = x
        x = self.dropout2(x)      
		#--------- 2 fullband-subband layer----------------
        x = x.view(nb,nf,nt,-1).permute(0,2,1,3)
        x = x.reshape(nb*nt,nf,-1)
        #skip
        x = x + skip_emb_fb
        x, _ = self.fullbandLstm2(x)
        skip_emb_fb = x
        x = self.dropout3(x)

        x = x.view(nb,nt,nf,-1).permute(0,2,1,3)
        x = x.reshape(nb*nf,nt,-1)
        x = x + skip_emb_nb
        x, _ = self.nbbandLstm2(x)
        skip_emb_nb = x
        x = self.dropout4(x)
		#--------- 3 fullband-subband layer----------------
        x = x.view(nb,nf,nt,-1).permute(0,2,1,3)
        x = x.reshape(nb*nt,nf,-1)
        x = x + skip_emb_fb
        x, _ = self.fullbandLstm3(x)
        x = self.dropout5(x)

        x = x.view(nb,nt,nf,-1).permute(0,2,1,3)
        x = x.reshape(nb*nf,nt,-1)
        x = x + skip_emb_nb
        x, _ = self.nbbandLstm3(x)
        x = self.dropout6(x)
		
		#--------------------------------------------------
        ipd = self.pooling(x)
        ipd = self.tanh(self.emb2ipd(ipd))

        _, nt2, nn = ipd.shape
        ipd = ipd.view(nb,nf,nt2,-1)
        ipd = ipd.permute(0,2,1,3)
        ipd_real = ipd[:,:,:,0]
        ipd_image = ipd[:,:,:,1]
        ipd_final = torch.cat((ipd_real,ipd_image),dim=2)
        #print(ipd_final)
        # ipd_final, _ = self.ipdLstm(ipd_final)
        # ipd_final = self.tanh(self.ipdlinear(ipd_final))
        return ipd_final
    

if __name__ == "__main__":
	import torch
	input = torch.randn((2,4,256,298)).cuda()
	net = FN_SSL().cuda()
	ouput = net(input)
	print(ouput.shape)
	print('# parameters:', sum(param.numel() for param in net.parameters()))