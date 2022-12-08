import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchsummary import summary


class LSTNet(nn.Module):
    def __init__(self, data_m, window=24*7, hidRNN=100, hidCNN=100, hidSkip=5, CNN_kernel=6, skip=24, highway_window=24, dropout=0.2, output_fun=None):
        super(LSTNet, self).__init__()
        # self.horizon = horizon
        self.P = window
        self.m = data_m
        self.hidR = hidRNN
        self.hidC = hidCNN
        self.hidS = hidSkip
        self.Ck = CNN_kernel
        self.skip = skip
        self.hw = highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=dropout)
        if (self.skip > 0):
            self.pt = (self.P - self.Ck) // self.skip
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        # self.linear2 = nn.Linear(self.m, self.horizon * self.m)

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = output_fun
        if (output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (output_fun == 'tanh'):
            self.output = F.tanh
        if (output_fun == 'relu'):
            self.output = F.relu

    def forward(self, x):
#         print('input shape is :',x.shape)
        b,t,n,c = x.shape
        x = x.reshape(b,t,n*c)
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn
        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)
        
        res = self.linear1(r)

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z
            
        res = res.unsqueeze(1).unsqueeze(-1) # [B,N] -> [B,1,N,1] BTNC
        # res = self.linear2(res).view(-1, self.horizon, self.m)
        if (self.output):
            res = self.output(res)
            
#         print('output shape is :',res.shape)
        return res
def print_params(model_name, model):
    param_count=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += param.numel()
    print(f'{model_name}, {param_count} trainable parameters in total.')
    # print_params('STNorm',model)  
    return    

def main():
    N_NODE = 228
    TIMESTEP_IN = 16
    TIMESTEP_OUT = 3
    CHANNEL = 1
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model = LSTNet(data_m=N_NODE*CHANNEL,
                 window=TIMESTEP_IN,
                 hidRNN=64,
                 hidCNN=64,
                 CNN_kernel=3,
                 skip=3,
                 highway_window=16
).to(device)
    summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)
    print_params('LSTNet',model) 

if __name__ == '__main__':
    main()