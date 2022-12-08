import torch
from torch import nn, optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from torchsummary import summary

class GRU(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layer, device):
        super().__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layer = hidden_layer
        self.unit = 8
        self.gru = torch.nn.GRU(in_dim, hidden_layer, self.unit, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_layer, out_dim)



    def forward(self, x):
        x = x[:,:,:,0].permute(0,2,1) # [B,T,N,C] > [B,N,T]
#         print('x shape is ', )
        batch = x.shape[0]
        h_0 = torch.randn(size = (self.unit, batch, self.hidden_layer)).to(self.device)
        output, h1 = self.gru(x, h_0)
        out = self.fc1(output)  # [2, 207, 12]
        out = torch.unsqueeze(out,-1).permute(0,2,1,3) 
#         print('out shape is: ',out.shape)
        return out
    
def main():
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    N_NODE,TIMESTEP_IN,TIMESTEP_OUT,CHANNEL = 207,12,12,1
    model = GRU(in_dim=TIMESTEP_IN, out_dim=TIMESTEP_OUT, hidden_layer=16, device=device)
    summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)
    
if __name__ == '__main__':
    main()  
