import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.nn import Parameter

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DMSTGCN(nn.Module):
    def __init__(self, device, num_nodes, source_num, dropout=0.3,
                 out_dim=3, residual_channels=16, dilation_channels=16, end_channels=16,
                 kernel_size=2, blocks=1, layers=4, days=288, dims=40, order=2, in_dim=1, normalization="batch"):
        super(DMSTGCN, self).__init__()
        skip_channels = 8
        self.source_num = source_num
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.normal = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.gconv_a2p = nn.ModuleList()
        self.start_conv = nn.ModuleList()
        
        self.nodevec_a1 = nn.ParameterList()
        self.nodevec_a2 = nn.ParameterList()
        self.nodevec_a3 = nn.ParameterList()
        self.nodevec_ak = nn.ParameterList()
        self.nodevec_a2p1 = nn.ParameterList()
        self.nodevec_a2p2 = nn.ParameterList()
        self.nodevec_a2p3 = nn.ParameterList()
        self.nodevec_a2pk = nn.ParameterList()
        
        
        for i in range(source_num):
            self.filter_convs.append(nn.ModuleList())
            self.gate_convs.append(nn.ModuleList())
            self.normal.append(nn.ModuleList())
            self.gconv.append(nn.ModuleList())
            self.start_conv.append(nn.Conv2d(in_channels=in_dim,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1)))
        for i in range(source_num-1):
            self.gconv_a2p.append(nn.ModuleList())
            
            
        self.skip_convs = nn.ModuleList()
        

        receptive_field = 1

        self.supports_len = 1

        self.nodevec_p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)     
        
        for i in range(source_num-1):
            self.nodevec_a1.append(nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device))
            self.nodevec_a2.append(nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device))
            self.nodevec_a3.append(nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device))
            self.nodevec_ak.append(nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device))
            self.nodevec_a2p1.append(nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device))
            self.nodevec_a2p2.append(nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device))
            self.nodevec_a2p3.append(nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device))
            self.nodevec_a2pk.append(nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device))            
            
   

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
               # dilated convolutions
                for s in range(source_num):
                    self.filter_convs[s].append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                    self.gate_convs[s].append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                    # 1x1 convolution for residual connection
                    if normalization == "batch":
                        self.normal[s].append(nn.BatchNorm2d(residual_channels))
                    elif normalization == "layer":    
                        self.normal[s].append(nn.LayerNorm([residual_channels, num_nodes, 16 - receptive_field - new_dilation + 1]))
                    self.gconv[s].append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                


                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

                for s2 in range(source_num-1):
                    self.gconv_a2p[s2].append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))

        self.relu = nn.ReLU(inplace=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels * (15+13+9+1),
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
#         print('receptive_field ',self.receptive_field)

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def forward(self, inputs, ind):
        """
        (3869, 19, 98, 4 or 3, 1)   [B,T,N,S,C] --> [B,T,N,S（F）]
        input: (B, F, N, T)    
        """
        inputs = torch.squeeze(inputs).permute(0,3,2,1)   #  [B,T,N,S（F）] -> [B,F,N,T]
#         print('inputs shape is :',inputs.shape)
        
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            xo = inputs
        x =[0]*(self.source_num)
        for s in range(self.source_num):
            x[s] = self.start_conv[s](xo[:, [s]])
            
            
        skip = 0

        # dynamic graph construction
        adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        
        adp_a = [0]*(self.source_num-1)
        adp_a2p = [0]*(self.source_num-1)
        for s2 in range(self.source_num-1):
            adp_a[s2]=self.dgconstruct(self.nodevec_a1[s2][ind], self.nodevec_a2[s2], self.nodevec_a3[s2], self.nodevec_ak[s2])
            adp_a2p[s2] = self.dgconstruct(self.nodevec_a2p1[s2][ind], self.nodevec_a2p2[s2], self.nodevec_a2p3[s2], self.nodevec_a2pk[s2])
              

        new_supports = [adp]
        
        new_supports_a = [0]*(self.source_num-1)
        new_supports_a2p = [0]*(self.source_num-1)
        for s2 in range(self.source_num-1):
            new_supports_a[s2] = [adp_a[s2]]
            new_supports_a2p[s2] = [adp_a2p[s2]]
               
        
        for i in range(self.blocks * self.layers):
            # tcn for primary part
            # tcn for auxiliary part
            residual = [0]*self.source_num
            filter = [0]*self.source_num
            gate = [0]*self.source_num
            for s in range(self.source_num):
                residual[s] = x[s]
                filter[s] = self.filter_convs[s][i](residual[s])
                filter[s] = torch.tanh(filter[s])
                gate[s] = self.gate_convs[s][i](residual[s])
                gate[s] = torch.sigmoid(gate[s])
                x[s] = filter[s] * gate[s]


            # skip connection   
            s = x[0]
#             print('s shape：',s.shape)
            s = self.skip_convs[i](s)
            if isinstance(skip, int):  # B F N T
                skip = s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]).contiguous()
            else:
                skip = torch.cat([s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], dim=1).contiguous()

            # dynamic graph convolutions
            
            for s in range(self.source_num):
                x[s] = self.gconv[s][i](x[s], new_supports)


            # multi-faceted fusion module
            x_ap = [0]*(self.source_num-1)
            temp = x[0]
            for s2 in range(self.source_num-1):
                x_ap[s2] = self.gconv_a2p[s2][i](x[s2+1], new_supports_a2p[s2])
                temp+=x_ap[s2]
            x[0] = x[0]+temp
                


            # residual and normalization
            for s2 in range(self.source_num-1):
                x[s2+1] = x[s2+1] = residual[s2+1][:, :, :, -x[s2+1].size(3):]
                
            x[0] = x[0] + residual[0][:, :, :, -x[0].size(3):]
            for s in range(self.source_num):
                x[s] = self.normal[s][i](x[s])

        # output layer
        x = F.relu(skip)
        print('x shape is:',x.shape)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = torch.unsqueeze(x,-1)
#         print('out shape is:', x.shape)
        return x
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print(param.shape)



def print_params(model_name, model):
    param_count=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += param.numel()
    print(f'{model_name}, {param_count} trainable parameters in total.')
    return    

import sys    
import numpy as np    
def main():
    CHANNEL = 1
    N_NODE = 98
    N_SOURCE = 4
    TIMESTEP_IN = 16
    TIMESTEP_OUT = 3
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '1'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model =  DMSTGCN(device, num_nodes=N_NODE, source_num = 4, dropout=0.3,
                 out_dim=TIMESTEP_OUT, residual_channels=16, dilation_channels=16, end_channels=512,
                 kernel_size=2, blocks=1, layers=4, days=288, dims=40, order=2, in_dim=1, normalization="batch").to(device)
    X = torch.Tensor(torch.randn(8,4,98,16)).to(device)
    ind = np.random.randint(1,48,8)
    print('ind :',ind)
    print('ind shape is :',ind.shape)
    model(X,ind)
#     summary(model, [(CHANNEL, N_NODE, TIMESTEP_IN),(TIMESTEP_IN,)], device=device)
    print_params('DMSTGCN',model)


if __name__ == '__main__':
    main()    