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
    def __init__(self, n_his, device, num_nodes, dropout=0.3,
                 out_dim=3, residual_channels=16, dilation_channels=16, end_channels=16,
                 kernel_size=2, blocks=1, layers=4, days=288, dims=40, order=2, in_dim=1, normalization="batch"):
        super(DMSTGCN, self).__init__()
        skip_channels = 8
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.normal = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.filter_convs_a1 = nn.ModuleList()
        self.filter_convs_a2 = nn.ModuleList()
        self.filter_convs_a3 = nn.ModuleList()
        self.gate_convs_a1 = nn.ModuleList()
        self.gate_convs_a2 = nn.ModuleList()
        self.gate_convs_a3 = nn.ModuleList()
        self.residual_convs_a = nn.ModuleList()
        self.skip_convs_a = nn.ModuleList()
        self.normal_a1 = nn.ModuleList()
        self.normal_a2 = nn.ModuleList()
        self.normal_a3 = nn.ModuleList()
        self.gconv_a1 = nn.ModuleList()
        self.gconv_a2 = nn.ModuleList()
        self.gconv_a3 = nn.ModuleList()

        
        self.gconv_a2p1 = nn.ModuleList()
        self.gconv_a2p2 = nn.ModuleList()
        self.gconv_a2p3 = nn.ModuleList()

        self.start_conv_a1 = nn.Conv2d(in_channels=in_dim,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1))
        self.start_conv_a2 = nn.Conv2d(in_channels=in_dim,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1))
        self.start_conv_a3 = nn.Conv2d(in_channels=in_dim,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1))     
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        receptive_field = 1

        self.supports_len = 1
        self.nodevec_p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        
        self.nodevec1_a1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec1_a2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec1_a3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec1_ak = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        self.nodevec1_a2p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec1_a2p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec1_a2p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec1_a2pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        self.nodevec2_a1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec2_a2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec2_a3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec2_ak = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        self.nodevec2_a2p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec2_a2p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec2_a2p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec2_a2pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        self.nodevec3_a1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec3_a2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec3_a3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec3_ak = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        self.nodevec3_a2p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec3_a2p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec3_a2p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec3_a2pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)        

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
               
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.filter_convs_a1.append(nn.Conv2d(in_channels=residual_channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))
                self.filter_convs_a2.append(nn.Conv2d(in_channels=residual_channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))                
                self.filter_convs_a3.append(nn.Conv2d(in_channels=residual_channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))                

                self.gate_convs_a1.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs_a2.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs_a3.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))                

                # 1x1 convolution for residual connection
                self.residual_convs_a.append(nn.Conv1d(in_channels=dilation_channels,
                                                       out_channels=residual_channels,
                                                       kernel_size=(1, 1)))
                if normalization == "batch":
                    self.normal.append(nn.BatchNorm2d(residual_channels))
                    self.normal_a1.append(nn.BatchNorm2d(residual_channels))
                    self.normal_a2.append(nn.BatchNorm2d(residual_channels))
                    self.normal_a3.append(nn.BatchNorm2d(residual_channels))
                elif normalization == "layer":
                    self.normal.append(nn.LayerNorm([residual_channels, num_nodes, 16 - receptive_field - new_dilation + 1]))
                    self.normal_a1.append(nn.LayerNorm([residual_channels, num_nodes, 16 - receptive_field - new_dilation + 1]))
                    self.normal_a2.append(nn.LayerNorm([residual_channels, num_nodes, 16 - receptive_field - new_dilation + 1]))
                    self.normal_a3.append(nn.LayerNorm([residual_channels, num_nodes, 16 - receptive_field - new_dilation + 1]))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                
                self.gconv_a1.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.gconv_a2.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.gconv_a3.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))                
                self.gconv_a2p1.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.gconv_a2p2.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.gconv_a2p3.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))               
#                 print('receptive_field ',receptive_field)
        self.relu = nn.ReLU(inplace=True)
        if n_his == 4:
            para = 4
        elif n_his == 8:
            para = 13
        elif n_his == 16:
            para = 38
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels * (para),  #  4 in : 4    /   8 in : 1+4+8
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
        (3869, 19, 98, 4, 1)   [B,T,N,S,C] --> [B,T,N,S（F）]
        input: (B, F, N, T)    
        """
        inputs = inputs[:,:,:,:,0]
#         print('inputs shape is: ',inputs.shape)
        inputs = inputs.permute(0,3,2,1)   #  [B,T,N,S（F）] -> [B,F,N,T]
#         print('inputs shape is :',inputs.shape)
        
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            xo = inputs
        x = self.start_conv(xo[:, [0]])
        x_a1 = self.start_conv_a1(xo[:, [1]])
        x_a2 = self.start_conv_a2(xo[:, [2]])
        x_a3 = self.start_conv_a3(xo[:, [3]])
        skip = 0

        # dynamic graph construction
        adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        
        adp_a1 = self.dgconstruct(self.nodevec1_a1[ind], self.nodevec1_a2, self.nodevec1_a3, self.nodevec1_ak)
        adp_a2p1 = self.dgconstruct(self.nodevec1_a2p1[ind], self.nodevec1_a2p2, self.nodevec1_a2p3, self.nodevec1_a2pk)
        adp_a2 = self.dgconstruct(self.nodevec2_a1[ind], self.nodevec2_a2, self.nodevec2_a3, self.nodevec2_ak)
        adp_a2p2 = self.dgconstruct(self.nodevec2_a2p1[ind], self.nodevec2_a2p2, self.nodevec2_a2p3, self.nodevec2_a2pk)
        adp_a3 = self.dgconstruct(self.nodevec3_a1[ind], self.nodevec3_a2, self.nodevec3_a3, self.nodevec3_ak)
        adp_a2p3 = self.dgconstruct(self.nodevec3_a2p1[ind], self.nodevec3_a2p2, self.nodevec3_a2p3, self.nodevec3_a2pk)        

        new_supports = [adp]
        
        new_supports1_a = [adp_a1]
        new_supports1_a2p = [adp_a2p1]
        new_supports2_a = [adp_a2]
        new_supports2_a2p = [adp_a2p2]
        new_supports3_a = [adp_a3]
        new_supports3_a2p = [adp_a2p3]        

        for i in range(self.blocks * self.layers):
            # tcn for primary part
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # tcn for auxiliary part
            residual_a1 = x_a1   ###  auxiliary 1
            filter_a1 = self.filter_convs_a1[i](residual_a1)
            filter_a1 = torch.tanh(filter_a1)
            gate_a1 = self.gate_convs_a1[i](residual_a1)
            gate_a1 = torch.sigmoid(gate_a1)
            x_a1 = filter_a1 * gate_a1
            
            residual_a2 = x_a2   ###  auxiliary 2
            filter_a2 = self.filter_convs_a2[i](residual_a2)
            filter_a2 = torch.tanh(filter_a2)
            gate_a2 = self.gate_convs_a2[i](residual_a2)
            gate_a2 = torch.sigmoid(gate_a2)
            x_a2 = filter_a2 * gate_a2
            
            residual_a3 = x_a3   ###  auxiliary 3
            filter_a3 = self.filter_convs_a3[i](residual_a3)
            filter_a3 = torch.tanh(filter_a3)
            gate_a3 = self.gate_convs_a3[i](residual_a3)
            gate_a3 = torch.sigmoid(gate_a3)
            x_a3 = filter_a3 * gate_a3            
            

            # skip connection   
            s = x
#             print('s shape：',s.shape)
            s = self.skip_convs[i](s)
            if isinstance(skip, int):  # B F N T
                skip = s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]).contiguous()
            else:
                skip = torch.cat([s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], dim=1).contiguous()

            # dynamic graph convolutions
            x = self.gconv[i](x, new_supports)
            x_a1 = self.gconv_a1[i](x_a1, new_supports1_a)
            x_a2 = self.gconv_a2[i](x_a2, new_supports2_a)
            x_a3 = self.gconv_a3[i](x_a3, new_supports3_a)

            # multi-faceted fusion module
            x_a2p1 = self.gconv_a2p1[i](x_a1, new_supports1_a2p)
            x_a2p2 = self.gconv_a2p2[i](x_a2, new_supports2_a2p)
            x_a2p3 = self.gconv_a2p3[i](x_a3, new_supports3_a2p)
            x = x_a2p1 + x_a2p2 + x_a2p3 + x

            # residual and normalization
            x_a1 = x_a1 + residual_a1[:, :, :, -x_a1.size(3):]
            x_a2 = x_a2 + residual_a2[:, :, :, -x_a2.size(3):]
            x_a3 = x_a3 + residual_a3[:, :, :, -x_a3.size(3):]
            
            x = x + residual[:, :, :, -x.size(3):]
            x = self.normal[i](x)
            x_a1 = self.normal_a1[i](x_a1)
            x_a2 = self.normal_a2[i](x_a2)
            x_a3 = self.normal_a3[i](x_a3)

        # output layer
        x = F.relu(skip)
#         print('x shape is:',x.shape)
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
    model =  DMSTGCN(n_his, device, num_nodes=N_NODE, dropout=0.3,
                 out_dim=TIMESTEP_OUT, residual_channels=16, dilation_channels=16, end_channels=512,
                 kernel_size=2, blocks=1, layers=2, days=288, dims=40, order=2, in_dim=1, normalization="batch").to(device)
    X = torch.Tensor(torch.randn(8,4,98,4)).to(device)
    ind = np.random.randint(1,48,8)
    print('ind :',ind)
    print('ind shape is :',ind.shape)
    model(X,ind)
#     summary(model, [(CHANNEL, N_NODE, TIMESTEP_IN),(TIMESTEP_IN,)], device=device)
    print_params('DMSTGCN',model)


if __name__ == '__main__':
    main()    