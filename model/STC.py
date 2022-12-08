import torch
import torch.nn as nn
# from One_hot_encoder import One_hot_encoder
import torch.nn.functional as F
import numpy as np
import sys
from torchsummary import summary
from torch.nn import Parameter

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal) or S, N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context

class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, T, N, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V) # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4) #[B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim) # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        return output


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4) # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V) #[B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4) #[B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim) # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        return output 

 
    


class STransformer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, gpu, dropout):        
        super(STransformer, self).__init__()
        # Spatial Embedding
#         self.adj = adj
#         self.D_S = adj.to('cuda:0')
        self.gpu = gpu
#         self.embed_liner = nn.Linear(adj.shape[0], embed_size)
        
        self.attention = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        # value, key, query:  [B, N, T, C]        
        # Spatial Embedding 
        B, N, T, C = query.shape
        D_S = get_sinusoid_encoding_table(N, C).to(self.gpu) 
        D_S = D_S.expand(B, T, N, C) #[B, T, N, C] 
        D_S = D_S.permute(0, 2, 1, 3) #[B, N, T, C]
       
        # Spatial Transformer 
        query = query + D_S
        attention = self.attention(query, query, query)  #(B, N, T, C)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))
        
        out = U_S
        return out  #(B, N, T, C)    

class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num,forward_expansion, gpu, dropout):
        super(TTransformer, self).__init__()
        
        # Temporal embedding One hot
        self.time_num = time_num
        
#         self.one_hot = One_hot_encoder(embed_size, time_num)          # temporal embedding  by one-hot   OR：
#         self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding by nn.Embedding
        
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.gpu = gpu

    def forward(self, value, key, query):
        B, N, T, C = query.shape
        
#         D_T = self.one_hot(t, N, T)                          # temporal embedding by one-hot
#         D_T = self.temporal_embedding(torch.arange(0, T).to('cuda:0'))    # temporal embedding by nn.Embedding
        D_T = get_sinusoid_encoding_table(T, C).to(self.gpu)
        D_T = D_T.expand(B, N, T, C)

        # temporal embedding add to query。 it is concatenated in original paper
        query = query + D_T    
        attention = self.attention(query, query, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

    
    
### STBlock

class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, time_num, forward_expansion, gpu, dropout):
        super(STTransformerBlock, self).__init__()
        self.STransformer = STransformer(embed_size, heads, forward_expansion, gpu, dropout)
        self.TTransformer = TTransformer(embed_size, heads, time_num, forward_expansion, gpu, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query):
    # value,  key, query: [N, T, C] [B, N, T, C]
        # Add skip connection,run through normalization and finally dropout
        x1 = self.norm1(self.STransformer(value, key, query) + query) #(B, N, T, C)
        x2 = self.dropout( self.norm2(self.TTransformer(x1, x1, x1) + x1) ) 
        return x2




### Encoder
class Encoder(nn.Module):
    # multi layer ST-Transformer Block
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        time_num,
        forward_expansion,
        gpu,
        dropout
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.gpu = gpu
        self.layers = nn.ModuleList(
            [
                STTransformerBlock(
                    embed_size,
                    heads,
                    time_num,
                    forward_expansion,
                    gpu,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
    # x:  [B, N, T, C]
        out = self.dropout(x)        
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out, out, out)
        return out     
    


### Transformer   
class STTransformer(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        time_num,
        forward_expansion, ##？
        gpu,
        dropout
    ):
        super(STTransformer, self).__init__()
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            time_num,
            forward_expansion,
            gpu,
            dropout
        )
        self.gpu = gpu

    def forward(self, src): 
        ## scr:    [B, N, T, C]
        enc_src = self.encoder(src) 
        return enc_src # [B, N, T, C]


### ST Transformer: Total Model
class CMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, NT, S, C = input_Q.shape
        # [B, NT, S, C] --> [B, NT, S, h * d_k] --> [B, NT, S, h, d_k] --> [B, h, NT, S, d_k]
        Q = self.W_Q(input_Q).view(B, NT, S, self.heads, self.head_dim).permute(0,3,1,2,4)  # Q: [B, h, NT, S, d_k]
        K = self.W_K(input_K).view(B, NT, S, self.heads, self.head_dim).permute(0,3,1,2,4)  # K: [B, h, S, NT, d_k]
        V = self.W_V(input_V).view(B, NT, S, self.heads, self.head_dim).permute(0,3,1,2,4)  # V: [B, h, S, NT, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V) # [B, h, NT, S, d_k]
        context = context.permute(0, 2, 3, 1, 4) #[B, NT, S, h, d_k]
        context = context.reshape(B, NT, S, self.heads * self.head_dim) # [B, NT, S, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        return output
    
class CTransformer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, gpu, dropout):        
        super(CTransformer, self).__init__()
        # Spatial Embedding
#         self.adj = adj
#         self.D_S = adj.to('cuda:0')
        self.gpu = gpu
#         self.embed_liner = nn.Linear(adj.shape[0], embed_size)
        
        self.attention = CMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        # value, key, query: [N*T, S, C]  [B, N*T, S, C]        
        B, NT, S, C = query.shape
        D_S = get_sinusoid_encoding_table(S, C).to(self.gpu) 
        D_S = D_S.expand(B, NT, S, C) #[B, NT, S, C]    
        # Spatial Transformer 
        query = query + D_S
        attention = self.attention(query, query, query) #(B, NT, S, C)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))
        
        out = U_S
        return out  #(B, NT, S, C)

class STC(nn.Module):
    def __init__(
        self, 
        in_channels,
        source_num,
        embed_size, 
        time_num,
        num_layers,
        T_dim,
        output_T_dim,  
        heads,    
        forward_expansion, 
        gpu,
        dropout = 0 
    ):        
        super(STC, self).__init__()
        self.source_num = source_num
        self.forward_expansion = forward_expansion
        # C -> embed_size
        self.conv1 = nn.ModuleList()
#         self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.STtrans = nn.ModuleList()
        for i in range(source_num):
            self.conv1.append(nn.Conv2d(in_channels, embed_size, 1))
            self.STtrans.append(STTransformer(embed_size, num_layers, heads, time_num,forward_expansion,gpu,dropout = 0))
#         self.STTransformer = STTransformer(embed_size, num_layers, heads, time_num,forward_expansion,gpu,dropout = 0)
        self.Ctrans = CTransformer(embed_size, heads, forward_expansion, gpu, dropout)
        
        # time 16 -> 3
        self.conv2 = nn.Conv3d(in_channels=T_dim,
                                    out_channels=output_T_dim,
                                    kernel_size=(1,1,1),
                                    bias=True)
        
        # channel to 1 
        self.conv3 = nn.Conv3d(in_channels=embed_size,
                                    out_channels=1,
                                    kernel_size=(1,1,1),
                                    bias=True)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
    
    def forward(self, x):  # (b,t,n,s,c)
        
        # C:channel  N:nodes  T:time
        st_out = []
        for i in range(self.source_num):
            st_input = x[:,:,:,i,:] #[B, T, N, S, 2]—> [B, T, N, 2]
            st_input = st_input.permute(0,3,2,1) # [B, T, N, 2] > [B, 2, T, N]
            st_input = self.conv1[i](st_input) #  [B, 2, N, T] > [B, C, N, T]
            st_input = st_input.permute(0,2,3,1) # [B, C, N, T] -> [B, N, T, C]
            sttran_single = self.STtrans[i](st_input)
            sttran_single = torch.unsqueeze(sttran_single,-2)  #(B, N, T, C) -> (B, N, T, 1, C)
            st_out.append(sttran_single)
        st_out = torch.cat(st_out,-2)  #(B, N, T, 1, C) -> (B, N, T, S, C)
        B, N, T, S, C = st_out.shape
        c_in = torch.reshape(st_out,(B,N*T,S,C))  # (B, N, T, S, C)-> (B, N*T, S, C)
    
        c_out = self.Ctrans(c_in,c_in,c_in) # (B, N*T, S, C)
        c_out = torch.reshape(c_out,(B,N,T,S,C))
        out = c_out.permute(0,2,1,3,4) # (B, N, T, S, C)-> (B, T, N, S, C)
        out = self.relu1(self.conv2(out))  # (B, T, N, S, C) ->  (B, 3, N, S, C) 
        out = out.permute(0,4,1,2,3)
        out = self.conv3(out)  # (B, C, 3, N, S) ->  (B, 1, 3, N, S) 
        out = out.permute(0,2,3,4,1)
        return out #(b,t,n,s,c)

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
        
        
def main():
    CHANNEL = 2
    INPUT_STEP = 16
    PRED_STEP =  3
    N_NODE = 98
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '7'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model = STC(in_channels = CHANNEL, source_num=4, embed_size = 64, time_num = 16 , num_layers = 1, T_dim = INPUT_STEP, output_T_dim = PRED_STEP, heads = 8,forward_expansion = 32, gpu = device, dropout = 0).to(device)
    summary(model, (INPUT_STEP, N_NODE,4, CHANNEL), device=device)   
    
    print_params("STTransformer_sinembedding", model)

if __name__ == '__main__':
    main()        
        