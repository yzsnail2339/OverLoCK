import torch
from torch import nn
from torch.cuda.amp import autocast
from collections import deque
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from mamba_ssm import Mamba
import copy

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2).contiguous()
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims).contiguous()

        return out
    
class BiPixelMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)


        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,
        )

     
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)


        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,

        )

      
        # adjust the window size here to fit the feature map
        self.p = p*5
        self.p1 = 5*p
        self.p2 = 7*p
        self.p3 = 6*p
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
       


    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)

        B, C = x.shape[:2]

        assert C == self.dim
        img_dims = x.shape[2:]

        if ll == 5: #3d
         
            Z,H,W = x.shape[2:]

            if Z%self.p1==0 and H%self.p2==0 and W%self.p3==0:
                x_div = x.reshape(B, C, Z//self.p1, self.p1, H//self.p2, self.p2, W//self.p3, self.p3)
                x_div = x_div.permute(0, 3, 5, 7, 1, 2, 4, 6).contiguous().view(B*self.p1*self.p2*self.p3, C, Z//self.p1, H//self.p2, W//self.p3)
            else:
                x_div = x

        elif ll == 4: #2d
            H,W = x.shape[2:]

            if H%self.p==0 and W%self.p==0:                
                x_div = x.reshape(B, C, H//self.p, self.p, W//self.p, self.p).permute(0, 3, 5, 1, 2, 4).contiguous().view(B*self.p*self.p, C, H//self.p, W//self.p)            
            else:
                x_div = x
        

        NB = x_div.shape[0]
        if ll == 5: #3d
            NZ,NH,NW = x_div.shape[2:]
        else:
            NH,NW = x_div.shape[2:]

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(NB, C, n_tokens).transpose(-1, -2).contiguous()
        x_norm = self.norm(x_flat)

        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))

        if ll == 5:
            if Z%self.p1==0 and H%self.p2==0 and W%self.p3==0:
                x_out = x_out.transpose(-1, -2).reshape(B, self.p1, self.p2, self.p3, C, NZ, NH, NW).permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous().reshape(B, C, *img_dims)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims).contiguous()
        if ll == 4:
            if H%self.p==0 and W%self.p==0:
                x_out = x_out.transpose(-1, -2).reshape(B, self.p, self.p, C, NH, NW).permute(0, 3, 4, 1, 5, 2).contiguous().reshape(B, C, *img_dims)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims).contiguous()
        out = x_out + x

        return out


class BiWindowMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.p = p
        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,
        )

     
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)

        

        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,

        )

      

       
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
       


    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)
     
       

        B, C = x.shape[:2]

        assert C == self.dim
   
        img_dims = x.shape[2:]



        if ll == 5: #3d
            
            Z,H,W = x.shape[2:]

            if Z%self.p==0 and H%self.p==0 and W%self.p==0:
                pool_layer = nn.AvgPool3d(self.p, stride=self.p)
                x_div = pool_layer(x)
            else:
                x_div = x

        elif ll == 4: #2d

            H,W = x.shape[2:]
            if self.p==0:
                self.p = 1
            if H%self.p==0 and W%self.p==0:
                pool_layer = nn.AvgPool2d(self.p, stride=self.p)
                x_div = pool_layer(x)
            else:
                x_div = x
        

      
        if ll == 5: #3d
            NZ,NH,NW = x_div.shape[2:]
        else:
            NH,NW = x_div.shape[2:]

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(B, C, n_tokens).transpose(-1, -2).contiguous()
        x_norm = self.norm(x_flat)

        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))

        if ll == 5:
            if Z%self.p==0 and H%self.p==0 and W%self.p==0:
                unpool_layer = nn.Upsample(scale_factor=self.p, mode='nearest')
                x_out = x_out.transpose(-1, -2).reshape(B, C, NZ, NH, NW).contiguous()
                x_out = unpool_layer(x_out)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims).contiguous()
        if ll == 4:
            if self.p==0:
                self.p = 1
            if H%self.p==0 and W%self.p==0:
                unpool_layer = nn.Upsample(scale_factor=self.p, mode='nearest')
                x_out = x_out.transpose(-1, -2).reshape(B, C, NH, NW).contiguous()
                x_out = unpool_layer(x_out)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims).contiguous()
                
        out = x_out + x

        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query=query, key=key, value=value)
        return attn_output

class MemoryBank(nn.Module):
    def __init__(self, dim, depth=2, bank_size=10, num_heads=8):
        super().__init__()
        self.dim = dim
        self.bank_size = bank_size
        self.memory = deque(maxlen=bank_size)

        self.mamba_layers = nn.ModuleList()
        # Mamba and Transformer encoder
        for i in range(depth):
            self.mamba_layers.append(BiWindowMambaLayer(dim, i+1))
            self.mamba_layers.append(BiPixelMambaLayer(dim, i+1))


        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Cross Attention
        self.cross_attention = CrossAttention(dim=dim, num_heads=num_heads)
        
        # Final projection
        self.proj = nn.Linear(dim * 2, dim)

        self.fusion_norm = nn.LayerNorm(dim)
        self.fusion_scale = nn.Parameter(torch.ones(1))

    def add(self, x):
        self.memory.append(x)
        
    @autocast(enabled=False)
    def forward(self):
        if len(self.memory) == 0:
            return None

        # Stack成 [B, T, C, H, W]
        mem = torch.stack(list(self.memory), dim=1)
        B, T, C, H, W = mem.shape

        # 先flatten + mean处理给Transformer
        mem_flat = mem.flatten(3).mean(-1)  # [B, T, C]
        mem_flat = mem_flat.to(dtype=torch.float32)  # ✅ 保证类型一致

        # Mamba部分保留空间结构
        mamba_out = mem.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        for mamba_layer in self.mamba_layers:
            mamba_out = mamba_layer(mamba_out)  # [B, C, T, H, W]
        mamba_out = mamba_out.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]

        # Transformer处理flatten后的特征
        transformer_out = self.transformer(mem_flat)  # [B, T, C]

        # attention的key/value也需要flatten+mean
        att_mamba_out = mamba_out.flatten(3).mean(-1)  # [B, T, C]

        # Cross Attention: 双向
        feature_TM = self.cross_attention(transformer_out, att_mamba_out, att_mamba_out)  # T->M
        feature_MT = self.cross_attention(att_mamba_out, transformer_out, transformer_out)  # M->T


        # 拼接+融合
        final_feature = torch.cat([feature_TM, feature_MT], dim=-1)  # [B, T, 2C]
        final_feature = self.proj(final_feature)  # [B, T, C]

        # 取最后一个time step作为最终融合特征
        fusion_vec = final_feature[:, -1]  # [B, C]

        fusion_vec = self.fusion_norm(fusion_vec) * self.fusion_scale

        # 将fusion_vec扩展成带空间结构的特征图
        fusion_vec = fusion_vec.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # 用空间特征 mamba_out[:, -1] 作为基础特征，叠加上融合后的全局特征
        final_mm = mamba_out[:, -1] + fusion_vec  # [B, C, H, W]

        self.memory.clear()

        return final_mm


class MemoryBank_xt(nn.Module):
    def __init__(self, dim, depth=2, bank_size=10):
        super().__init__()
        self.dim = dim
        self.bank_size = bank_size
        self.memory = deque(maxlen=bank_size)

        self.mamba_layers = nn.ModuleList()
        # Mamba and Transformer encoder
        for i in range(depth):
            self.mamba_layers.append(BiWindowMambaLayer(dim, i+1))
            self.mamba_layers.append(BiPixelMambaLayer(dim, i+1))

    def add(self, x):
        self.memory.append(x)
        
    @autocast(enabled=False)
    def forward(self):
        if len(self.memory) == 0:
            return None


        mamba_out = torch.stack(list(self.memory), dim=2).contiguous()
        for mamba_layer in self.mamba_layers:
            mamba_out = mamba_layer(mamba_out) 

        final_mm = mamba_out[:,:, -1]

        self.memory.clear()

        return final_mm

class MemoryBank_s(nn.Module):
    def __init__(self, dim, depth=2, bank_size=10, num_heads=8):
        super().__init__()
        self.dim = dim
        self.bank_size = bank_size
        self.memory = deque(maxlen=bank_size)

        self.mamba_layers = nn.ModuleList()
        # Mamba and Transformer encoder
        for i in range(depth):
            self.mamba_layers.append(BiWindowMambaLayer(dim, i+1))
            self.mamba_layers.append(BiPixelMambaLayer(dim, i+1))


        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Cross Attention
        self.cross_attention = CrossAttention(dim=dim, num_heads=num_heads)
        
        # Final projection
        self.proj = nn.Linear(dim * 2, dim)

        self.fusion_norm = nn.LayerNorm(dim)



    def add(self, x):
        self.memory.append(x)
        
    @autocast(enabled=False)
    def forward(self):
        if len(self.memory) == 0:
            return None

        # Stack成 [B, T, C, H, W]
        mem = torch.stack(list(self.memory), dim=1)
        B, T, C, H, W = mem.shape

        # 先flatten + mean处理给Transformer
        mem_flat = mem.flatten(3).mean(-1)  # [B, T, C]
        mem_flat = mem_flat.to(dtype=torch.float32)  # ✅ 保证类型一致

        # Mamba部分保留空间结构
        mamba_out = mem.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        for mamba_layer in self.mamba_layers:
            mamba_out = mamba_layer(mamba_out)  # [B, C, T, H, W]
        mamba_out = mamba_out.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]

        # Transformer处理flatten后的特征
        transformer_out = self.transformer(mem_flat)  # [B, T, C]

        # attention的key/value也需要flatten+mean
        att_mamba_out = mamba_out.flatten(3).mean(-1)  # [B, T, C]

        # Cross Attention: 双向
        feature_TM = self.cross_attention(transformer_out, att_mamba_out, att_mamba_out)  # T->M
        feature_MT = self.cross_attention(att_mamba_out, transformer_out, transformer_out)  # M->T


        # 拼接+融合
        final_feature = torch.cat([feature_TM, feature_MT], dim=-1)  # [B, T, 2C]
        final_feature = self.proj(final_feature)  # [B, T, C]

        # 进行权重归一化，并为时间尺度融合做准备
        final_feature = self.fusion_norm(final_feature) # [B, T, C]
        attn_weights = torch.softmax(final_feature, dim=1)  # [B, T, C]
        attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # [B, T, C, 1, 1]

        final_mm = (mamba_out * attn_weights).sum(dim=1)  # [B, C, H, W]

        self.memory.clear()

        return final_mm