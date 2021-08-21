import torch
import torch.nn as nn
import math
import numpy as np


OPS = {
  # candidate opoeration Dictionary
  'transformer_focal'     : lambda dim: Transformer_focal(dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.),
  'transformer_performer' : lambda dim: Transformer_performer(dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kernel_ratio=0.5),
  'transformer_cswin'     : lambda dim: Transformer_cswin(dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kernel_ratio=0.5),
  'transformer'        : lambda dim: Transformer(dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kernel_ratio=0.5),
  'conv_mobile'           : lambda dim: Conv_mobile(dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                             drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, class_token=False, group=64, tokens_type='transformer'),
  'conv_bottleneck'       : lambda dim: Conv_bottleneck(dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                             drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, class_token=False, group=64, tokens_type='transformer'),
  'conv_xx'               : lambda dim: Conv_xx(dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                             drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, class_token=False, group=64, tokens_type='transformer'),                             
}

# all the operations share the same para

#########################################################
# Transformer Operations
#########################################################

class Transformer_focal(nn.Module):
    def __init__(self):
        super(Transformer_focal).__init__()

    
    def forward(self,x):

        return x


class Transformer_performer(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kernel_ratio=0.5):
        super(Transformer_performer,self).__init__()
        self.emb = dim * num_heads # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(proj_drop)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = num_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.epsilon = 1e-8  # for stable in division
        self.drop_path = nn.Identity()

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection
        return y

    def forward(self, x):
        x = self.attn(x)
        return x

class Transformer_cswin(nn.Module):
    def __init__(self):
        super(Transformer_cswin).__init__()

    
    def forward(self,x):

        return x


class Transformer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Transformer,self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#########################################################
# CNN Operations
#########################################################

class Conv_mobile(nn.Module):
    def __init__(self):
        super(Conv_mobile).__init__()

    
    def forward(self,x):

        return x



class Conv_bottleneck(nn.Module):
    def __init__(self):
        super(Conv_bottleneck).__init__()

    
    def forward(self,x):

        return x



class Conv_xx(nn.Module):
    def __init__(self):
        super(Conv_xx).__init__()

    
    def forward(self,x):

        return x

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

