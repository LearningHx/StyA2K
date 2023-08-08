import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler
from torch.nn import functional as f 
import random


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment layers; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            lr_l = 0.3 ** max(0, epoch + opt.epoch_count - opt.n_epochs)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class A2K(nn.Module):

    def __init__(self, in_planes,max_sample=256 * 256):
        super(A2K, self).__init__()
        self.q = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.k = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.v = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.q2 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.k2= nn.Conv2d(in_planes, in_planes, (1, 1))
        self.v2 = nn.Conv2d(in_planes, in_planes, (1, 1))

        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample
        self.fusion1 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.fusion2 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sim_alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.sim_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        
    def forward(self, content, style, layer,seed=None):
        layer = int(layer)
        if layer == 3:
            region = 8
            stride = 8
        elif layer==4:
            region = 8
            stride = 8
        elif layer==5:
            region =4
            stride = 4

        b,c,h,w = content.shape

        head =8
        norm_content = mean_variance_norm(content)
        norm_style = mean_variance_norm(style)
        
        Q1 = self.q(norm_content)
        K1 = self.k(norm_style)
        V1 = self.v(style)
        Q2 = self.q2(norm_content)
        K2 = self.k2(norm_style)
        V2 = self.v2(style)
        
        DA_q = block(Q1,region,stride)
        b,c,n,r = DA_q.shape
        DA_q = DA_q.view(b,head,c//head,n,r)
        DA_k = block(K1,region,stride).view(b,head,c//head,n,r)
        DA_v = block(V1,region,stride).view(b,head,c//head,n,r)

        PA_q = block(Q2,region,stride).view(b,head,c//head,n,r)
        PA_k = block(K2,region,stride).view(b,head,c//head,n,r)
        PA_v = block(V2,region,stride).view(b,head,c//head,n,r)
        
        DA_k_centers = torch.mean(DA_k,dim=-1)
        DA_v_centers = torch.mean(DA_v,dim=-1)
        dis =  torch.einsum("bhcx,bhcxy->bhxy",DA_k_centers,DA_k)
        sim = torch.sigmoid(self.sim_beta+self.sim_alpha*dis)
        DA_k_agg = (torch.einsum("bhxy,bhcxy->bhcx",sim,DA_k) + DA_k_centers)/(r+1)
        DA_v_agg = (torch.einsum("bhxy,bhcxy->bhcx",sim,DA_v) + DA_v_centers)/(r+1)

        logits = torch.einsum("bhcxy,bhcz->bhyxz",DA_q,DA_k_agg)
        scores =  self.sm(logits)                                     #global
        DA = torch.einsum("bhyxz,bhcz->bhcxy",scores,DA_v_agg)
        DA_unblock =unblock(DA.contiguous().view(b,c,n,r),region,stride,h)

        PA1_logits = torch.einsum("bhcxy,bhczy->bhxz",PA_q,PA_k)
        index = torch.argmax(PA1_logits,dim = -1).view(b,head,1,n,1).expand_as(PA_k)  
        PA_k_reshuffle = torch.gather(PA_k,-2,index)
        PA_v_reshuffle = torch.gather(PA_v,-2,index)
        logits2 = torch.einsum("bhcxy,bhcxz->bhxyz",PA_q,PA_k_reshuffle)
        scores2 = self.sm(logits2)                                     #local
        PA = torch.einsum("bhxyz,bhcxz->bhcxy",scores2,PA_v_reshuffle)
        PA_unblock =unblock(PA.contiguous().view(b,c,n,r),region,stride,h)

        O_DA = self.fusion1(DA_unblock)
        O_PA = self.fusion2(PA_unblock)
        O = (O_DA + O_PA)
        Z = O + content

        return Z

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):

        """ avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3 """
        
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

def block(x,patch_size = 4,stride=4):
    b,c,h,w = x.shape
    r = int(patch_size**2)
    y = f.unfold(x, kernel_size=patch_size, stride=stride)
    y = y.permute(0, 2, 1)
    y = y.view(b, -1, c,r).permute(0,2,1,3)
    #y = y.reshape(b,-1,r,c)      #b*c*l*r
    return y

def unblock(x,patch_size,stride,h):
    b,c,l,r = x.shape
    x = x.permute(0,2,1,3)
    x = x.contiguous().view(b,l,-1).permute(0,2,1)
    y = f.fold(x,h,kernel_size=patch_size, stride=stride)
    # norm_map = f.fold(f.unfold(torch.ones(x.shape)))
    return y


class Transform(nn.Module):

    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.attn_adain_4_1 = A2K(in_planes=in_planes)
        self.attn_adain_5_1 = A2K(in_planes=in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1, seed=None):
        
        attn_feats5_1 = self.attn_adain_5_1(content5_1, style5_1, layer='5', seed=seed)
        attn_feats4_1 =  self.attn_adain_4_1(content4_1, style4_1, layer='4', seed=seed)
        out = self.merge_conv(self.merge_conv_pad( attn_feats4_1+self.upsample5_1(attn_feats5_1)))
        
        return out


class Decoder(nn.Module):

    def __init__(self, skip_connection_3=False):
        super(Decoder, self).__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256 + 256 if skip_connection_3 else 256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, cs, c_A2K_feat_3=None):
        cs = self.decoder_layer_1(cs)
        if c_A2K_feat_3 is None:
            cs = self.decoder_layer_2(cs)
        else:
            cs = self.decoder_layer_2(torch.cat((cs, c_A2K_feat_3), dim=1))
        return cs


