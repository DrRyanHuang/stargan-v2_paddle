# -*- coding: utf-8 -*-
'''
StarGAN_v2_paddle.core.model

@author: RyanHuang
@github: DrRyanHuang

@updateTime: 2020.8.15
@notice: GPL v3
'''
import math
import copy

import numpy as np
from munch import Munch 

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable, Conv2D, InstanceNorm, Linear

#from .wing import FAN
#from core.wing import FAN

class ResBlk(fluid.dygraph.Layer):

    def __init__(self, dim_in, dim_out, actv=None,
                 normalize=False, downsample=False):

        super(ResBlk, self).__init__(self.__class__.__name__)
        if actv is None:
            actv = (lambda x:fluid.layers.leaky_relu(x, alpha=0.2))
        self.actv = actv
        self.downsample = downsample
        self.learned_sc = (dim_in != dim_out)
        self.normalize = normalize
        self._build_weights(dim_in, dim_out)
    
    def _build_weights(self, dim_in, dim_out):

        self.conv1 = Conv2D(num_channels=dim_in, num_filters=dim_in,  filter_size=3, stride=1, padding=1)
        self.conv2 = Conv2D(num_channels=dim_in, num_filters=dim_out, filter_size=3, stride=1, padding=1)
        if self.normalize:
            self.norm1 = InstanceNorm(dim_in) # 没有 `momentum` 部分, 已在github提交issue
            self.norm2 = InstanceNorm(dim_in)
        if self.learned_sc:
            self.conv1x1 = Conv2D(dim_in, dim_out, 1, 1, 0, bias_attr=False)
            
    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = fluid.layers.pool2d(x, pool_size=2, pool_stride=2, pool_type='avg')
        return x
    
    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = fluid.layers.pool2d(x, pool_size=2, pool_stride=2, pool_type='avg')
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x
    
    def forward(self, inputs):
        x = self._shortcut(inputs) + self._residual(inputs)
        return x / math.sqrt(2)   # unit variance



class AdaIN(fluid.dygraph.Layer):

    def __init__(self, style_dim, num_features):
        super(AdaIN, self).__init__(self.__class__.__name__)
        self.norm = InstanceNorm(num_features)
        self.fc = Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = fluid.layers.reshape(h, shape=(h.shape[0], h.shape[1], 1, 1))
        gamma, beta = fluid.layers.split(h, num_or_sections=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta



class AdainResBlk(fluid.dygraph.Layer):

    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0, actv=None,
                    upsample=False):

        super(AdainResBlk, self).__init__(self.__class__.__name__)
        self.w_hpf = w_hpf
        self.actv = (lambda x:fluid.layers.leaky_relu(x, alpha=0.2)) if actv is None else actv
        self.upsample = upsample
        self.learned_sc = (dim_in != dim_out)
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):

        self.conv1 = Conv2D(dim_in, dim_out, 3, 1, 1)
        self.conv2 = Conv2D(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = Conv2D(dim_in, dim_out, 1, 1, 0)
    
    def _shortcut(self, x):
        if self.upsample:
            x = fluid.layers.image_resize(x, resample='NEAREST', scale=2)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = fluid.layers.image_resize(x, resample='NEAREST', scale=2)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
        


class HighPass(fluid.dygraph.Layer):
    
    def __init__(self, w_hpf):
        super(HighPass, self).__init__(self.__class__.__name__)
        self.filter_ = np.array([[-1, -1, -1],
                                 [-1, 8., -1],
                                 [-1, -1, -1]]).reshape(1, 1, 3, 3) / w_hpf
    
    def forward(self, x):

        filter_ = self.filter_.repeat(x.shape[1], axis=0)
        param = fluid.initializer.NumpyArrayInitializer(filter_)
        x = fluid.layers.conv2d(x, num_filters=1, filter_size=3, padding=1, groups=x.shape[1], param_attr=param, bias_attr=False)
        return x



class LeakyRelu(fluid.dygraph.Layer):
    def __init__(self, alpha=0.2):
        super(LeakyRelu, self).__init__(self.__class__.__name__)
        self.alpha = alpha
    def forward(self, x):
        return fluid.layers.leaky_relu(x, self.alpha)


# =============================================================================
# 通过该函数解决 `out[idx, y]` 不能直接索引的问题
# =============================================================================
def get_value_by_index(out, idx, y, who='Discriminator'):
    
    temp = []
    for i, j in zip(idx, y):
        item = out[int(i)][int(j)]
        if who == 'not_discri':
            item = fluid.layers.reshape(item, (-1, item.shape[0]))
        temp.append(item)
    temp = fluid.layers.concat(temp)
    return temp


class Generator(fluid.dygraph.Layer):
            
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super(Generator, self).__init__(self.__class__.__name__)
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = Conv2D(3, dim_in, 3, 1, 1)

        self.encode = fluid.dygraph.LayerList()
        self.decode = fluid.dygraph.LayerList()
        self.to_rgb = fluid.dygraph.Sequential(
            InstanceNorm(dim_in),
            LeakyRelu(0.2),
            Conv2D(dim_in, 3, 1, 1, 0)
        )

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True)
            )
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True)  # stack-like
            )
            dim_in = dim_out
        
        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True)
            )
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf)
            )
        
        if w_hpf > 0:
            self.hpf = HighPass(w_hpf)
    
    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                cache[x.shape[2]] = x
            x = block(x)
        
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (s.shape[2] in [32, 64, 128]):
                mask = masks[0] if x.shape[2] in [32] else masks[1]
                mask = fluid.layers.image_resize(mask, size=x.shape[2], resample='BILINEAR')
                x = x + self.hpf(mask * cache[x.shape[2]])
        return self.to_rgb(x)

class MappingNetwork(fluid.dygraph.Layer):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super(MappingNetwork, self).__init__(self.__class__.__name__)
        layers = []
        layers += [Linear(latent_dim, 512, act='relu')]
        
        for _ in range(3):
            layers += [Linear(512, 512, act='relu')]
        self.shared = fluid.dygraph.Sequential(
            *layers
        )

        self.unshared = fluid.dygraph.LayerList()
        for _ in range(num_domains):
            self.unshared.append(
                fluid.dygraph.Sequential(
                    Linear(512, 512, act='relu'),
                    Linear(512, 512, act='relu'),
                    Linear(512, 512, act='relu'),
                    Linear(512, style_dim, act=None)
                )
            )
    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = fluid.layers.stack(out, axis=1)                       # (batch, num_domains, style_dim)

        idx = to_variable(np.arange(y.shape[0], dtype=np.int))
        s = get_value_by_index(out, idx, y, who='not_discri')   # (batch, style_dim)
        return s

class StyleEncoder(fluid.dygraph.Layer):

    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super(StyleEncoder, self).__init__(self.__class__.__name__)
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [Conv2D(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [LeakyRelu(0.2)]
        blocks += [Conv2D(dim_out, dim_out, 4, 1, 0)]
        blocks += [LeakyRelu(0.2)]
        self.shared = fluid.dygraph.Sequential(*blocks)
        
        self.unshared = fluid.dygraph.LayerList()
        for _ in range(num_domains):
            self.unshared.append(Linear(dim_out, style_dim))
    
    def forward(self, x, y):
        
        h = self.shared(x)
        h = fluid.layers.reshape(h, (h.shape[0], -1))
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = fluid.layers.stack(out, axis=1)                       # (batch, num_domains, style_dim)
        idx = to_variable(np.arange(y.shape[0], dtype=np.int))
        
        s = get_value_by_index(out, idx, y, who='not_discri')   # (batch, style_dim)
        return s

class Discriminator(fluid.dygraph.Layer):

    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super(Discriminator, self).__init__(self.__class__.__name__)
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [Conv2D(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        
        blocks += [LeakyRelu(0.2)]
        blocks += [Conv2D(dim_out, dim_out, 4, 1, 0)]
        blocks += [LeakyRelu(0.2)]
        blocks += [Conv2D(dim_out, num_domains, 1, 1, 0)]
        self.main = fluid.dygraph.Sequential(*blocks)
    
    def forward(self, x, y):
        out = self.main(x)
        out = fluid.layers.reshape(out, (x.shape[0], -1))
        idx = to_variable(np.arange(y.shape[0], dtype=np.int))
        out = get_value_by_index(out, idx, y)
        return out

def build_model(args):
    generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    discriminator = Discriminator(args.img_size, args.num_domains)
    
    # --------- TypeError: can't pickle XX objects ---------
    # generator_ema = copy.deepcopy(generator)
    # mapping_network_ema = copy.deepcopy(mapping_network)
    # style_encoder_ema = copy.deepcopy(style_encoder)
    # ------------------------------------------------------
    generator_ema = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    generator_ema.set_dict(generator.state_dict().copy())
    
    mapping_network_ema = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    mapping_network_ema.set_dict(mapping_network.state_dict().copy())
    
    style_encoder_ema = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    style_encoder_ema.set_dict(style_encoder.state_dict().copy())
    
    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)
    if args.w_hpf > 0:
        fan = FAN(fname_pretrained=args.wing_path).eval()
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema