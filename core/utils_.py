# -*- coding: utf-8 -*-
'''
StarGAN_v2_paddle.core.util

@author: RyanHuang
@github: DrRyanHuang

@updateTime: 2020.8.15
@notice: GPL v3
'''
import os
import json
import math
#
import cv2
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Conv2D
from paddle.fluid.dygraph import to_variable

# =============================================================================
# 与 Pytorch版本 `core.util.save_json` 对应
# =============================================================================
def save_json(json_dict, filename):
    '''
    @Brife:
        将字典保存为 `json` 文件
    @Param:
        json_dict  : 数据字典
        filename   : `json` 文件路径 + 名字
    '''
    with open(filename, 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=False)


# =============================================================================
# 这是 `win10 Python` 上的 `bug` os.path.join 出问题了
# =============================================================================
def ospj(*args):
    
    args = list(args)
    for i, arg in enumerate(args):
        if '{:' in arg:
            args[i] = arg.replace('{:', '___')
    return os.path.join(*args).replace('___', '{:')


# =============================================================================
# 
# =============================================================================
def print_network(network, name):
    pass



# =============================================================================
# 与 Pytorch版本 `core.util.he_init` 对应
# =============================================================================
def he_init(name, module):
    
    if isinstance(module, (Conv2D, Linear)):
        
        print('Initializing %s...' % name)
        
        param_dict = module.state_dict()

        weight_shape = param_dict['weight'].shape
        weight = fluid.layers.create_parameter(shape=weight_shape, 
                                               dtype="float32", 
                                               attr=None, 
                                               is_bias=False, 
                                               default_initializer=fluid.initializer.MSRAInitializer(uniform=False, fan_in=None))  # 激活是 `Relu` ??
        module.add_parameter('weight', weight)
        
        # ------- 设置偏置 -------
        if 'bias' in param_dict.keys():
            bias_shape = param_dict['bias'].shape
            bias = fluid.layers.create_parameter(shape=bias_shape, 
                                                 dtype="float32", 
                                                 attr=None, 
                                                 is_bias=True, 
                                                 default_initializer=fluid.initializer.ConstantInitializer(value=0))
            module.add_parameter('bias', bias)


# =============================================================================
# 与 Pytorch版本 `core.util.subdirs` 对应
# =============================================================================
def subdirs(dname):
    # 返回目录 `dname` 下的所有子目录
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


# =============================================================================
# 与 Pytorch版本 `core.util.denormalize` 对应
# =============================================================================
def denormalize(x):
    out = (x + 1) / 2
    return fluid.layers.clamp(out, min=0, max=1)

r'''
# ------------------------- 测试 save_numpy_img -------------------------
img = cv2.imread(r'C:\Users\HaoziHuang\Desktop\1.jpg').transpose(2, 0, 1)
img_array = np.concatenate([img[None, :]]*26)
filename = 'xxx.jpg'
nrow = 5
padding = 0
a = save_numpy_img(img_array, filename, nrow, padding)
import matplotlib.pyplot as plt
plt.imshow(a)
cv2.imwrite(filename, a)
'''

# =============================================================================
# 与 `torchvision.utils.save_image` 函数相对应
# =============================================================================
def save_numpy_img(img_array, filename, nrow, padding=0):
    '''
    img_array   : `NCHW` 图片
    nrow        : 一行有几个图片, 也就是列数
    padding     : 图片之间的 `padding` 数字
    '''
    if (padding >=0) and isinstance(padding, int):
        pass
    else:
        raise ValueError("别搞, `padding` 要大于等于 0 且为int")
    
    # --------- 将图片拆分开 ---------
    N, C, H, W = img_array.shape
    img_list = []
    for img in img_array:
        img_list.append(img.transpose(1, 2, 0))
        
    # ------ 图片不够, 白图来凑 ------
    ones_img = np.ones_like(img.transpose(1, 2, 0))
    for _ in range(nrow-N%nrow):
        img_list.append(ones_img)
        
    # ------ 构建 `padding array` ------
    pad_array_col = np.ones(shape=(H, padding, C), dtype=img_array.dtype)                      # 列方向的 pad
    pad_array_raw = np.ones(shape=(padding, (padding+W)*(nrow-1)+W, C), dtype=img_array.dtype) # 行方向的 pad
    
    img_row_col = []
    row_num = math.ceil(N/nrow)
    for i in range(row_num):
        img_row = []
        for j in range(nrow):

            if nrow-j != 1:  # 如果是最后一个则跳过
                if padding==0:
                    temp = img_list[i*nrow+j]
                else:
                    temp = np.concatenate([img_list[i*nrow+j], pad_array_col], axis=1)
                img_row.append(temp)
                
            else:
                img_row.append(img_list[i*nrow+j])
        img_row = np.concatenate(img_row, axis=1)
        if row_num-i != 1:   # 如果是最后一个则跳过
            if not padding==0:
                img_row = np.concatenate([img_row, pad_array_raw], axis=0)
        img_row_col.append(img_row)
    
    all_pic = np.concatenate(img_row_col, axis=0)
    cv2.imwrite(filename, all_pic)
    
    return all_pic



# =============================================================================
# 与 Pytorch版本 `core.util.save_image` 对应
# =============================================================================
def save_image(x, ncol, filename):
    x = denormalize(x)
    save_numpy_img(x.numpy(), filename, nrow=ncol, padding=0)



# =============================================================================
# 与 Pytorch版本 `core.util.translate_and_reconstruct` 对应
# =============================================================================
# @fluid.dygraph.no_grad
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.shape
    s_ref = nets.style_encoder(x_ref, y_ref)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_fake = nets.generator(x_src, s_ref, masks=masks)
    s_src = nets.style_encoder(x_src, y_src)
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_src, masks=masks)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = fluid.layers.concat(x_concat, axis=0)
    save_image(x_concat, N, filename)
    del x_concat



# @fluid.dygraph.no_grad
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename):
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    for i, y_trg in enumerate(y_trg_list):
        z_many = np.random.randn(10000, latent_dim)
        z_many = fluid.dygraph.to_variable(z_many)
        y_many = np.ones(shape=(10000,), dtype=np.int32) * y_trg[0]
        y_many = fluid.dygraph.to_variable(y_many.astype(np.int32))
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg, masks=masks)
            x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)

# =============================================================================
# 与 Pytorch版本 `core.util.translate_using_reference` 对应
# =============================================================================
# @fluid.dygraph.no_grad
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.shape
    wb = fluid.layers.ones(shape=[1, C, H, W], dtype='float32')
    x_src_with_wb = fluid.layers.concat([wb, x_src], axis=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    print(3333333333)
    s_ref = nets.style_encoder(x_ref, y_ref)
    print(s_ref.shape)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake_with_ref = fluid.layers.concat([x_ref[i:i+1], x_fake], axis=0)
        x_concat += [x_fake_with_ref]

    x_concat = fluid.layers.concat(x_concat, axis=0)
    save_image(x_concat, N+1, filename)
    del x_concat



# =============================================================================
# 与 Pytorch版本 `core.util.debug_image` 对应
# =============================================================================
# @fluid.dygraph.no_grad
def debug_image(nets, args, inputs, step):
    x_src, y_src = to_variable(inputs.x_src), to_variable(inputs.y_src)
    x_ref, y_ref = to_variable(inputs.x_ref), to_variable(inputs.y_ref)
          
    N = x_src.shape[0]

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    # latent-guided image synthesis    
    y_trg_list = [fluid.dygraph.to_variable(np.array([y]*N))
                  for y in range(min(args.num_domains, 5))]
    
    z_trg_list = np.tile(np.random.randn(args.num_outs_per_domain, 1, args.latent_dim), [1, N, 1])
    z_trg_list = fluid.dygraph.to_variable(z_trg_list)
    for psi in [0.5, 0.7, 1.0]:
        filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)















#class RandomHorizontalFlip(fluid.dygraph.Layer):
#
#    def __init__(self, p=0.5):
#        '''
#        @Brife:
#            将一个Batch的图片随机水平翻转
#        @Param:
#            p   :  水平翻转的概率
#        @Notice:
#            图片均为NCHW格式
#        '''
#        super(RandomHorizontalFlip, self).__init__(self.__class__.__name__)
#        self.p = p
#        self.dim = [2]
#        
#    def forward(self, x):
#        if np.random.random() < self.p:
#            x = fluid.layers.flip(x, self.dim, name=None)
#        return x
#    
#class RandomVerticalFlip(fluid.dygraph.Layer):
#
#    def __init__(self, p=0.5):
#        '''
#        @Brife:
#            将一个Batch的图片随机竖直翻转
#        @Param:
#            p   :  竖直翻转的概率
#        @Notice:
#            图片均为NCHW格式
#        '''
#        super(RandomVerticalFlip, self).__init__(self.__class__.__name__)
#        self.p = p
#        self.dim = [3]
#        
#    def forward(self, x):
#        if np.random.random() < self.p:
#            x = fluid.layers.flip(x, self.dim, name=None)
#        return x
#    
