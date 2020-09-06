'''
@Brife:
    与 `Pytorch` 中的部分 `transforms` 相对应
@Author:
    RyanHuang
@Github:
    DrRyanHuang
@License:
    GPL v3
@Notice:
    时间紧迫, 仅仅提供 `CHW2HWC` 和 `HWC2CHW` 用于通道转换
    有的类支持 `CHW`, 有的支持 `HWC`
    暂时就不写 `data_format` 参数了
    同时在 `Sequencial` 里, 多次调用预处理函数, 浪费时间, 需要优化
'''


import cv2
import numpy as np

__all__ = []


# =============================================================================
# 类之基类
# =============================================================================
class TransBase():
    
    def __init__():
        pass
    
    def deal_dim3(self):
        raise NotImplementedError("就甭调用基类了吧哈哈")
    
    def deal_dim4(self):
        raise NotImplementedError("就甭调用基类了吧哈哈")
    
    def deal_dim4_from_dim3(self, x):
        # 将 dim4 的数据拆分开, 分别用 dim3 的方法来处理
        processed = []
        for x_i in x:
            x_i = self.deal_dim3(x_i)
            processed.append(x_i)
        return np.array(processed)
    
    def pre_process(self, x):
        # 用于对初始数据进行初始化的方法
        return x
    
    def last_process(self, x):
        # 用于对数据做最后的处理
        return x
        
    def __call__(self, x):
        
        if not isinstance(x, np.ndarray):
            raise ValueError('兄弟, 只支持 `np.ndarray` 输出参数是{}'.format(type(x)))
        if len(x.shape) == 3:
            x = self.pre_process(x)
            x = self.deal_dim3(x)
            x = self.last_process(x)
            return x
        elif len(x.shape) == 4:
            x = self.pre_process(x)
            x = self.deal_dim4(x)
            x = self.last_process(x)
            return x
        else:
            raise ValueError("图片数据必须是 `CHW` 或 `HWC`, 当前图片shape为{}".format(x.shape))



# =============================================================================
# 通道数转换
# =============================================================================
class CHW2HWC(TransBase):
    
    def __init__(self):
        pass
    
    def deal_dim3(self, x):
        return x.transpose(1, 2, 0)
    
    def deal_dim4(self, x):
        return x.transpose(0, 2, 3, 1)


class HWC2CHW(TransBase):
    
    def __init__(self):
        pass
    
    def deal_dim3(self, x):
        return x.transpose(2, 0, 1)
    
    def deal_dim4(self, x):
        return x.transpose(0, 3, 1, 2)



# =============================================================================
# 翻转操作
# =============================================================================
class RandomFlip(TransBase):
    
    def __init__(self, px=0.5, py=0.5, data_format='CHW'):
        '''
        @Notice:
            图片数据必须是 `CHW` 或 `NCHW`
        '''
        self.px = px
        self.py = py
    
    def deal_dim3(self, x):
        if np.random.random() < self.px:
            x = np.flip(x, 2)
        if np.random.random() < self.py:
            x = np.flip(x, 1)
        return x
        
    def deal_dim4(self, x):
        return self.deal_dim4_from_dim3(x)

   
class RandomHorizontalFlip(RandomFlip):
    '''
    @Notice:
        图片数据必须是 `CHW` 或 `NCHW`
    '''
    def __init__(self, px=0.5):
        super(RandomHorizontalFlip, self).__init__(px=px, py=0)

    
class RandomVerticalFlip(RandomFlip):
    '''
    @Notice:
        图片数据必须是 `CHW` 或 `NCHW`
    '''
    def __init__(self, py=0.5):
        super(RandomVerticalFlip, self).__init__(px=0, py=py)



# =============================================================================
# 图片缩放
# =============================================================================
class Resize(TransBase):
    '''
    @Notice:
        图片数据必须是 `HWC` 或 `NHWC`
        参数 `shape` 是 `ndarray.shape` 而不是图片的 shape(宽高)
    @Param:
        0 : cv2.INTER_NEAREST
        1 : cv2.INTER_LINEAR
        2 : cv2.INTER_CUBIC
        3 : cv2.INTER_AREA
        4 : cv2.INTER_LANCZOS4
    '''
    def __init__(self, *shape, interpolation=1):
        
        # 传入 shape=(224, 224)时, 为 `Resize(224, 224)` 而不是 `Resize(224, 224)`
        self.shape = shape
        self.ONE = (len(shape)==1)
        self.interp = interpolation
    
    def deal_dim3(self, x):
        if self.ONE:
            
            min_idx  = int(x.shape[0] > x.shape[1])
            max_idx  = int(x.shape[0] < x.shape[1])

            new_shape = list(x.shape[:-1])
            new_shape[max_idx] = int(self.shape[0] / new_shape[min_idx] * new_shape[max_idx])
            new_shape[min_idx] = int(self.shape[0])
            
            x = cv2.resize(x, tuple(new_shape), interpolation=self.interp)
        else:
            x = cv2.resize(x, (self.shape[1], self.shape[0]), interpolation=self.interp)
        return x
    
    def deal_dim4(self, x):
        return self.deal_dim4_from_dim3(x)



# =============================================================================
# 随机裁剪缩放
# =============================================================================
class RandomResizedCrop(Resize):
    '''
    @Notice:
        图片格式必须是 `HWC`
    '''
    def __init__(self, *img_size, scale=None, ratio=None, interpolation=1):
        
        super(RandomResizedCrop, self).__init__(*img_size, interpolation=interpolation)
        
        scale = [0.8, 1.0] if scale is None else scale
        ratio = [0.9, 1.1] if ratio is None else ratio
        
        self.scale_num = np.random.uniform(*scale)
        self.ratio_num = np.random.uniform(*ratio)

    def random_crop_from_scale(self, img, scale_num=0.8):
        '''
        @Brife:
            在原图上根据 `scale_num` 随机裁剪
        @Notice:
            图片格式必须是 `HWC`
            为了可拓展性, 将此方法单独列出
        '''
        img_shape = np.array(img.shape[:-1])
        # 获得左上角坐标的范围
        left_up_max = (img_shape * (1-scale_num)).astype(np.int)
        # 获得左上角坐标
        y_min = np.random.randint(left_up_max[0])
        x_min = np.random.randint(left_up_max[1])
        # 获得右下角坐标
        y_max = y_min + int(img_shape[0]*scale_num)
        x_max = x_min + int(img_shape[1]*scale_num)
        
        return img[y_min:y_max, x_min:x_max]
    
    def random_crop_from_ratio(self, img, ratio=1):
        '''
        @Brife:
            在图片熵根据宽高比随机裁剪
        @Param:
            ratio  :   宽高比
        @Notice:
            图片格式必须是 `HWC`
            为了可拓展性, 将此方法单独列出
        '''
        
        y, x = img.shape[:-1]
        if y/x >= ratio:
            y_new = int(ratio * x)
            x_new = x
            y_start = np.random.randint(y-y_new)
            return img[y_start:y_start+y_new, :x_new]
        else:
            x_new = int(y / ratio)
            y_new = y
            x_start = np.random.randint(x-x_new)
            return img[:y_new, x_start:x_start+x_new]
            
    def deal_dim3(self, x):
        
        x = self.random_crop_from_scale(x, scale_num=self.scale_num)
        x = self.random_crop_from_ratio(x, ratio=self.ratio_num)
        x = super(RandomResizedCrop, self).deal_dim3(x)
        return x
    
    def deal_dim4(self, x):
        
        return self.deal_dim4_from_dim3(x)
    
    

# =============================================================================
# 标准化处理
# =============================================================================    
class Normalize(TransBase):
    '''
    @Notice:
        图片数据必须是 `HWC` 或 `NHWC`
        且图片为 `RGB` 图片
        若数据为 `np.uint8` 则会预处理为 `np.float32` 范围 [0, 1]
    '''
    def __init__(self, mean=0.5, std=0.5):
        '''
        @Param:
            mean  :  shape为(3,)的ndarray 或者 float
            std   :  shape为(3,)的ndarray 或者 float
        '''
        self.mean = np.array(mean)
        if isinstance(mean, int):
            self.mean = np.array([mean]*3)
        
        self.std = np.array(std)
        if isinstance(std, int):
            self.mean = np.array([std]*3)
    
    def pre_process(self, x):
        # 若为 `uint8` 则需要转化为 `float32`        
        if x.dtype == np.uint8:
            x = (x / 255).astype(np.float32)
        return x
        
    def deal_dim3(self, x):
          x = (x - self.mean) / self.std
          return x
    
    def deal_dim4(self, x):
        return self.deal_dim4_from_dim3(x)



# =============================================================================
# `TransBase` 序列化
# =============================================================================
class Sequential(TransBase):
    
    def __init__(self, *trans_list):
        
        self.trans_list = list(trans_list)
        
        for trans_layer in self.trans_list:
            if not isinstance(trans_layer, TransBase):
                raise ValueError("`Sequential`的构造函数中必须传入继承自`TransBase`的实例")
    
    def deal_dim3(self, x):
        
        for trans_layer in self.trans_list:
            x = trans_layer(x)
        return x
            
    def deal_dim4(self, x):
        
        for trans_layer in self.trans_list:
            x = trans_layer(x)
        return x

    
# =============================================================================
# 自定义匿名处理
# =============================================================================
class Lambda(TransBase):
    '''
    @Notice:
        由于是自定义函数处理故而, 用户需要自行处理 `CHW` 还是 `HWC` 的问题
    '''
    def __init__(self, func):
        
        if not callable(func):
            raise ValueError("`func` 变量应为函数或者说定义 `__call__` 方法")
        self.func = func
    
    def deal_dim3(self, x):
        return self.func(x)

    def deal_dim4(self, x):
        return self.deal_dim4_from_dim3(x)



# =============================================================================
# 进行数据 `dtype` 处理
# =============================================================================
class ConvertDtype(TransBase):
    '''
    类型转换, 为了优雅, 也写到 `pipline` 里
    '''
    def __init__(self, dtype):
        if dtype not in [np.float32, 'float32']:
            raise TypeError('Paddlepaddle最常用的是 float32, 别的就先不写了')
        self.dtype = dtype
        
    def deal_dim3(self, x):
        return x.astype(self.dtype)

    def deal_dim4(self, x):
        return x.astype(self.dtype)
        
    
#import matplotlib.pyplot as plt
#pic_p = 'D:\\git\\starganv2\\data\\celeba_hq\\train\\female\\000155.jpg'
#pic = cv2.imread(pic_p)[:,:,::-1]
##plt.imshow(pic)
#n = RandomResizedCrop(500, 323, scale=[0.5, 0.6], ratio=[1, 1.15])
#ppic_ = n(pic)
#plt.imshow(ppic_)
#
#plt.figure()
#pic = pic[None, :]
#ppic = n(pic)
#
#ppic_ = n(pic)
#plt.imshow(ppic_[0])



    
