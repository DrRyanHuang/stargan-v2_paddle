'''
@Brife:
    与Pytorch版本 `core.data_loader.py` 相对应
@Author:
    RyanHuang
@Github:
    DrRyanHuang
@License:
    GPL v3
@Notice:
    由于基于 `cv2` 实现, 路径不能有中文
    本代码大部分是自己写的轮子, 技术较差, 部分地方写的不优雅, 大家凑合看
    同时, 我在改Pytorch代码时, 留下了我的疑问和思考, 欢迎大家留言提issue
'''
import os
import random
from pathlib import Path
from itertools import chain

import cv2
import numpy as np
from munch import Munch
from multiprocessing import cpu_count

import paddle.fluid as fluid

try:
    import transforms_ as transforms
except ModuleNotFoundError:
    import core.transforms_ as transforms
# =============================================================================
# 与 `Pytorch` 版本中的 `core.data_loader.list_imgdir` 对应
# =============================================================================
def list_imgdir(dname):
    '''
    @Brife:
        获得某目录下的所有图片绝对路径列表
    @Param:
        dname  :  目标目录
    '''
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    # 每个元素都是 `pathlib.WindowsPath` 对象
    return fnames



# =============================================================================
# 希望能与 `torch.utils.data.Dataset` 对应
# =============================================================================
class Dataset:
    
    def __init__(self):
        pass
    
    def __add__(self, other):
        raise NotImplementedError
        
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError



# =============================================================================
# 与 `Pytorch` 版本中的 `core.data_loader.DefaultDataset` 对应
# =============================================================================
class DefaultDataset(Dataset):
    
    def __init__(self, root, transform=None):
        self.samples = list_imgdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None
        
    def __getitem__(self, index):
        fname = str(self.samples[index])
        img = cv2.imread(fname)[:, :, ::-1]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)



# =============================================================================
# 与 `torchvision.datasets.ImageFolder` 对应
# =============================================================================
class ImageFolder(Dataset):
    
    def __init__(self, root, transform=None, 
                 target_transform=None):
        
        self.root = root
        self.transform = lambda x:x if transform is None else transform
        self.target_transform = lambda x:x if target_transform is None else target_transform
        
        self.classes = self._get_class()
        self.class_to_idx = self._get_idx_to_class()
        self.img_list = self._get_img()
        self.targets = self._get_target()
        
        self.index = -1 # 为了可以 __next__
        
    # ---------------- 自定义 func ----------------
    def _get_class(self):
        return os.listdir(self.root)
    
    def _get_idx_to_class(self):
        cls2idx = {k:v for k, v in zip(range(len(self.classes)), self.classes)}
        return cls2idx
    
    def _get_target(self):
        return [item[1] for item in self.img_list]
    
    def _get_img(self):
        # 这里需要手动除去非图片的文件
        
        # 放置数据的列表
        img_data = []
        for idx, class_  in self.class_to_idx.items():
        
            # 当前目录
            current_dir = os.path.join(self.root, class_)
            # 当前目录所有文件
            current_dir_imgs = os.listdir(current_dir)
    
            for img_file in current_dir_imgs:
                
                # 当前照片绝对路径
                img_path = os.path.join(current_dir, img_file)
                # 添加到数据列表
                img_data.append((img_path, idx))
    
        return img_data
        
    # ---------------- magic func ----------------
    def __getitem__(self, key):
        
        # 注意 `img_path` 这里不可以有中文
        img_path, label = self.img_list[key]
        img_array = cv2.imread(img_path)[:, :, ::-1]
        
        img_array = self.transform(img_array)
        label = self.target_transform(label)
        
        return img_array, label
    
    def __len__(self):
        return len(self.img_list)
    
    def __next__(self):
        
        self.index += 1
        if self.index > len(self)-1:
            raise StopIteration()
        return self[self.index]    
    
    def __iter__(self):
        return self
    
    def __call__(self):
        # 在 `paddle` 的 `fluid.io.xmap_readers` 中需要 `callable`
        return self.__next__()
    
    # ---------------- 自定义外部可调用 func ----------------
    def _shuffle(self):
        # 给该元素列表 `shuffle`
        np.random.shuffle(self.img_list)
        self.targets = self._get_target()



# =============================================================================
# 与 `Pytorch` 版本中的 `core.data_loader.ReferenceDataset` 对应
# =============================================================================
class ReferenceDataset(Dataset):
    
    def __init__(self, root, transform=None):
        self.root = root
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform
        
        self.index = -1 # 为了可以 __next__
        
    def _make_dataset(self, root, shuffle=False):
        # 获取 "域" 列表
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = list_imgdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames)) # 这不就是 `shuffle` 一下?
            labels += [idx] * len(cls_fnames)
        if shuffle:
            order_idx = np.random.permutation(len(labels))
            fnames = fnames[order_idx]
            fnames2 = fnames2[order_idx]
            labels = labels[order_idx]
            
        return list(zip(fnames, fnames2)), labels # 每个域里的图片互相随机组队

    def __getitem__(self, index):
        fname, fname2 = self.samples[index] # 读取随机组队的图片
        label = self.targets[index]

        img = cv2.imread(str(fname))[:, :, ::-1]
        img2 = cv2.imread(str(fname2))[:, :, ::-1]
        
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)
    
    def __next__(self):
        
        self.index += 1
        if self.index > len(self)-1:
            raise StopIteration()
        return self[self.index]
    
    def __iter__(self):
        return self
    
    def __call__(self):
        # 在 `paddle` 的 `fluid.io.xmap_readers` 中需要 `callable`
        return self.__next__()
        
    # ---------------- 自定义外部可调用 func ----------------
    def _shuffle(self):
        self.samples, self.targets = self._make_dataset(self.root, shuffle=True)



# =============================================================================
# 与 `Pytorch` 中 `torch.utils.data.sampler.Sampler` 对应
# =============================================================================
class Sampler(object):

    # 一个 迭代器 基类
    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
        


# =============================================================================
# 与 `Pytorch` 中的 `torch.utils.data.sampler.WeightedRandomSampler` 对应
# =============================================================================
class WeightedRandomSampler(Sampler):
    
    def __init__(self, weights, num_samples=None):
        # 目前仅支持一维数组 `weights`
        
        self.weights = np.array(weights)
        self.num_samples = len(weights) if num_samples is None else num_samples
        
        self.weights_real = self.weights / self.weights.sum()
        self.samples = np.arange(len(weights), dtype=np.int)
        
        self.sample_array = np.random.choice(self.samples, 
                                             p=self.weights_real, 
                                             size=(self.num_samples,))
        
        self.idx = 0 # 为了能 __next__
        
    def __next__(self):
        
        self.idx += 1
        if self.idx > self.num_samples:
            raise StopIteration
        return self.sample_array[self.idx-1]
    
    def __iter__(self):
        return self
        
    def __len__(self):
        return self.num_samples
    
    def __call__(self):
        # 该方法在 `Pytorch` 中可能未定义, 此处仅为了方便调用
        return self.sample_array



# =============================================================================
# 与 `Pytorch` 版本中的 `core.data_loader._make_balanced_sampler` 对应
# =============================================================================
def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))



# =============================================================================
# 与 `torch.utils.data.DataLoader` 相对应
# =============================================================================
class DataLoader():
    
    def __init__(self, 
                 dataset, 
                 transforms=None, # 感觉在 `fluid.io.xmap_readers` 中可以加速, 故而将 `transform` 提出来
                 sampler=None,
                 batch_size=16,
                 shuffle=True,
                 num_workers=1, # 这里不能改成 0
                 buffer_size=None,
                 drop_last=False):
        
        '''
        @Param
            transforms : `callable`
            dataset    : `callable` 同时返回一个 `generator`
        '''
        
        self.dataset = dataset
        self.transforms = transforms
        
        # `sampler` 的优先级高于 `shuffle`
        self.shuffle = shuffle if sampler is None else False 
        self.sampler = sampler
        
        # 以下交给 paddle 的参数即可
        self.batch_size = batch_size     
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.buffer_size = cpu_count() if buffer_size is None else buffer_size  
        
        
        # 进行 `shuffle` 操作
        if self.shuffle:
            self.dataset._shuffle()
    
        self.xmap_reader = fluid.io.xmap_readers(self._mapper, 
                                                 self._reader, 
                                                 self.num_workers, 
                                                 self.buffer_size)
        self.batch_reader = fluid.io.batch(self.xmap_reader, 
                                           batch_size=self.batch_size, 
                                           drop_last=self.drop_last)
        
    # ---------------- magic func ----------------
    def __call__(self):
        # 意思是 call `DataLoader` 对象后会返回一个生成器
        return self.batch_reader()
    
    # ---------------- 自定义 func (不应外部调用) ----------------
    def _reader(self):
        if self.sampler is None:
            return self.dataset
        def sampler_reader():
            for idx in self.sampler:
                yield self.dataset[idx]
        return sampler_reader()    # 注意 `call` `sampler_reader` 才返回生成器
    
    def _mapper(self, x):
        # 该类只能处理数据对形如 (X, y) 的 tuple
        # 若其他形状 如(X1, X2, y) 的 tuple, 设置参数transforms=None, 类外自行处理即可
        if self.transforms is not None:
            temp = self.transforms(x[0])
            return temp, x[1]
        else:
            return x



# ----------------------- `DataLoader` 测试代码 --------------------------------
# img_size = 256
# prob = 0.5
# 
# crop = transforms.RandomResizedCrop(
#         img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
# rand_crop = transforms.Lambda(
#         lambda x:crop(x) if random.random() < prob else x)  
#       
# transform = transforms.Sequential(
#     rand_crop, # `HWC`
#     transforms.Resize(img_size, img_size), # `HWC`
#     transforms.HWC2CHW(), # `CHW`
#     transforms.RandomHorizontalFlip(), # `CHW`
#     transforms.CHW2HWC(), # `HWC`
#     transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                          std=[0.5, 0.5, 0.5]), # `HWC`
#     transforms.HWC2CHW(), # `CHW`
#     transforms.ConvertDtype(np.float32)
# )
# 
# 
# ds = ImageFolder(r'D:\git\starganv2\data\2')
# sampler = _make_balanced_sampler(ds.targets)
# x1 = DataLoader(ds, transform, sampler)
# x2 = DataLoader(ds, transform, None)
# 
# for i in x1():
#     print(i)
#     break
# -----------------------------------------------------------------------------



# =============================================================================
# 与 `Pytorch` 版本中的 `core.data_loader.get_train_loader` 对应
# =============================================================================
def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)
    
    crop = transforms.RandomResizedCrop(
            img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
            lambda x:crop(x) if random.random() < prob else x)
    
    transform = transforms.Sequential(
         rand_crop,                             # `HWC`
         transforms.Resize(img_size, img_size), # `HWC`
         transforms.HWC2CHW(),                  # `CHW`
         transforms.RandomHorizontalFlip(),     # `CHW`
         transforms.CHW2HWC(),                  # `HWC`
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5]), # `HWC`
         transforms.HWC2CHW(),                  # `CHW`
         transforms.ConvertDtype(np.float32)
     )
    
    if which == 'source':
        dataset = ImageFolder(root, None)
        # `if` 和 `elif` 不传入 `transform` 的原因是此处不可以多线程
        # 我们用 `paddle.fluid.io.xmap_readers` 来多线程读取
        sampler = _make_balanced_sampler(dataset.targets)
        return DataLoader(dataset=dataset,
                          transforms=transform,
                          batch_size=batch_size,
                          sampler=sampler,
                          num_workers=num_workers,
                          drop_last=True)
    elif which == 'reference':
        # 你可能会问, 为啥不一块返回, 这次不一样了
        # 可以看看 call `ReferenceDataset`对象返回的 tuple, 是三个
        # 而我在 `DataLoader._mapper` 中只会返回两个元素
        dataset = ReferenceDataset(root, transform)
        sampler = _make_balanced_sampler(dataset.targets)
        return DataLoader(dataset=dataset,
                          transforms=None,
                          batch_size=batch_size,
                          sampler=sampler,
                          num_workers=num_workers,
                          drop_last=True)
    else:
        raise NotImplementedError
    



                 
# =============================================================================
# 与 `Pytorch` 版本中的 `core.data_loader.get_eval_loader` 对应
# =============================================================================
def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    
    transform = transforms.Sequential(
         transforms.Resize(img_size, img_size), # `HWC`
         transforms.Resize(height, width),      # `HWC`
         transforms.Normalize(mean=mean,
                              std=std),         # `HWC`
         transforms.HWC2CHW(),                  # `CHW`
         transforms.ConvertDtype(np.float32)    # `CHW`
     )

    dataset = DefaultDataset(root, transform=None)
    return DataLoader(dataset=dataset,
                      transforms=transform,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      drop_last=drop_last)
    
    
    
# =============================================================================
# 与 `Pytorch` 版本中的 `core.data_loader.get_test_loader` 对应
# =============================================================================
def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):

    print('Preparing DataLoader for the generation phase...')
        
    transform = transforms.Sequential(
         transforms.Resize(img_size, img_size),      # `HWC`
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5]),  # `HWC`
         transforms.HWC2CHW(),                       # `CHW`
         transforms.ConvertDtype(np.float32)         # `CHW`
    )

    dataset = ImageFolder(root, None)
    return DataLoader(dataset=dataset,
                      transforms=transform,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)
    


# =============================================================================
# 与 `Pytorch` 版本中的 `core.data_loader.InputFetcher` 对应
# =============================================================================
class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader # 这玩意儿是个生成器或者迭代器应该都行, 我这里是生成器
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device_bool = fluid.is_compiled_with_cuda() # 目前版本的`paddle`用不着
        self.mode = mode
    
    # ---------------------- 以下三个为改编原来的 method ----------------------
    def _fetch_inputs(self):
        
        data_group= next(self.loader)
        x = np.array([data[0] for data in data_group])  # 四维张量
        y = np.array([data[1] for data in data_group])  # 一维张量
        return x, y

    def _fetch_refs(self):

        data_group = next(self.loader_ref)
        x  = np.array([data[0] for data in data_group])  # 四维张量
        x2 = np.array([data[1] for data in data_group])  # 四维张量
        y  = np.array([data[2] for data in data_group])  # 一维张量
        
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = np.random.randn(x.shape[0], self.latent_dim).astype(np.float32)  # 切记! `float32`
            z_trg2 = np.random.randn(x.shape[0], self.latent_dim).astype(np.float32)
            inputs = Munch(x_src=x, 
                           y_src=y, 
                           x_ref=x_ref, 
                           x_ref2=x_ref2,
                           y_ref=y_ref,
                           z_trg=z_trg, 
                           z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, 
                           y_src=y,
                           x_ref=x_ref, 
                           y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError("mode {} 没实现".format(self.mode))

        return Munch({k : v for k, v in inputs.items()}) # 什么鬼这是?瞎折腾?





# ----------------------- `InputFetcher` 测试代码 ------------------------------
# img_size = 256
# prob = 0.5
# 
# crop = transforms.RandomResizedCrop(
#         img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
# rand_crop = transforms.Lambda(
#         lambda x:crop(x) if random.random() < prob else x)  
#       
# transform = transforms.Sequential(
#     rand_crop, # `HWC`
#     transforms.Resize(img_size, img_size), # `HWC`
#     transforms.HWC2CHW(), # `CHW`
#     transforms.RandomHorizontalFlip(), # `CHW`
#     transforms.CHW2HWC(), # `HWC`
#     transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                          std=[0.5, 0.5, 0.5]), # `HWC`
#     transforms.HWC2CHW(), # `CHW`
#     transforms.ConvertDtype(np.float32)
# )
# 
# 
# ds = ImageFolder(r'D:\git\starganv2\data\celeba_hq\train')
# sampler = _make_balanced_sampler(ds.targets)
#
# x1 = DataLoader(ds, transform, sampler)
# x2 = DataLoader(ds, transform, None)
#
# x3 = get_train_loader(r'D:\git\starganv2\data\celeba_hq\train', 'source')
# x4 = get_train_loader(r'D:\git\starganv2\data\celeba_hq\train', 'reference')
#
# ss = InputFetcher(x3(), x4(), mode='train')
#
# a = next(ss)
# -----------------------------------------------------------------------------




    

    
