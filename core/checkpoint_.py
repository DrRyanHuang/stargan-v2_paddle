import os
import paddle
import paddle.fluid as fluid
#from fluid.dygraph import 

class CheckpointIO(object):
    
    def __init__(self, fname_template, **kwargs):
        
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs
    
    def register(self, **kwargs):
        # 将旧参数传入以更新 `self.module_dict`
        self.module_dict.update(kwargs)
    
    
    def save(self, step):
        
        # 要在 `fluid.dygraph.guard()` 的环境下运行
        fname = self.fname_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            outdict[name] = module.state_dict()
        fluid.save_dygraph(outdict, fname)
        
        
    def load(self, step):
        
        # 要在 `fluid.dygraph.guard()` 的环境下运行
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname+" does not exist!"
        print('Loading checkpoint from %s...' % fname)
        
        module_param_dict, _ = fluid.load_dygraph(fname)
        for name, module in self.module_dict.items():
            module.set_dict(module_param_dict[name])