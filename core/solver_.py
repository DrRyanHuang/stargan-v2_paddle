# -*- coding: utf-8 -*-
'''
StarGAN_v2_paddle.core.util

@author: RyanHuang
@github: DrRyanHuang

@updateTime: 2020.8.15
@notice: GPL v3
'''
import os
from core.utils_ import ospj



import time
import datetime
from munch import Munch

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable


from core.model_ import build_model
from core.checkpoint_ import CheckpointIO
import core.utils_ as utils
from core.data_loader_ import InputFetcher

class Solver(fluid.dygraph.Layer):
    
    def __init__(self, args):
        
        super(Solver, self).__init__(self.__class__.__name__)
        self.args = args
        self.cuda_bool = fluid.is_compiled_with_cuda()
        
        self.nets, self.nets_ema = build_model(args)
        
        for name, module in self.nets.items():
            # `pytorch` 版本中此处有打印网络结构部分
            setattr(self, name, module)
        
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)
        
        
        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = fluid.optimizer.AdamOptimizer(
                    parameter_list=self.nets[net].parameters(),
                    learning_rate=args.f_lr if net == 'mapping_network' else args.lr,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    regularization=fluid.regularizer.L2Decay(regularization_coeff=args.weight_decay))
                    # weight_decay=args.weight_decay) # 不知道这个怎么解决
        
            print(1)
            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema)]
        
        for name, network in self.named_sublayers():  # 迭代那 `7`个网络
            
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                # 初始化参数
                utils.he_init(name, network)

    def _save_checkpoint(self, step):
        # 需要 `fluid.dygraph.guard()` 环境
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        # 需要 `fluid.dygraph.guard()` 环境
        for ckptio in self.ckptios:
            ckptio.load(step)
        
    def _reset_grad(self):
        for optim in self.optims.values():
            # 清除需要优化的参数的梯度
            optim.clear_gradients()
    
    def train(self, loaders):
        
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src(), loaders.ref(), args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val(), None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            # -------------------- 设置 `stop_gradient` --------------------
            x_real, y_org = to_variable(inputs.x_src), to_variable(inputs.y_src)
            x_real.stop_gradient = False
            y_org.stop_gradient = False
            
            x_ref, x_ref2, y_trg = to_variable(inputs.x_ref), to_variable(inputs.x_ref2), to_variable(inputs.y_ref)
            x_ref.stop_gradient = False
            x_ref2.stop_gradient = False
            y_trg.stop_gradient = False
            
            z_trg, z_trg2 = to_variable(inputs.z_trg), to_variable(inputs.z_trg2)
            z_trg.stop_gradient = False
            z_trg2.stop_gradient = False
        
            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None
        
            # ------------ train the discriminator ------------
            d_loss, d_losses_latent = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
            d_loss.backward()
            optims.discriminator.minimize(d_loss)
            self._reset_grad()
            
            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            d_loss.backward()
            optims.discriminator.minimize(d_loss)
            self._reset_grad()
            
            # ------------ train the generator ------------
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.minimize(g_loss)
            optims.mapping_network.minimize(g_loss)
            optims.style_encoder.minimize(g_loss)

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.minimize(g_loss)

            # -------- compute moving average of network parameters --------
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # -------- decay weight for diversity sensitive loss --------
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # ---------------- print out log info ----------------
            if (i+1) % args.print_every == 0:
                # -------------- 打印时间信息 --------------
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                
                # 准备 `loss` 信息
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)
            
            i = -1
            # ------- generate images for debugging -------
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # ------- save model checkpoints -------
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # ------- compute FID and LPIPS if necessary -------
            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='latent')
                calculate_metrics(nets_ema, args, i+1, mode='reference')
                
                

# =============================================================================
# 与 Pytorch 版本 `core.solver.compute_d_loss` 相对应
# =============================================================================
def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    
    assert (z_trg is None) != (x_ref is None)
    # with real images
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with fluid.dygraph.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg # shape : [1] ; Pytorch shape : []
    return loss, Munch(real=loss_real.numpy()[0],
                       fake=loss_fake.numpy()[0],
                       reg=loss_reg.numpy()[0])
    
    
    
# =============================================================================
# 与 Pytorch 版本 `core.solver.compute_g_loss` 相对应
# =============================================================================
def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # --------- style reconstruction loss ---------
    s_pred = nets.style_encoder(x_fake, y_trg)
    
    loss_sty = fluid.layers.abs(s_pred - s_trg)
    loss_sty = fluid.layers.mean(loss_sty)


    # --------- diversity sensitive loss ---------
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = fluid.layers.abs(x_fake - x_fake2)
    loss_ds = fluid.layers.mean(loss_ds)
    

    # --------- cycle-consistency loss ---------
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = fluid.layers.abs(x_rec - x_real)
    loss_cyc = fluid.layers.mean(loss_cyc)

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.numpy()[0],
                       sty=loss_sty.numpy()[0],
                       ds=loss_ds.numpy()[0],
                       cyc=loss_cyc.numpy()[0])



# =============================================================================
# 与 Pytorch 版本 `core.solver.moving_average` 相对应
# =============================================================================  
def moving_average(model, model_test, beta=0.999):
    
    model_test_param = model_test.state_dict()
    for (name, param), (name_test, param_test) in zip(model.named_parameters(), model_test.named_parameters()):
        
        # 进行有效性检验
        assert name==name_test, "名字不相符!! 请检查 `model_emo`"
        
        temp_param = param + (param_test - param) * beta
        model_test_param[name] = temp_param
    model_test.set_dict(model_test_param)



# =============================================================================
# 与 Pytorch 版本 `core.solver.adv_loss` 相对应
# =============================================================================       
def adv_loss(logits, target):
    assert target in [1, 0]

    # paddle 1.8
    targets = fluid.layers.full_like(logits, fill_value=target)
    # paddle 2.0
    # targets = paddle.full_like(logits, fill_value=target)
    
    # 注意此处的 `binary_cross_entropy_with_logits` 会添加 `sigmoid`
    loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=logits, label=targets)
    loss = fluid.layers.reduce_mean(loss)
    return loss
        


# =============================================================================
# 与 Pytorch 版本 `core.solver.r1_reg` 相对应
# =============================================================================         
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.shape[0]
    
    # paddle 2.0
    # d_out = paddle.sum(d_out)
    # paddle 1.8
    d_out = fluid.layers.reduce_sum(d_out)
    
    # 注意 paddle2.0 中的文档是 `paddle.grad` 可能有问题？
    # paddle1.8.4 不会有
    grad_dout = fluid.dygraph.grad(outputs=d_out, 
                                   inputs=x_in, 
            retain_graph=True, create_graph=False, only_inputs=True)[0]
    
    # paddle 2.0
    # grad_dout = paddle.pow(x, 2)
    # paddle 1.8.4
    grad_dout2 = fluid.layers.pow(grad_dout, factor=2)
    
    assert(grad_dout2.shape == x_in.shape)
    
    temp = fluid.layers.reshape(grad_dout2, shape=(batch_size, -1))
    temp = fluid.layers.reduce_sum(temp, dim=1)
    reg = 0.5 * fluid.layers.reduce_mean(temp, dim=0) # `Pytorch` 版本这里返回的是标量
    return reg
        
        
        
        
        