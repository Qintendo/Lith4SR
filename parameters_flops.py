# -- coding: utf-8 --
import torch
import os
from thop import profile
from model import Lith4SR

import argparse
import options

opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

# Model
print('==> Building model..')
model = Lith4SR(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_projection=opt.token_projection,token_mlp=opt.token_mlp)

dummy_input = torch.randn(1, 3, 256, 256)
flops, params = profile(model, (dummy_input, dummy_input))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
