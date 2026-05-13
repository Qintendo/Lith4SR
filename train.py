import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./auxiliary/'))
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)

import utils
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True
# from piqa import SSIM
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx
from utils import save_img
from losses import CharbonnierLoss, cosine_map_loss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import get_training_data, get_validation_data

from muon import Muon

######### Logs dir ###########
log_dir = os.path.join(dir_name, 'log', opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)



######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### DataParallel ###########
model_restoration = torch.nn.DataParallel (model_restoration)
model_restoration.cuda()

######### optimizer builder ###########
def build_optimizer(model):
    if opt.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),
                          eps=1e-8, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'muon':
        muon_params = [p for name, p in model.named_parameters()
                       if p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name]
        adamw_params = [p for name, p in model.named_parameters()
                        if not (p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name)]
        return Muon(lr=opt.lr_initial, wd=opt.weight_decay,
                    muon_params=muon_params, adamw_params=adamw_params)
    else:
        raise Exception("Error optimizer...")

######### Resume / Finetune / Fresh ###########
start_epoch = 1
best_psnr = 0.0

if opt.resume:
    path_chk_rest = opt.pretrain_weights
    print('===> Resuming from %s' % os.path.abspath(path_chk_rest))
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    best_psnr = 0.0 if opt.reset_best else utils.load_best_psnr(path_chk_rest)

    optimizer = build_optimizer(model_restoration)

    lr_resumed = utils.load_optim(optimizer, path_chk_rest)
    if lr_resumed is None:
        lr_resumed = optimizer.param_groups[0]['lr']
    for pg in optimizer.param_groups:
        pg['lr'] = lr_resumed

    print('------------------------------------------------------------------------------')
    print(f"==> Resuming at epoch {start_epoch} with lr={lr_resumed}")
    print('------------------------------------------------------------------------------')

elif getattr(opt, 'finetune', False) and opt.pretrain_weights:
    # ft
    path_chk_rest = opt.pretrain_weights
    print('===> Finetune: loading weights ONLY from %s' % os.path.abspath(path_chk_rest))
    ckpt = torch.load(path_chk_rest, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    model_restoration.load_state_dict(state, strict=False)

    start_epoch = 1
    best_psnr = 0.0
    optimizer = build_optimizer(model_restoration)
    print(f"==> Finetune start: epoch={start_epoch}, best_psnr reset, lr={opt.lr_initial}")

else:
    optimizer = build_optimizer(model_restoration)

if getattr(opt, 'epochs_to_run', 0):
    opt.nepoch = start_epoch + opt.epochs_to_run - 1

######### Scheduler ###########
if opt.resume:
    print("Resuming with cosine schedule (no warmup).")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.nepoch - start_epoch + 1, eta_min=1e-6)
else:
    if opt.warmup:
        print("Using warmup + cosine.")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.nepoch - warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    else:
        step = 50
        print(f"Using StepLR, step={step}!")
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)



######### Loss ###########
criterion = CharbonnierLoss().cuda() if opt.criterion.lower() == "charbonnier" else nn.L1Loss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_epoch = 0
best_iter = 0
eval_now = 1000
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
ii=0
index = 0
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    epoch_ssim_loss = 0
    for i, data in enumerate(train_loader, 0): 
        # zero_grad
        index += 1
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()
        hsl_in = data[2].cuda()
        if epoch > 5:
            target, input_ = utils.MixUp_AUG().aug(target, input_)
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_, hsl_in)
            restored = torch.clamp(restored,0,1)
            loss = criterion(restored, target)

            m = model_restoration.module
            align_loss = 0.0

            # full-res: encoderlayer_0 adaptor  vs  decoderlayer_2 feature
            if (getattr(m, "_enc_aug_full", None) is not None) and (getattr(m, "_dec_feat_full", None) is not None):
                dec_full_proj = m.proj_dec_full(m._dec_feat_full).detach()  # stop-grad decoder
                align_loss += cosine_map_loss(m._enc_aug_full, dec_full_proj)

            # half-res: encoderlayer_1 adaptor  vs  decoderlayer_1 feature
            if (getattr(m, "_enc_aug_half", None) is not None) and (getattr(m, "_dec_feat_half", None) is not None):
                dec_half_proj = m.proj_dec_half(m._dec_feat_half).detach()  # stop-grad decoder
                align_loss += cosine_map_loss(m._enc_aug_half, dec_half_proj)

            loss = loss + opt.lam_align * align_loss

        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
        epoch_loss += loss.item()
        #### Evaluation ####
        if (index + 1) % eval_now == 0 and i > 0:
            # eval_shadow_rmse = 0
            # eval_nonshadow_rmse = 0
            # eval_rmse = 0
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    hsl_in = data_val[2].cuda()
                    filenames = data_val[3]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_, hsl_in)
                    restored = torch.clamp(restored,0,1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb)/len(val_loader)
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'best_psnr': best_psnr,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))
                print("[Ep %d it %d\t PSNR : %.4f] " % (epoch, i, psnr_val_rgb))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep %d best_it %d Best_PSNR %.4f] " \
                        % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
                model_restoration.train()
                torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    epoch_time = time.time() - epoch_start_time

    avg_loss = epoch_loss / (i + 1) if (i + 1) > 0 else float('nan')

    lr_used = optimizer.param_groups[0]['lr']

    print()

    print("------------------------------------------------------------------")
    print(f"Epoch: {epoch}\tTime: {epoch_time:.2f}s\tLoss: {avg_loss:.6f}\tLearningRate {lr_used:.8f}", flush=True)
    print("------------------------------------------------------------------")

    with open(logname, 'a', encoding='utf-8') as f:
        f.write(f"Epoch: {epoch}\tTime: {epoch_time:.2f}s\tLoss: {avg_loss:.6f}\tLearningRate {lr_used:.8f}\n")
        f.flush()

    scheduler.step()

    torch.save({'epoch': epoch,
                'best_psnr': best_psnr,
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch,
                    'best_psnr': best_psnr,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch)))
print("Now time is : ",datetime.datetime.now().isoformat())



