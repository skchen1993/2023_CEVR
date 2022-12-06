import sys
path = "../"
sys.path.append(path)

import os
import os.path
from os import path
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.Data_provider import build_dataset
from core.utils import avg_psnr
import pdb
import wandb

#Loss
import perceptual_loss
import cb_loss
import tv_loss

import random


torch.set_num_threads(5)


# TODO YAML
parser = argparse.ArgumentParser()

# exp name
parser.add_argument('--name', type=str, default="H_codebase_HDR")

# traing strategy
parser.add_argument('--num_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--log_epoch', type=int, default=20)
parser.add_argument('--loader_worker', type=int, default=1)
parser.add_argument('--loss_mode', type=str, default='sum')
parser.add_argument('--warmup_thr', type=float, default=28)

# LR: ReduceLROnPlateau
parser.add_argument('--LR_Plateau', action='store_true', default=False) 
parser.add_argument('--gamma', type=float, default=0.5)
# LR: CosineAnnealingWarmRestarts
parser.add_argument('--LR_CosAnneal', action='store_true', default=False) 

# Loss specify
parser.add_argument('--FloatLoss', action='store_true', default=False)
parser.add_argument('--Float_eff', type=float, default=0.1)

parser.add_argument('--FloatLossv2', action='store_true', default=False)
parser.add_argument('--Floatv2_eff', type=float, default=0.1)

parser.add_argument('--FloatLossv3', action='store_true', default=False)
parser.add_argument('--Floatv3_eff', type=float, default=0.1)

parser.add_argument('--PerceptualLoss', action='store_true', default=False)
parser.add_argument('--Percept_eff', type=float, default=100)

parser.add_argument('--CobiLoss', action='store_true', default=False)
parser.add_argument('--Cobi_eff', type=float, default=100)
parser.add_argument('--band_width', type=float, default=0.5)

parser.add_argument('--TVLoss', action='store_true', default=False)
parser.add_argument('--TV_eff', type=float, default=100)


# dataset
parser.add_argument('--set_name', type=str, default='half') # dataset choice
parser.add_argument('--img_num', type=int, default=7)
parser.add_argument('--img_set', type=int, default=4)
parser.add_argument('--dataset_mode', type=str, default='decrease') 
parser.add_argument('--img_height', type=int, default=256)
parser.add_argument('--cache', action='store_true', default=False)
parser.add_argument('--augment', action='store_true', default=False)
parser.add_argument('--norm', action='store_true', default=False)
parser.add_argument('--VDS_our', action='store_true', default=False)

# model
parser.add_argument('--model_name', type=str, default='affine_pad')
parser.add_argument('--decode_name', type=str, default='mult')
parser.add_argument('--act', type=str, default='leaky_relu')
parser.add_argument('--mlp_num', type=int, default=3)
parser.add_argument('--pretrain', type=str, default='vgg')
parser.add_argument('--cycle', action='store_true', default=False)
parser.add_argument('--sep', type=float, default=0.5)
#parser.add_argument('--DIF',  action='store_true', default=False)
parser.add_argument('--EV_info',  type=int, default=2, help="1: only cat dif, 2: cat source and dif, 3: Embed DIF to 16 dim vec")
parser.add_argument('--init_weight',  action='store_true', default=False)


# record
parser.add_argument('--check_path', type=str, default='default_inc')
parser.add_argument('--data_root', type=str, default='/home/skchen/ML_practice/LIIF_on_HDR/VDS_dataset')
parser.add_argument('--seed', type=int, default=14589521019421467395)
parser.add_argument('--wandb', action='store_true', default=False)

args = parser.parse_args()

print("!!!!!!!!!!!!!!!!!!!!!!cache: ", args.cache)
print("!!!!!!!!!!!!!!!!!!!!!!wandb: ", args.wandb)
print("!!!!!!!!!!!!!!!!!!!!!!log_epoch: ", args.log_epoch)
print("!!!!!!!!!!!!!!!!!!!!!!FloatLoss: ", args.FloatLoss)
print("!!!!!!!!!!!!!!!!!!!!!!FloatLossv2: ", args.FloatLossv2)
print("!!!!!!!!!!!!!!!!!!!!!!FloatLossv3: ", args.FloatLossv3)
print("!!!!!!!!!!!!!!!!!!!!!!PerceptualLoss: ", args.PerceptualLoss)
print("!!!!!!!!!!!!!!!!!!!!!!CobiLoss: ", args.CobiLoss)
print("!!!!!!!!!!!!!!!!!!!!!!TVLoss: ", args.TVLoss)
print("!!!!!!!!!!!!!!!!!!!!!!warmup_thr: ", args.warmup_thr)
print("!!!!!!!!!!!!!!!!!!!!!!LR_CosAnneal: ", args.LR_CosAnneal)
print("!!!!!!!!!!!!!!!!!!!!!!augment: ", args.augment)
print("!!!!!!!!!!!!!!!!!!!!!!norm: ", args.norm)
print("!!!!!!!!!!!!!!!!!!!!!!EV_info: ", args.EV_info)

if args.wandb == True:
    exp_name = args.name + "_" + args.dataset_mode
    wandb.init(name=exp_name, project="SecondHalf_Exp")

# Folder establish
exp_path = "./experiment/" + args.name
if path.exists(exp_path) == False:
    print("makedir: ", exp_path)
    os.makedirs(exp_path)
else:
    print("path: ", exp_path, " existed!")

check_path = exp_path + "/" + args.check_path
if path.exists(check_path) == False:
    print("makedir: ", check_path)
    os.makedirs(check_path)
else:
    print("path: ", check_path, " existed!")

# Device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.seed:
    torch.manual_seed(args.seed)
    seed = args.seed
else:
    seed = torch.seed()
print("Initializing with device:", device, ",seed:", seed)

# Dataset 
print('dataset:', args.set_name, ', batch size:', args.batch_size)
trainset, testset = build_dataset(args)
train_loader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=args.loader_worker, pin_memory=False)
test_loader = DataLoader(testset, args.batch_size, shuffle=False, num_workers=args.loader_worker, pin_memory=False)

# Model
from core.HDR_model import build_network
print("args.model_name: ", args.model_name)
print('normal model')
model = build_network(args)
model.to(device)


# Training and testing process

# Loss
criterion_l1 = nn.L1Loss(reduction=args.loss_mode)
if args.PerceptualLoss:
    feature_layers=[2]
    criterion_per = perceptual_loss.VGGPerceptualLoss(feature_layers=feature_layers, style_layers=[]).to('cuda')

if args.CobiLoss:
    criterion_cobi = cb_loss.ContextualBilateralLoss(band_width=args.band_width, use_vgg = True, vgg_layer = 'relu3_4', device = 'cuda')

if args.TVLoss:
    criterion_tv = tv_loss.TVLoss()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.LR_Plateau:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=args.gamma)
if args.LR_CosAnneal:
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2, eta_min=0, last_epoch=-1, verbose=False)
    

# Loss record
loss_record_train = {"train_loss":0, "train_L1Loss":0 , "train_psnr":0, "lr":0}
loss_record_test = {"test_loss":0, "test_psnr":0} 

if args.PerceptualLoss:
    loss_record_train['train_PerceptLoss'] = 0
if args.CobiLoss:
    loss_record_train['train_CobiLoss'] = 0
if args.FloatLoss:
    loss_record_train['train_FloatLoss'] = 0
if args.FloatLossv2:
    loss_record_train['train_FloatLossv2'] = 0
if args.FloatLossv3:
    loss_record_train['train_FloatLossv3'] = 0
if args.TVLoss:
    loss_record_train['train_TVLoss'] = 0

# For warmup threshold
psnr_rec = 1
psnr_best= 1

iters = len(train_loader)

for epoch in tqdm(range(args.num_epoch)):
    model.train()

    # loss record initialization
    for key in loss_record_train:
        loss_record_train[key] = 0
    for key in loss_record_test:
        loss_record_test[key] = 0

    #for data in train_loader:
    for i, data in enumerate(train_loader):
        loss_train = 0 # total training loss
        loss_l1 = 0
        loss_percept = 0
        loss_cobi = 0
        loss_float = 0
        loss_floatv2 = 0
        loss_floatv3 = 0
        loss_tv = 0


        optimizer.zero_grad()
        
        img, gt, step,  ori = data
        img = img.to(device)
        gt = gt.to(device)
        step = step.to(device)
        ori = ori.to(device)

        out = model(img, step, ori)
        
        #l1 loss
        loss_l1 = criterion_l1(out, gt)
        loss_record_train["train_L1Loss"] += loss_l1.item()
        #loss_train += loss_l1

        if psnr_rec > args.warmup_thr:
            #perceptual loss
            if args.PerceptualLoss:
                loss_percept = criterion_per(out, gt)
                loss_record_train["train_PerceptLoss"] += loss_percept.item()

            #Cobi loss
            if args.CobiLoss:
                loss_cobi = criterion_cobi(out, gt)
                loss_record_train["train_CobiLoss"] += loss_cobi.item()

            #Naive float loss (直接/2)
            if args.FloatLoss:
                out_temp = model(img, step/2, ori)
                out_float = model(out_temp, step/2, step/2)
                loss_float = criterion_l1(out_float, gt)
                loss_record_train["train_FloatLoss"] += loss_float.item()

            #Float_loss_v2 
            if args.FloatLossv2:
                for i in range(args.batch_size):
                    if step[i] > 0:
                        step[i] = step[i] - 0.5
                        step2 = torch.full((args.batch_size, 1), 0.5).to(device)
                    elif step[i] < 0:
                        step[i] = step[i] + 0.5
                        step2 = torch.full((args.batch_size, 1), -0.5).to(device)

                out_temp = model(img, step, ori)
                out_float = model(out_temp, step2, step)
                loss_floatv2 = criterion_l1(out_float, gt)
                loss_record_train["train_FloatLossv2"] += loss_floatv2.item()

            #Float_loss_v3
            if args.FloatLossv3:
                rand_fact = random.uniform(0.1, 0.9)
                step1 = step * rand_fact
                step2 = step * (1 - rand_fact)

                out_temp = model(img, step1, ori)
                out_float = model(out_temp, step2, step1)
                loss_floatv3 = criterion_l1(out_float, gt)
                loss_record_train["train_FloatLossv3"] += loss_floatv3.item()      


            if args.TVLoss:
                loss_tv = criterion_tv(out)
                loss_record_train["train_TVLoss"] += loss_tv.item()
            
        loss_train = loss_l1 + args.Float_eff * loss_float + args.Floatv2_eff * loss_floatv2 + args.Floatv3_eff * loss_floatv3 + args.Percept_eff * loss_percept + args.Cobi_eff * loss_cobi + args.TV_eff * loss_tv
     
        with torch.no_grad():
            psnr = avg_psnr(out, gt, reduce=False)

        loss_train.backward()
        optimizer.step()

        loss_record_train["train_loss"] += loss_train.item()
        loss_record_train["train_psnr"] += psnr

        if args.LR_CosAnneal:
            scheduler.step(epoch + i / iters)


    with torch.no_grad():
        loss_test = 0 # total testing loss
        model.eval()
        for data in test_loader:
            img, gt, step, ori = data
            img = img.to(device)
            gt = gt.to(device)
            step = step.to(device)
            ori = ori.to(device)

            out = model(img, step, ori)

            #l1 loss
            loss_l1 = criterion_l1(out, gt)
            loss_test += loss_l1

            if psnr_rec > args.warmup_thr:
                #perceptual loss
                if args.PerceptualLoss:
                    loss_percept = criterion_per(out, gt)
                    loss_test += args.Percept_eff * loss_percept

                #Cobi loss
                if args.CobiLoss:
                    loss_cobi = criterion_cobi(out, gt)
                    loss_test += args.Cobi_eff * loss_cobi

                #Naive float loss (直接/2)
                if args.FloatLoss:
                    out_temp = model(img, step/2, ori)
                    out_float = model(out_temp, step/2, step/2)
                    loss_float = criterion_l1(out_float, gt)
                    loss_test += args.Float_eff * loss_float

                #Float_loss_v2 
                if args.FloatLossv2:
                    for i in range(args.batch_size):
                        if step[i] > 0:
                            step[i] = step[i] - 0.5
                            step2 = torch.full((args.batch_size, 1), 0.5).to(device)
                        elif step[i] < 0:
                            step[i] = step[i] + 0.5
                            step2 = torch.full((args.batch_size, 1), -0.5).to(device)

                    out_temp = model(img, step, ori)
                    out_float = model(out_temp, step2, step)
                    loss_floatv2 = criterion_l1(out_float, gt)
                    loss_test += args.Floatv2_eff * loss_floatv2

                #Float_loss_v3 
                if args.FloatLossv3:
                    rand_fact = random.uniform(0.1, 0.9)
                    step1 = step * rand_fact
                    step2 = step * (1 - rand_fact)      
  
                    out_temp = model(img, step1, ori)
                    out_float = model(out_temp, step2, step1)
                    loss_floatv3 = criterion_l1(out_float, gt)
                    loss_test += args.Floatv3_eff * loss_floatv3

                if args.TVLoss:
                    loss_tv = criterion_tv(out)
                    loss_test += args.TV_eff * loss_tv
                    

            """
            Prune out EV0
            """
            nonzero_index = step != 0
            nonzero_index = nonzero_index.squeeze(1)

            nonzero_gt = gt[nonzero_index]
            nonzero_out = out[nonzero_index]

            psnr = avg_psnr(nonzero_out, nonzero_gt, reduce=False)

            
            loss_record_test["test_loss"] += loss_test.item()
            loss_record_test["test_psnr"] += psnr

    # Average accumulated record 
    loss_test /= len(testset)
    for key in loss_record_train:
        loss_record_train[key] /= len(trainset)   
    for key in loss_record_test:
        loss_record_test[key] /= (len(testset) * 3/4) 

    psnr_rec = loss_record_test["test_psnr"]

    if psnr_rec > psnr_best:
        psnr_best = psnr_rec
        torch.save(model.state_dict(), check_path + '/model_best.pth')

    cur_lr = optimizer.state_dict()['param_groups'][0]['lr']

    if args.LR_Plateau:
        scheduler.step(loss_test)

    loss_record_train["lr"] = cur_lr 

    if args.wandb == True:
        all_record = {**loss_record_train,**loss_record_test}
        wandb.log(all_record) 

    if (epoch + 1) % args.log_epoch == 0:
        name = 'model_' + str(epoch+1) + '.pth'
        torch.save(model.state_dict(),  check_path + '/' + name)

torch.save(model.state_dict(), check_path + '/final_model.pth')
print("best PSNR: ", psnr_best)