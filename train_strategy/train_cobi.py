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
import cb_loss

import pdb
import wandb


# TODO YAML
parser = argparse.ArgumentParser()

# exp name
parser.add_argument('--name', type=str, default="H_codebase_HDR")

# traing strategy
parser.add_argument('--num_epoch', type=int, default=350)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--gamma', type=float, default=0.5) #Learning rate strategy
parser.add_argument('--log_epoch', type=int, default=20)
parser.add_argument('--loader_worker', type=int, default=1)
parser.add_argument('--band_width', type=float, default=0.5)
parser.add_argument('--loss_mode', type=str, default='sum')
parser.add_argument('--cobi_eff', type=float, default=0.1)

# Pretrained
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('--PT_exp', type=str, default='default')

# dataset
parser.add_argument('--set_name', type=str, default='half') # dataset choice
parser.add_argument('--img_num', type=int, default=7)
parser.add_argument('--img_set', type=int, default=4)
parser.add_argument('--dataset_mode', type=str, default='decrease') 
parser.add_argument('--img_height', type=int, default=256)
parser.add_argument('--cache', action='store_true', default=False)
parser.add_argument('--VDS_our', action='store_true', default=False)

# model
parser.add_argument('--model_name', type=str, default='affine_pad')
parser.add_argument('--decode_name', type=str, default='mult')
parser.add_argument('--act', type=str, default='leaky_relu')
parser.add_argument('--mlp_num', type=int, default=3)
parser.add_argument('--pretrain', type=str, default='vgg')
parser.add_argument('--cycle', action='store_true', default=False)
parser.add_argument('--sep', type=float, default=0.5)

# record
parser.add_argument('--check_path', type=str, default='default_inc')
parser.add_argument('--data_root', type=str, default='/home/skchen/ML_practice/LIIF_on_HDR/VDS_dataset')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--wandb', action='store_true', default=False)

args = parser.parse_args()

print("!!!!!!!!!!!!!!!!!!!!!!cache: ", args.cache)
print("!!!!!!!!!!!!!!!!!!!!!!wandb: ", args.wandb)
print("!!!!!!!!!!!!!!!!!!!!!!log_epoch: ", args.log_epoch)
print("!!!!!!!!!!!!!!!!!!!!!!cobi_eff: ", args.cobi_eff)

if args.wandb == True:
    exp_name = args.name + "_" + args.dataset_mode
    wandb.init(name=exp_name, project="Hunlin_codebase_HDR_ModelArchi")

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
train_loader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=args.loader_worker)
test_loader = DataLoader(testset, args.batch_size, shuffle=False, num_workers=args.loader_worker)

# Model
if args.cycle:
    from core.cycle_model import build_network
    print('cycle model')
    model = build_network(args)
else:
    from core.HDR_model import build_network
    print('normal model')
    model = build_network(args)
model.to(device)

if args.pretrained:
    weight_path = "./experiment/" + args.PT_exp + "/" + args.check_path + "/final_model.pth"
    model.load_state_dict(torch.load(weight_path))
    print("Pretrained weight loading")


# Training and testing process
#lossL1 = nn.L1Loss(reduction='sum')
lossL1 = nn.L1Loss(reduction=args.loss_mode)
lossCobi = cb_loss.ContextualBilateralLoss(band_width=args.band_width, use_vgg = True, vgg_layer = 'relu3_4', 
                           device = 'cuda')

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=args.gamma)

for epoch in tqdm(range(args.num_epoch)):
    model.train()
    train_loss_l1 = 0.0
    train_loss_cobi = 0.0
    train_loss_total = 0.0
    train_psnr = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        
        img, gt, step,  ori = data
        img = img.to(device)
        gt = gt.to(device)
        step = step.to(device)
        ori = ori.to(device)

        out = model(img, step, ori)
        loss_l1 = lossL1(out, gt)
        loss_cobi = lossCobi(out, gt)

        loss = loss_l1 + args.cobi_eff * loss_cobi
        

        with torch.no_grad():
            psnr = avg_psnr(out, gt, reduce=False)

        loss.backward()
        optimizer.step()

        train_loss_l1 += loss_l1.item()
        train_loss_cobi += loss_cobi.item()
        train_loss_total += loss.item()
        train_psnr += psnr

    train_psnr /= len(trainset)  

    test_loss_total = 0.0
    test_psnr = 0.0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            img, gt, step, ori = data
            img = img.to(device)
            gt = gt.to(device)
            step = step.to(device)
            ori = ori.to(device)

            out = model(img, step, ori)
            loss_l1 = lossL1(out, gt)
            loss_cobi = lossCobi(out, gt)
            loss = loss_l1 + args.cobi_eff * loss_cobi
            psnr = avg_psnr(out, gt, reduce=False)
            
            test_loss_total += loss.item()
            test_psnr += psnr

    test_psnr /= len(testset)

    if args.wandb == True:
        wandb.log({"train_loss_l1": train_loss_l1, "train_loss_cobi": train_loss_cobi, "train_loss_total": train_loss_total, "test_psnr": test_psnr}) 

    scheduler.step(test_loss_total)

    if (epoch + 1) % args.log_epoch == 0:
        name = 'model_' + str(epoch+1) + '.pth'
        torch.save(model.state_dict(),  check_path + '/' + name)

print(test_psnr)
torch.save(model.state_dict(), check_path + '/final_model.pth')
