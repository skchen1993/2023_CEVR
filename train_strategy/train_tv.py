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

import tv_loss

import wandb
import pdb


# TODO YAML
parser = argparse.ArgumentParser()

# exp name
parser.add_argument('--name', type=str, default="H_codebase_HDR")

# traing strategy
parser.add_argument('--num_epoch', type=int, default=350)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--gamma', type=float, default=0.5) #Learning rate strategy
parser.add_argument('--lr_step', type=int, default=100)
parser.add_argument('--log_epoch', type=int, default=20)
parser.add_argument('--loader_worker', type=int, default=2)
parser.add_argument('--loss_mode', type=str, default='sum')
parser.add_argument('--tv_coff', type=float, default=0.1)



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
print("!!!!!!!!!!!!!!!!!!!!!!VDS_our: ", args.VDS_our)
print("!!!!!!!!!!!!!!!!!!!!!!log_epoch: ", args.log_epoch)

if args.wandb == True:
    exp_name = args.name + "_" + args.dataset_mode
    wandb.init(name=exp_name, project="Hunlin_codebase_HDR")

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

# Training and testing process
criterion_l1 = nn.L1Loss(reduction=args.loss_mode)
criterion_tv = tv_loss.TVLoss(mode=args.loss_mode)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=args.gamma)

for epoch in tqdm(range(args.num_epoch)):
    model.train()
    train_loss_l1 = 0.0
    train_loss_tv = 0.0
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
        loss_l1 = criterion_l1(out, gt)
        loss_tv = criterion_tv(out)

        #pdb.set_trace()

        loss = loss_l1 + args.tv_coff * loss_tv

        with torch.no_grad():
            psnr = avg_psnr(out, gt, reduce=False)

        loss.backward()
        optimizer.step()

        train_loss_l1 += loss_l1.item()
        train_loss_tv += loss_tv.item()
        train_loss_total += loss.item()
        train_psnr += psnr

    train_loss_l1 /= len(trainset)
    train_loss_tv /= len(trainset)
    train_loss_total /= len(trainset)
    train_psnr /= len(trainset)

    test_loss = 0.0
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
            loss_l1 = criterion_l1(out, gt)
            loss_tv = criterion_tv(out)
            loss = loss_l1 + args.tv_coff * loss_tv

            psnr = avg_psnr(out, gt, reduce=False)
            
            test_loss += loss.item()
            test_psnr += psnr

    test_loss /= len(testset)
    test_psnr /= len(testset)

    if args.wandb == True:
        wandb.log({"train_loss_l1": train_loss_l1, "train_loss_tv": train_loss_tv, "train_loss_total": train_loss_total, "test_psnr": test_psnr}) 

    cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
    scheduler.step(test_loss)

    if (epoch + 1) % args.log_epoch == 0:
        name = 'model_' + str(epoch+1) + '.pth'
        torch.save(model.state_dict(),  check_path + '/' + name)

print(test_psnr)
torch.save(model.state_dict(), check_path + '/final_model.pth')
