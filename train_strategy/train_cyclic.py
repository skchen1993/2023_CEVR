##################### Warning!!!!!!!!!!!!!!!!!!!!!!!!!#######################
# cyclic操作下, 如果要採用 "cat DIF" 的作法, 要記得去改寫 DIF 的運算, 不能再直接拿step當 DIF
# 當D output出 EV-1 (step=-1) 後, B model 的recover要記得把 step * (-1) -> 變成1
#
#
############################################################################

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

parser.add_argument('--PerceptualLoss', action='store_true', default=False)
parser.add_argument('--Percept_eff', type=float, default=100)

parser.add_argument('--CobiLoss', action='store_true', default=False)
parser.add_argument('--Cobi_eff', type=float, default=100)
parser.add_argument('--band_width', type=float, default=0.5)

# dataset
parser.add_argument('--set_name', type=str, default='half') # dataset choice
parser.add_argument('--img_set', type=int, default=4)
parser.add_argument('--img_num', type=int, default=7)
parser.add_argument('--dataset_mode', type=str, default='')  #兩個都用
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


#pretrained
parser.add_argument('--pretrained_exp', type=str, default='./experiment/decoder_ResizeConv_Bicubic/')
parser.add_argument('--weight_name', type=str, default='final_model.pth')



# record
parser.add_argument('--data_root', type=str, default='/home/skchen/ML_practice/LIIF_on_HDR/VDS_dataset')
parser.add_argument('--seed', type=int, default=14589521019421467395)
parser.add_argument('--wandb', action='store_true', default=False)

args = parser.parse_args()

print("!!!!!!!!!!!!!!!!!!!!!!cache: ", args.cache)
print("!!!!!!!!!!!!!!!!!!!!!!wandb: ", args.wandb)
print("!!!!!!!!!!!!!!!!!!!!!!log_epoch: ", args.log_epoch)
print("!!!!!!!!!!!!!!!!!!!!!!FloatLoss: ", args.FloatLoss)
print("!!!!!!!!!!!!!!!!!!!!!!PerceptualLoss: ", args.PerceptualLoss)
print("!!!!!!!!!!!!!!!!!!!!!!CobiLoss: ", args.CobiLoss)
print("!!!!!!!!!!!!!!!!!!!!!!warmup_thr: ", args.warmup_thr)
print("!!!!!!!!!!!!!!!!!!!!!!LR_CosAnneal: ", args.LR_CosAnneal)
print("!!!!!!!!!!!!!!!!!!!!!!augment: ", args.augment)
print("!!!!!!!!!!!!!!!!!!!!!!norm: ", args.norm)
print("!!!!!!!!!!!!!!!!!!!!!!EV_info: ", args.EV_info)

if args.wandb == True:
    exp_name = args.name + "_" + args.dataset_mode
    wandb.init(name=exp_name, project="cyclic_training strategy")

# Folder establish
exp_path = "./experiment/" + args.name
if path.exists(exp_path) == False:
    print("exp_path makedir: ", exp_path)
    os.makedirs(exp_path)
else:
    print("exp_path: ", exp_path, " existed!")

check_path_dec = exp_path + "/" + "dec"
if path.exists(check_path_dec) == False:
    print("check_path_dec makedir: ", check_path_dec)
    os.makedirs(check_path_dec)
else:
    print("check_path_dec path: ", check_path_dec, " existed!")

check_path_inc = exp_path + "/" + "inc"
if path.exists(check_path_inc) == False:
    print("check_path_inc makedir: ", check_path_inc)
    os.makedirs(check_path_inc)
else:
    print("check_path_inc path: ", check_path_inc, " existed!")


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
#----- INC dataset:
args.dataset_mode = 'increase'
trainset_inc, testset_inc = build_dataset(args)
train_loader_inc = DataLoader(trainset_inc, args.batch_size, shuffle=True, num_workers=args.loader_worker, pin_memory=False)
test_loader_inc = DataLoader(testset_inc, args.batch_size, shuffle=False, num_workers=args.loader_worker, pin_memory=False)

#----- DEC dataset:
args.dataset_mode = 'decrease'
trainset_dec, testset_dec = build_dataset(args)
train_loader_dec = DataLoader(trainset_dec, args.batch_size, shuffle=True, num_workers=args.loader_worker, pin_memory=False)
test_loader_dec = DataLoader(testset_dec, args.batch_size, shuffle=False, num_workers=args.loader_worker, pin_memory=False)


# Model establish and pretrained weight loading
if args.cycle:
    from core.cycle_model import build_network
    print('cycle model')
    model = build_network(args)
else:
    from core.HDR_model import build_network
    print('normal model')
    model_B = build_network(args).to(device)
    model_D = build_network(args).to(device)

    inc_PATH = args.pretrained_exp + "/inc/" + args.weight_name
    dec_PATH = args.pretrained_exp + "/dec/" + args.weight_name

    print()

    model_B.load_state_dict(torch.load(inc_PATH))
    model_D.load_state_dict(torch.load(dec_PATH))
    print("Pretrained model weight loading successfully!!!!")



# Training and testing process

# Loss
criterion_l1 = nn.L1Loss(reduction=args.loss_mode)
if args.PerceptualLoss:
    feature_layers=[2]
    criterion_per = perceptual_loss.VGGPerceptualLoss(feature_layers=feature_layers, style_layers=[]).to('cuda')

if args.CobiLoss:
    criterion_cobi = cb_loss.ContextualBilateralLoss(band_width=args.band_width, use_vgg = True, vgg_layer = 'relu3_4', device = 'cuda')

# Maintain two optimizer for B, D model
optimizer_B = optim.Adam(model_B.parameters(), lr=args.lr)
optimizer_D = optim.Adam(model_D.parameters(), lr=args.lr)

if args.LR_Plateau:
    scheduler_B = optim.lr_scheduler.ReduceLROnPlateau(optimizer_B,factor=args.gamma)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D,factor=args.gamma)
if args.LR_CosAnneal:
    scheduler_B = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_B, 10, T_mult=2, eta_min=0, last_epoch=-1, verbose=False)
    scheduler_D = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D, 10, T_mult=2, eta_min=0, last_epoch=-1, verbose=False)   

# Loss record
loss_record_train_B = {"train_loss_B":0, "train_L1Loss_B":0, "train_CyclicLoss_B":0, "lr_B":0}
loss_record_test_B = {"test_loss_B":0, "test_psnr_B":0} 
if args.PerceptualLoss:
    loss_record_train_B['train_PerceptLoss_B'] = 0
if args.CobiLoss:
    loss_record_train_B['train_CobiLoss_B'] = 0
if args.FloatLoss:
    loss_record_train_B['train_FloatLoss_B'] = 0

loss_record_train_D = {"train_loss_D":0, "train_L1Loss_D":0, "train_CyclicLoss_D":0, "lr_D":0}
loss_record_test_D = {"test_loss_D":0, "test_psnr_D":0} 
if args.PerceptualLoss:
    loss_record_train_D['train_PerceptLoss_D'] = 0
if args.CobiLoss:
    loss_record_train_D['train_CobiLoss_D'] = 0
if args.FloatLoss:
    loss_record_train_D['train_FloatLoss_D'] = 0


# For warmup threshold
psnr_rec = 1
iters = len(train_loader_dec)

####
#Debug 
####
if False:
    psnr_D = 0
    psnr_B = 0
    with torch.no_grad():
        ############################
        # EVAL on Model_D
        ###########################        
        loss_test_D = 0 # total testing loss
        model_D.eval()
        for data in test_loader_dec:
            img, gt, step, ori = data
            img = img.to(device)
            gt = gt.to(device)
            step = step.to(device)
            ori = ori.to(device)

            #print("EVAL_D. step: ", step)
            #print("EVAL_D. ori: ", ori)
            

            out = model_D(img, step, ori)

            #l1 loss
            loss_l1 = criterion_l1(out, gt)
            loss_test_D += loss_l1

            if psnr_rec > args.warmup_thr:
                #perceptual loss
                if args.PerceptualLoss:
                    loss_percept = criterion_per(out, gt)
                    loss_test_D += args.Percept_eff * loss_percept

                #Cobi loss
                if args.CobiLoss:
                    loss_cobi = criterion_cobi(out, gt)
                    loss_test_D += args.Cobi_eff * loss_cobi

                #Naive float loss (直接/2)
                if args.FloatLoss:
                    out_temp = model(img, step/2, ori)
                    out_float = model(out_temp, step/2, step/2)
                    loss_float = criterion_l1(out_float, gt)
                    loss_test_D += args.Float_eff * loss_float

            """
            Prune out EV0
            """
            nonzero_index = step != 0
            nonzero_index = nonzero_index.squeeze(1)

            nonzero_gt = gt[nonzero_index]
            nonzero_out = out[nonzero_index]

            psnr = avg_psnr(nonzero_out, nonzero_gt, reduce=False)
            psnr_D = psnr_D + psnr


        ############################
        # EVAL on Model_B
        ###########################        
        loss_test_B = 0 # total testing loss
        model_B.eval()
        for data in test_loader_inc:
            img, gt, step, ori = data
            img = img.to(device)
            gt = gt.to(device)
            step = step.to(device)
            ori = ori.to(device)

            #print("EVAL_B. step: ", step)
            #print("EVAL_B. ori: ", ori)
            

            out = model_B(img, step, ori)

            #l1 loss
            loss_l1 = criterion_l1(out, gt)
            loss_test_B += loss_l1

            if psnr_rec > args.warmup_thr:
                #perceptual loss
                if args.PerceptualLoss:
                    loss_percept = criterion_per(out, gt)
                    loss_test_B += args.Percept_eff * loss_percept

                #Cobi loss
                if args.CobiLoss:
                    loss_cobi = criterion_cobi(out, gt)
                    loss_test_B += args.Cobi_eff * loss_cobi

                #Naive float loss (直接/2)
                if args.FloatLoss:
                    out_temp = model(img, step/2, ori)
                    out_float = model(out_temp, step/2, step/2)
                    loss_float = criterion_l1(out_float, gt)
                    loss_test_B += args.Float_eff * loss_float

            """
            Prune out EV0
            """
            nonzero_index = step != 0
            nonzero_index = nonzero_index.squeeze(1)

            nonzero_gt = gt[nonzero_index]
            nonzero_out = out[nonzero_index]

            psnr = avg_psnr(nonzero_out, nonzero_gt, reduce=False)
            psnr_B = psnr_B + psnr

    psnr_B /= (len(testset_dec) * 3/4) 
    psnr_D /= (len(testset_dec) * 3/4) 

    print("avg psnr_B: ", psnr_B)
    print("avg psnr_D: ", psnr_D)



# 確定model 初始參數沒有問題
for epoch in tqdm(range(args.num_epoch)):
    model_B.train()
    model_D.train()

    # loss record initialization
    for key in loss_record_train_B:
        loss_record_train_B[key] = 0
    for key in loss_record_test_B:
        loss_record_test_B[key] = 0
    for key in loss_record_train_D:
        loss_record_train_D[key] = 0
    for key in loss_record_test_D:
        loss_record_test_D[key] = 0

    ############################
    # (1) D predict, B recover
    ###########################
    for i, data in enumerate(train_loader_dec):
        loss_train = 0 # total training loss
        loss_l1 = 0
        loss_percept = 0
        loss_cobi = 0
        loss_float = 0
        loss_cyclic = 0

        ############################
        # (1-1)L1 loss update Model_D
        ###########################
        optimizer_D.zero_grad()
        
        #EX: img: EV0, gt: EV-3, step:-3, ori:0 
        img, gt, step, ori = data
        img = img.to(device)
        gt = gt.to(device)
        step = step.to(device)
        ori = ori.to(device)

        #print("1Train. step: ", step)
        #print("1Train. ori: ", ori)
        
        out = model_D(img, step, ori)
        
        #l1 loss
        loss_l1 = criterion_l1(out, gt)
        loss_record_train_D["train_L1Loss_D"] += loss_l1.item()

        if psnr_rec > args.warmup_thr:
            #perceptual loss
            if args.PerceptualLoss:
                loss_percept = criterion_per(out, gt)
                loss_record_train_D["train_PerceptLoss_D"] += loss_percept.item()
                #loss_train += args.Percept_eff * loss_percept

            #Cobi loss
            if args.CobiLoss:
                loss_cobi = criterion_cobi(out, gt)
                loss_record_train_D["train_CobiLoss_D"] += loss_cobi.item()
                #loss_train += args.Cobi_eff * loss_cobi

            #Naive float loss (直接/2)
            if args.FloatLoss:
                out_temp = model(img, step/2, ori)
                out_float = model(out_temp, step/2, step/2)
                loss_float = criterion_l1(out_float, gt)
                loss_record_train_D["train_FloatLoss_D"] += loss_float.item()
                #loss_train += args.Float_eff * loss_float

        loss_train = loss_l1 + args.Float_eff * loss_float + args.Percept_eff * loss_percept + args.Cobi_eff * loss_cobi
        #loss_train.backward(retain_graph=True)
        loss_train.backward()
        optimizer_D.step()

        loss_record_train_D["train_loss_D"] += loss_train.item()

        if args.LR_CosAnneal:
            scheduler_D.step(epoch + i / iters)


        ############################
        # (1-2)cyclic loss update Model_B
        ###########################
        optimizer_B.zero_grad()
        out_t = out.detach()
        #Cyclic
        recover = model_B(out_t, -1*step ,step)
        loss_cyclic = criterion_l1(img, recover)
        loss_cyclic = loss_cyclic * 0.1
        loss_cyclic.backward()
        optimizer_B.step()

        loss_record_train_B["train_CyclicLoss_B"] += loss_cyclic.item()

        if args.LR_CosAnneal:
            scheduler_B.step(epoch + i / iters)

    ############################
    # (2) B predict, D recover
    ###########################
    for i, data in enumerate(train_loader_inc):
        loss_train = 0 # total training loss
        loss_l1 = 0
        loss_percept = 0
        loss_cobi = 0
        loss_float = 0
        loss_cyclic = 0

        ############################
        # (2-1)L1 loss update Model_B
        ###########################
        optimizer_B.zero_grad()
        
        #EX: img: EV0, gt: EV2, step:2, ori:0 
        img, gt, step,  ori = data
        img = img.to(device)
        gt = gt.to(device)
        step = step.to(device)
        ori = ori.to(device)

        #print("2Train. step: ", step)
        #print("2Train. ori: ", ori)
        

        out = model_B(img, step, ori)
        
        #l1 loss
        loss_l1 = criterion_l1(out, gt)
        loss_record_train_B["train_L1Loss_B"] += loss_l1.item()

        if psnr_rec > args.warmup_thr:
            #perceptual loss
            if args.PerceptualLoss:
                loss_percept = criterion_per(out, gt)
                loss_record_train_B["train_PerceptLoss_B"] += loss_percept.item()
                #loss_train += args.Percept_eff * loss_percept

            #Cobi loss
            if args.CobiLoss:
                loss_cobi = criterion_cobi(out, gt)
                loss_record_train_B["train_CobiLoss_B"] += loss_cobi.item()
                #loss_train += args.Cobi_eff * loss_cobi

            #Naive float loss (直接/2)
            if args.FloatLoss:
                out_temp = model(img, step/2, ori)
                out_float = model(out_temp, step/2, step/2)
                loss_float = criterion_l1(out_float, gt)
                loss_record_train_B["train_FloatLoss_B"] += loss_float.item()
                #loss_train += args.Float_eff * loss_float

        loss_train = loss_l1 + args.Float_eff * loss_float + args.Percept_eff * loss_percept + args.Cobi_eff * loss_cobi
        #loss_train.backward(retain_graph=True)
        loss_train.backward()
        optimizer_B.step()

        loss_record_train_B["train_loss_B"] += loss_train.item()

        if args.LR_CosAnneal:
            scheduler_B.step(epoch + i / iters)


        ############################
        # (2-2)cyclic loss update Model_D
        ###########################
        optimizer_D.zero_grad()
        out_t = out.detach()
        #Cyclic
        recover = model_D(out_t, -1*step ,step)
        loss_cyclic = criterion_l1(img, recover)
        loss_cyclic = loss_cyclic * 0.1
        loss_cyclic.backward()
        optimizer_D.step()

        loss_record_train_D["train_CyclicLoss_D"] += loss_cyclic.item()

        if args.LR_CosAnneal:
            scheduler_D.step(epoch + i / iters)

    with torch.no_grad():
        ############################
        # EVAL on Model_D
        ###########################        
        loss_test_D = 0 # total testing loss
        model_D.eval()
        for data in test_loader_dec:
            img, gt, step, ori = data
            img = img.to(device)
            gt = gt.to(device)
            step = step.to(device)
            ori = ori.to(device)

            #print("EVAL_D. step: ", step)
            #print("EVAL_D. ori: ", ori)
            

            out = model_D(img, step, ori)

            #l1 loss
            loss_l1 = criterion_l1(out, gt)
            loss_test_D += loss_l1

            if psnr_rec > args.warmup_thr:
                #perceptual loss
                if args.PerceptualLoss:
                    loss_percept = criterion_per(out, gt)
                    loss_test_D += args.Percept_eff * loss_percept

                #Cobi loss
                if args.CobiLoss:
                    loss_cobi = criterion_cobi(out, gt)
                    loss_test_D += args.Cobi_eff * loss_cobi

                #Naive float loss (直接/2)
                if args.FloatLoss:
                    out_temp = model(img, step/2, ori)
                    out_float = model(out_temp, step/2, step/2)
                    loss_float = criterion_l1(out_float, gt)
                    loss_test_D += args.Float_eff * loss_float

            #Prune out EV0
            nonzero_index = step != 0
            nonzero_index = nonzero_index.squeeze(1)

            nonzero_gt = gt[nonzero_index]
            nonzero_out = out[nonzero_index]

            psnr = avg_psnr(nonzero_out, nonzero_gt, reduce=False)

            
            loss_record_test_D["test_loss_D"] += loss_test_D.item()
            loss_record_test_D["test_psnr_D"] += psnr

        ############################
        # EVAL on Model_B
        ###########################        
        loss_test_B = 0 # total testing loss
        model_B.eval()
        for data in test_loader_inc:
            img, gt, step, ori = data
            img = img.to(device)
            gt = gt.to(device)
            step = step.to(device)
            ori = ori.to(device)

            #print("EVAL_B. step: ", step)
            #print("EVAL_B. ori: ", ori)
            

            out = model_B(img, step, ori)

            #l1 loss
            loss_l1 = criterion_l1(out, gt)
            loss_test_B += loss_l1

            if psnr_rec > args.warmup_thr:
                #perceptual loss
                if args.PerceptualLoss:
                    loss_percept = criterion_per(out, gt)
                    loss_test_B += args.Percept_eff * loss_percept

                #Cobi loss
                if args.CobiLoss:
                    loss_cobi = criterion_cobi(out, gt)
                    loss_test_B += args.Cobi_eff * loss_cobi

                #Naive float loss (直接/2)
                if args.FloatLoss:
                    out_temp = model(img, step/2, ori)
                    out_float = model(out_temp, step/2, step/2)
                    loss_float = criterion_l1(out_float, gt)
                    loss_test_B += args.Float_eff * loss_float

            
            #Prune out EV0
            nonzero_index = step != 0
            nonzero_index = nonzero_index.squeeze(1)

            nonzero_gt = gt[nonzero_index]
            nonzero_out = out[nonzero_index]

            psnr = avg_psnr(nonzero_out, nonzero_gt, reduce=False)

            
            loss_record_test_B["test_loss_B"] += loss_test_B.item()
            loss_record_test_B["test_psnr_B"] += psnr


    # Average accumulated record - Model_D 
    loss_test_D /= len(testset_dec)
    for key in loss_record_train_D:
        loss_record_train_D[key] /= len(trainset_dec)   
    for key in loss_record_test_D:
        loss_record_test_D[key] /= (len(testset_dec) * 3/4) 

    psnr_rec_D = loss_record_test_D["test_psnr_D"]

    cur_lr = optimizer_D.state_dict()['param_groups'][0]['lr']

    if args.LR_Plateau:
        scheduler_D.step(loss_test_D)

    loss_record_train_D["lr_D"] = cur_lr 

    # Average accumulated record - Model_B 
    loss_test_B /= len(testset_inc)
    for key in loss_record_train_B:
        loss_record_train_B[key] /= len(trainset_inc)   
    for key in loss_record_test_B:
        loss_record_test_B[key] /= (len(testset_inc) * 3/4) 

    psnr_rec_B = loss_record_test_B["test_psnr_B"]

    cur_lr = optimizer_B.state_dict()['param_groups'][0]['lr']

    if args.LR_Plateau:
        scheduler_B.step(loss_test_B)

    loss_record_train_B["lr_B"] = cur_lr 

    psnr_rec = (psnr_rec_D + psnr_rec_B)/2


    if args.wandb == True:
        all_record = {**loss_record_train_B,**loss_record_test_B, **loss_record_train_D,**loss_record_test_D}
        wandb.log(all_record) 

    if (epoch + 1) % args.log_epoch == 0:
        name = 'model_' + str(epoch+1) + '.pth'
        torch.save(model_B.state_dict(),  check_path_inc + '/' + name)
        torch.save(model_D.state_dict(),  check_path_dec + '/' + name)

torch.save(model_B.state_dict(), check_path_inc + '/final_model.pth')
torch.save(model_D.state_dict(), check_path_dec + '/final_model.pth')
