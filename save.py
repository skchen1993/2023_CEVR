import torch
import torch.nn as nn
import argparse
import json
import os
from torch.utils.data import DataLoader

from core.Data_provider import build_dataset
from core.HDR_model import build_network
from core.utils import avg_psnr, save_fig, AverageMeter

# TODO YAML
parser = argparse.ArgumentParser()
parser.add_argument('--set_name', type=str, default='half')
parser.add_argument('--img_num', type=int, default=7)
parser.add_argument('--img_set', type=int, default=4)
parser.add_argument('--dataset_mode', type=str, default='increase')
parser.add_argument('--img_height', type=int, default=256)

parser.add_argument('--model_name', type=str, default='img')
parser.add_argument('--decode_name', type=str, default='comb')
parser.add_argument('--mlp_num', type=int, default=3)
parser.add_argument('--pretrain', type=str, default='vgg')

parser.add_argument('--weight_path', type=str, default='runs/inc')
parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--out_root', type=str, default='./result')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Initializing with device:", device)

os.makedirs(args.out_root)

root = args.data_root
train = False
_, dataset = build_dataset(args)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

path = args.weight_path
model = build_network(args)
model.load_state_dict(torch.load(path + '/final_model.pth'))
model.to(device)

stack_psnr = []
num = 4
for _ in range(num):
    stack_psnr.append(AverageMeter())
with torch.no_grad():
    model.eval()
    for ind, data in enumerate(dataloader):
        img, gt, step, ori = data
        img = img.to(device)
        step = step.to(device)
        gt = gt.to(device)
        ori = ori.to(device)

        out = model(img, step, ori)
        psnr = avg_psnr(out, gt)
        
        ev = ind % num
        stack_psnr[ev].update(psnr)

set_psnr = []
for i in range(num):
    set_psnr.append(stack_psnr[i].avg)
    print(stack_psnr[i].avg)
print('avg:',sum(set_psnr)/ num)

if train:
    name = '/train_result.json'
else:
    name = '/test_result.json'

with open(path + name, 'w') as fp:
    json.dump(set_psnr, fp)
