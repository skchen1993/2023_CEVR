import torch
import torch.nn as nn
import argparse
import json
from torch.utils.data import DataLoader

from core.Data_provider import build_dataset
from core.utils import avg_psnr, save_fig, AverageMeter

# TODO YAML
parser = argparse.ArgumentParser()

#dataset
parser.add_argument('--set_name', type=str, default='half')
parser.add_argument('--img_num', type=int, default=7)
parser.add_argument('--img_set', type=int, default=4)
parser.add_argument('--dataset_mode', type=str, default='increase')
parser.add_argument('--img_height', type=int, default=256)

#model
parser.add_argument('--model_name', type=str, default='affine_pad')
parser.add_argument('--decode_name', type=str, default='mult')
parser.add_argument('--act', type=str, default='leaky_relu')
parser.add_argument('--mlp_num', type=int, default=3)
parser.add_argument('--pretrain', type=str, default='vgg')
parser.add_argument('--cycle', action='store_true', default=False)
parser.add_argument('--sep', type=float, default=0.5)

parser.add_argument('--weight_path', type=str, default='runs/leaky+')
parser.add_argument('--data_root', type=str, default='./data')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Initializing with device:", device)

root = args.data_root
train = False
_, dataset = build_dataset(args)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

path = args.weight_path
if args.cycle:
    from core.cycle_model import build_network
    print('cycle model')
    model = build_network(args)
else:
    from core.HDR_model import build_network
    print('normal model')
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

if args.dataset_mode=='increase':
    print('avg:',sum(set_psnr[1:])/ (num-1))
else:
    print('avg:',sum(set_psnr[:-1])/ (num-1))


if train:
    name = '/train_result.json'
else:
    name = '/test_result.json'

with open(path + name, 'w') as fp:
    json.dump(set_psnr, fp)
