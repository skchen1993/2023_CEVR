import torch
import torch.nn as nn
import argparse
import json
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from core.Data_provider import build_dataset
from core.utils import avg_psnr, save_fig, AverageMeter
import glob
import os
import os.path
from os import path
from PIL import Image

import pdb

# VDS our 的EV是有小數點 0.0, VDS原本的沒有, 有空再來統一


# TODO YAML
parser = argparse.ArgumentParser()


#model
parser.add_argument('--design', type=str, default='adj3')
parser.add_argument('--model_name', type=str, default='affine_pad')
parser.add_argument('--decode_name', type=str, default='mult')
parser.add_argument('--act', type=str, default='leaky_relu')
parser.add_argument('--mlp_num', type=int, default=3)
parser.add_argument('--pretrain', type=str, default='vgg')
parser.add_argument('--cycle', action='store_true', default=False)
parser.add_argument('--sep', type=float, default=0.5)

# dataset
parser.add_argument('--data_root', type=str, default='/home/skchen/ML_practice/LIIF_on_HDR/VDS_dataset/')
parser.add_argument('--VDS_our', action='store_true', default=False)

# exp path
parser.add_argument('--exp_path', type=str, default='./train_strategy/experiment/Adjust3/') # Exp folder
parser.add_argument('--resize', type=int, default=256) # Exp folder

args = parser.parse_args()

size = (args.resize, args.resize)
transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Initializing with device:", device)

# Set up dataset info
if args.VDS_our:
	data_path = "/home/skchen/Research_HDR_hunlin/VDS_data_Our/test_set_F/"
else:
	data_path = args.data_root + "test/set/"

scene_path = glob.glob(data_path + "*")
scene_fold = []
for i in scene_path:
    list_ = i.split("/")
    scene_fold.append(list_[-1])   # scene_fold = ['t03', "t33",...]

print("scene_fold: ", scene_fold)
if args.VDS_our:
	exp_fold = [-3.0, -2.5 ,-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
	print("Using VDS our, exp_fold = ", exp_fold)
else:
	exp_fold = [-3, -2, -1, 1, 2, 3]
	print("Using VDS original, exp_fold = ", exp_fold)

print("Dataset info preparation!!")

# Build up output image folder
save_path = args.exp_path + "exp_result/"
if path.exists(save_path) == False:
    print("makedir: ", save_path )
    os.makedirs(save_path)
else:
    print("exp_result folder: ", save_path , " existed!")



#Record PSNR
ev_dict = {}
for ev in exp_fold:
	ev_dict[str(ev)] = AverageMeter()

# Build up inc/dec model and load weight
if args.cycle:
    from core.cycle_model import build_network
    print('cycle model')
    model = build_network(args)
else:
    from core.HDR_model import build_network
    print('normal model')

    if args.design == "adj3":
    	args.act = 'prelu'
    	model_inc = build_network(args)
    	args.act = 'mprelu'
    	model_dec = build_network(args)

    else:
	    model_inc = build_network(args)
	    model_dec = build_network(args)

model_inc.load_state_dict(torch.load(args.exp_path + 'inc/final_model.pth'))
model_inc.to(device)
model_dec.load_state_dict(torch.load(args.exp_path + 'dec/final_model.pth'))
model_dec.to(device)
print("Model build up and load weight successfully!!")


# inference
with torch.no_grad():
	model_inc.eval()
	model_dec.eval()

	for scene in scene_fold:
		print("Processing Scene: ", scene)
		# build up scene folder in exp_result
		scene_path = save_path + scene
		if path.exists(scene_path) == False:
			print("makedir: ", scene_path)
			os.makedirs(scene_path)


		# Get source image
		if args.VDS_our:
			EV_zero_img_path = data_path + scene+ "/" + scene+ "_0.0EV_true.jpg.png"
		else:
			EV_zero_img_path = data_path + scene+ "/" + scene+ "_0EV_true.jpg.png"

		EV_zero_img = transform(Image.open(EV_zero_img_path).convert('RGB')).unsqueeze(0).to(device)

		for tar_exp in exp_fold:
			#print("tar_exp= ", tar_exp)

			# Get ground truth image
			gt_path = data_path + scene+ "/" + scene + "_" + str(tar_exp) + "EV_true.jpg.png"
			gt = transform(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)

			step = torch.tensor([0 + tar_exp], dtype=torch.float32).unsqueeze(0).to(device)
			ori = torch.tensor([0], dtype=torch.float32).unsqueeze(0).to(device)

			if tar_exp > 0:
				out = model_inc(EV_zero_img, step, ori)
				#print("inc act")
			if tar_exp < 0:
				out = model_dec(EV_zero_img, step, ori)
				#print("dec act")

			psnr = avg_psnr(out, gt)
			ev_dict[str(tar_exp)].update(psnr)

			out = out.squeeze(0).cpu() # From (bs,c,h,w) back to (c,h,w)
			
			output_path = scene_path + "/EV" + str(tar_exp) + ".png"
			save_img = save_fig(out, output_path)
			
			#pdb.set_trace()



# Reuslt (avg PSNR for each EV)
for ev in exp_fold:
	print("EV ", ev, " avg PSNR: ", ev_dict[str(ev)].avg)
		

