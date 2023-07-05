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
import numpy as np

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


import pdb

# TODO YAML
parser = argparse.ArgumentParser()


#model
parser.add_argument('--model_name', type=str, default='CEVR_NormNoAffine_Maps')
parser.add_argument('--decode_name', type=str, default='mult_resizeUp_map')
parser.add_argument('--act', type=str, default='leaky_relu')
parser.add_argument('--mlp_num', type=int, default=3)
parser.add_argument('--pretrain', type=str, default='vgg')
parser.add_argument('--cycle', action='store_true', default=False)
parser.add_argument('--sep', type=float, default=0.5)
parser.add_argument('--EV_info',  type=int, default=1, help="1: only cat dif, 2: cat source and dif, 3: Embed DIF to 16 dim vec")
parser.add_argument('--init_weight',  action='store_true', default=False)
parser.add_argument('--norm_type', type=str, default='GroupNorm', help="LayerNorm, GroupNorm, InstanceNorm") 
parser.add_argument('--NormAffine', action='store_true', default=False)

# dataset
parser.add_argument('--data_root', type=str, default='/home/skchen/ML_practice/LIIF_on_HDR/VDS_dataset/')
#parser.add_argument('--process_scene', type=str, default='t60')
parser.add_argument('--source_EV', type=int, default=0)
parser.add_argument('--target_EV', type=int, default=1)

# exp path
#parser.add_argument('--exp_path', type=str, default='./train_strategy/experiment/Standard_noLNAffine_Whole/') # Exp folder
parser.add_argument('--B_model_path', type=str, default='CEVR_NormNoAffine_Maps_GN_Bmodel/') # Exp folder
parser.add_argument('--D_model_path', type=str, default='CEVR_NormNoAffine_Maps_GN_Dmodel/') # Exp folder

parser.add_argument('--resize', action='store_true', default=True)
parser.add_argument('--epoch', type=str, default='300') # Exp folder

args = parser.parse_args()


scene_fold = ['t60', 't68',  't82']


exp_base = "./train_strategy/experiment/"
D_path = exp_base + args.D_model_path
B_path = exp_base + args.B_model_path

if args.resize:
	print("!!!!!!!!!!inference on 256*256")
	transform = transforms.Compose([
	    transforms.Resize((256, 256)),
	    transforms.ToTensor()
	])
else:
	print("!!!!!!!!!!inference on original size")
	transform = transforms.Compose([
	    transforms.ToTensor()
	])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Initializing with device:", device)

#------------ If avaliable, fetch image from exp folder, if not, fetch EV0 image into exp folder -------------------
if args.resize:
	save_path = D_path + "exp_result_recursive_" + "epoch" + args.epoch + '/'
else:
	save_path = D_path + "exp_result_recursive_" + "epoch" + args.epoch + '_OriginalSize/'

for process_scene in scene_fold:
	if path.exists(save_path) == False:
		# build fold
		print("makedir: ", save_path)
		os.makedirs(save_path)
		os.makedirs(save_path + process_scene)

		# save EV0 image into exp folder
		out_zero_path = save_path + process_scene + "/EV0.png"
		img_EV0_path = args.data_root + "test_set/"+  process_scene + "/" + process_scene + "_0EV_true.jpg.png"
		EV_zero_img = transform(Image.open(img_EV0_path).convert('RGB')).unsqueeze(0).to(device)
		
		zero_img = EV_zero_img.squeeze(0).cpu()
		save_img = save_fig(zero_img, out_zero_path)

# Build up inc/dec model and load weight
from core.HDR_model import build_network
model_inc = build_network(args)
model_dec = build_network(args)

weight_name = 'model_' + args.epoch + '.pth'
model_inc.load_state_dict(torch.load(B_path + 'inc/' + weight_name))
model_inc.to(device)
model_dec.load_state_dict(torch.load(D_path + 'dec/' + weight_name))
model_dec.to(device)
print("Model build up and load weight successfully!!", " Weight name: ", weight_name)



for process_scene in scene_fold:
	# inference
	with torch.no_grad():
		model_inc.eval()
		model_dec.eval()

		print("Processing ", process_scene, " EV", str(args.source_EV), " to EV", str(args.target_EV))

		# Get source image
		source_img_path = save_path + process_scene + "/EV" + str(args.source_EV) + ".png" # './train_strategy/experiment/CEVR_NormNoAffine_Maps_GN_Dmodel/exp_result_recursive_epochbest/t60/EV0.png'
		source_img = transform(Image.open(source_img_path).convert('RGB')).unsqueeze(0).to(device)

		# Processing source image into target EV
		EV_step = args.target_EV - args.source_EV
		step = torch.tensor([EV_step], dtype=torch.float32).unsqueeze(0).to(device)
		ori = torch.tensor([args.source_EV], dtype=torch.float32).unsqueeze(0).to(device)
		
		if EV_step > 0:
			out = model_inc(source_img, step, ori)
			#print("inc act")
		if EV_step < 0:
			out = model_dec(source_img, step, ori)
			#print("dec act")

		out = out.squeeze(0).cpu() # From (bs,c,h,w) back to (c,h,w)
		
		output_path = save_path + process_scene + "/EV" + str(args.target_EV) + ".png"
		
		save_img = save_fig(out, output_path)




		




