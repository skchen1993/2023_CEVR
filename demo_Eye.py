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

# TODO YAML
parser = argparse.ArgumentParser()


#model
parser.add_argument('--model_name', type=str, default='affine_pad')
parser.add_argument('--decode_name', type=str, default='mult')
parser.add_argument('--act', type=str, default='leaky_relu')
parser.add_argument('--mlp_num', type=int, default=3)
parser.add_argument('--pretrain', type=str, default='vgg')
parser.add_argument('--cycle', action='store_true', default=False)
parser.add_argument('--sep', type=float, default=0.5)
parser.add_argument('--EV_info',  type=int, default=2, help="1: only cat dif, 2: cat source and dif, 3: Embed DIF to 16 dim vec")
parser.add_argument('--init_weight',  action='store_true', default=False)

# dataset
parser.add_argument('--data_root', type=str, default='/home/skchen/Research_HDR_hunlin/HDREye/images/LDR/')
parser.add_argument('--Float_Stack1', action='store_true', default=False)
parser.add_argument('--Float_Stack2', action='store_true', default=False)
parser.add_argument('--Float_Stack3', action='store_true', default=False)

# exp path
parser.add_argument('--exp_path', type=str, default='./train_strategy/experiment/noaffine_ablation/') # Exp folder
parser.add_argument('--resize', action='store_true', default=False)
parser.add_argument('--epoch', type=str, default='620') # Exp folder

args = parser.parse_args()

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

# Set up dataset info
data_path = args.data_root # '/home/skchen/Research_HDR_hunlin/HDREye/images/LDR/'

scene_path = glob.glob(data_path + "*")
scene_fold = []
for i in scene_path:
    list_ = i.split("/")
    img_name = list_[-1]
    scene_name = img_name.split("_")[0]
    scene_fold.append(scene_name)   # scene_fold = ['C35', "C15",...]

print("scene_fold: ", scene_fold)


exp_fold_int = [-3, -2, -1, 1, 2, 3]
if args.Float_Stack1:
	exp_fold_float = [-3, -2.5 ,-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3]
	exp_fold = exp_fold_float
	print("Generating Floating EV stack1")

elif args.Float_Stack2:
	exp_fold_float = [-3, -2, -1.5, -1.25, -1, -0.5, 0.5, 1, 1.25, 1.5, 2, 3]
	exp_fold = exp_fold_float
	print("Generating Floating EV stack2")

elif args.Float_Stack3:
	exp_fold_float = [-3, -2.5, -2, -1.5, -1.25, -1, -0.5, 0.5, 1, 1.25, 1.5, 2, 2.5, 3]
	exp_fold = exp_fold_float
	print("Generating Floating EV stack3")

else:
	exp_fold = exp_fold_int
	print("Generating Integer EV stack")




print("Dataset info preparation!!")

# Build up output image folder
save_path = args.exp_path + "exp_result_HDREye_" + "epoch" + args.epoch + '/'
if path.exists(save_path) == False:
    print("makedir: ", save_path )
    os.makedirs(save_path)
else:
    print("exp_result folder: ", save_path , " existed!")

#Record PSNR
"""
ev_dict = {}
for ev in exp_fold_int:
	ev_dict[str(ev)] = AverageMeter()
"""

# Build up inc/dec model and load weight
if args.cycle:
    from core.cycle_model import build_network
    print('cycle model')
    model = build_network(args)
else:
    from core.HDR_model import build_network
    print('normal model')
    model_inc = build_network(args)
    model_dec = build_network(args)

"""
if args.best == False:
	model_inc.load_state_dict(torch.load(args.exp_path + 'inc/final_model.pth'))
	model_inc.to(device)
	model_dec.load_state_dict(torch.load(args.exp_path + 'dec/final_model.pth'))
	model_dec.to(device)
	print("Final Model build up and load weight successfully!!")
else:
	model_inc.load_state_dict(torch.load(args.exp_path + 'inc/model_best.pth'))
	model_inc.to(device)
	model_dec.load_state_dict(torch.load(args.exp_path + 'dec/model_best.pth'))
	model_dec.to(device)
	print("Best Model build up and load weight successfully!!")
"""
weight_name = 'model_' + args.epoch + '.pth'
model_inc.load_state_dict(torch.load(args.exp_path + 'inc/' + weight_name))
model_inc.to(device)
model_dec.load_state_dict(torch.load(args.exp_path + 'dec/' + weight_name))
model_dec.to(device)
print("Model build up and load weight successfully!!", " Weight name: ", weight_name)




# inference
with torch.no_grad():
	model_inc.eval()
	model_dec.eval()

	for scene in scene_fold:

		print("Processing Scene: ", scene)
		# build up scene folder in exp_result
		scene_path = save_path + scene     #'./train_strategy/experiment/milestone2-1/exp_result_HDREye/C35'
		if path.exists(scene_path) == False:
			print("makedir: ", scene_path)
			os.makedirs(scene_path)

		# Get source image
		#EV_zero_img_path = data_path + scene+ "/" + scene+ "_0EV_true.jpg.png"
		EV_zero_img_path = data_path + scene + "_LDR.tif"

		#pdb.set_trace()


		EV_zero_img = transform(Image.open(EV_zero_img_path).convert('RGB')).unsqueeze(0).to(device)

		for tar_exp in exp_fold:
			#print("tar_exp= ", tar_exp)

			# Get ground truth image
			"""
			if tar_exp in exp_fold_int:			
				gt_path = data_path + scene+ "/" + scene + "_" + str(tar_exp) + "EV_true.jpg.png"
				gt = transform(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)
			"""

			step = torch.tensor([0 + tar_exp], dtype=torch.float32).unsqueeze(0).to(device)
			ori = torch.tensor([0], dtype=torch.float32).unsqueeze(0).to(device)

			if tar_exp > 0:
				out = model_inc(EV_zero_img, step, ori)
				#print("inc act")
			if tar_exp < 0:
				out = model_dec(EV_zero_img, step, ori)
				#print("dec act")

			"""
			if tar_exp in exp_fold_int:
				psnr = avg_psnr(out, gt)
				print("Scene ", scene, ", EV ", tar_exp, " PSNR:",psnr)
				ev_dict[str(tar_exp)].update(psnr)
			"""

			out = out.squeeze(0).cpu() # From (bs,c,h,w) back to (c,h,w)
			
			if args.resize:
				output_path = scene_path + "/EV" + str(tar_exp) + ".png"
			else:
				output_path = scene_path + "/EV" + str(tar_exp) + "_ori.png"
			save_img = save_fig(out, output_path)
			#pdb.set_trace()

		if args.resize:
			out_zero_path =  scene_path + "/EV0.png"
		else:
			output_path = scene_path + "/EV" + str(tar_exp) + "_ori.png"
		zero_img = EV_zero_img.squeeze(0).cpu()
		save_img = save_fig(zero_img, out_zero_path)



"""
# Reuslt (avg PSNR for each EV)
for ev in exp_fold_int:
	print("EV ", ev, " avg PSNR: ", ev_dict[str(ev)].avg)
"""	

