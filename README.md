# CEVR 
ICCV 2023 Learning Continuous Exposure Value Representations for Single-Image HDR Reconstruction
[Sy-Kai Chen](chensykai.backup@gmail.com), [Hung-Lin Yen](http://), [Yu-Lun Liu](https://yulunalexliu.github.io/), [Min-Hung Chen](https://minhungchen.netlify.app/), [Hou-Ning Hu](https://eborboihuc.github.io/), [Wen-Hsiao Peng](https://sites.google.com/g2.nctu.edu.tw/wpeng), [Yen-Yu Lin](https://sites.google.com/site/yylinweb/)

Paper is [here](http://)  
Demo website is [here](http://)  
⭐For fast evaluation, you download the data and script [here](http://)  ⭐
***
![CEVR](https://github.com/skchen1993/2023_CEVR/blob/main/img/teaset_png.PNG "CEVR")
> Deep learning is commonly used to reconstruct HDR images from LDR images. LDR stack-based methods are used for single-image HDR reconstruction, generating an HDR image from a deep learning-generated LDR stack. However, current methods generate the stack with predetermined exposure values (EVs), which may limit the quality of HDR reconstruction. To address this, we propose the continuous exposure value representation (CEVR), which uses an implicit function to generate LDR images with arbitrary EVs, including those unseen during training. Our approach generates a continuous stack with more images containing diverse EVs, significantly improving HDR reconstruction. We use a cycle training strategy to supervise the model in generating continuous EV LDR images without corresponding ground truths. Our CEVR model outperforms existing methods, as demonstrated by experimental results.

***
## Fast Evaluation
The ZIP file contain "results of each related work" and "the matlab script for evaluation". You can simply download the ZIP file and run the script to get the PSNR, HDR-VDP-2 score for tonemapped image and HDR file. (We evaluate on two datasets, VDS dataset and HDREye dataset, which are also in the ZIP file.)  
⭐[FILE_LINK is here](https://drive.google.com/file/d/1xeCT3APYkTnxeotb_t0wxdSPzBLbnU_p/view) ⭐

## Environment
- torch: 1.12.1  
- CUDA compilation tools: 10.1   
- python 3.8.16  

## Dataset
### VDS and HDREye Dataset 
[Deep chain HDRI github](https://siyeong-lee.github.io/hdr_vds_dataset/)
Download the dataset and modify the `--data_root` (path to the VDS dataset)
Note: Both VDS and HDREye dataset can be found in the [Fast evaluation file](https://drive.google.com/file/d/1xeCT3APYkTnxeotb_t0wxdSPzBLbnU_p/view?usp=sharing)

## Inference
Run the inference code to get the LDR images with different EV!
### VDS dataset:
Whole dataset and whole LDR stack inference:
```
CUDA_VISIBLE_DEVICES=7 python -u demo_VDS.py  --model_name=CEVR_NormNoAffine_Maps  --decode_name=mult_resizeUp_map --EV_info=1 --Float_Stack1 --resize  --epoch=best --norm_type=GroupNorm --D_model_path=CEVR_NormNoAffine_Maps_GN_Dmodel/ --B_model_path=CEVR_NormNoAffine_Maps_GN_Bmodel/
```
- data_root: Remember to set the path to the VDS dataset (Ex: `/home/skchen/xxx/xxx/VDS_dataset/`)
- model_name: Main model setting
- decode_name: Decoder setting
- resize: resize Image to 256x256
- Float_Stack1: If you want to generate floating point EV LDR, use this flag. Delete this flag if only the integer EV LDR image is needed.
- D_model_path, B_model_path: Find the model weight for the Bright model (EV up) and Dark model (EV down).

### HDREye dataset
Whole dataset and whole LDR stack inference:
```
CUDA_VISIBLE_DEVICES=7 python -u demo_Eye.py  --model_name=CEVR_NormNoAffine_Maps  --decode_name=mult_resizeUp_map --EV_info=1 --Float_Stack1 --resize  --epoch=best --norm_type=GroupNorm --D_model_path=CEVR_NormNoAffine_Maps_GN_Dmodel/ --B_model_path=CEVR_NormNoAffine_Maps_GN_Bmodel/
```
- data_root: Remember to set the path to the HDREye dataset (Ex: `'/home/skchen/xxx/xxx/HDREye/images/LDR/'`)
Note: HDREye dataset can be found in the [Fast evaluation file](https://drive.google.com/file/d/1xeCT3APYkTnxeotb_t0wxdSPzBLbnU_p/view?usp=sharing)

