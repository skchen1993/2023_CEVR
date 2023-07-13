# CEVR 
## Environment
torch: 1.12.1
CUDA compilation tools: 10.1 
python 3.8.16

## Dataset
[Deep chain HDRI github](https://siyeong-lee.github.io/hdr_vds_dataset/)
Download the dataset and modify the `--data_root` (path to the VDS dataset)

## Training 
- cd to training_strategy folder
- Using following code to do the bright model training:
``` 
CUDA_VISIBLE_DEVICES=4 python -u train_warmupLoss.py --dataset_mode=increase  --check_path=inc --name=CEVR_NormNoAffine_Maps_GN_Bmodel --model_name=CEVR_NormNoAffine_Maps --batch_size=4 --decode_name=mult_resizeUp_map --LR_CosAnneal --EV_info=1 --num_epoch=1250 --FloatLoss --Float_eff=0.1 --norm_type=GroupNorm --cache --wandb --augment 
```
- Using following code to do the dark model training:
```
CUDA_VISIBLE_DEVICES=5 python -u train_warmupLoss.py --dataset_mode=decrease  --check_path=dec --name=CEVR_NormNoAffine_Maps_GN_Dmodel --model_name=CEVR_NormNoAffine_Maps --batch_size=4 --decode_name=mult_resizeUp_map --LR_CosAnneal --EV_info=1 --num_epoch=1250  --FloatLoss --Float_eff=0.1 --norm_type=GroupNorm --cache --wandb
```

## Inference
### VDS dataset:
Whole dataset and whole LDR stack inference:
```
CUDA_VISIBLE_DEVICES=7 python -u demo_VDS.py  --model_name=CEVR_NormNoAffine_Maps  --decode_name=mult_resizeUp_map --EV_info=1 --Float_Stack1 --resize  --epoch=300 --norm_type=GroupNorm --D_model_path=CEVR_NormNoAffine_Maps_GN_Dmodel/ --B_model_path=CEVR_NormNoAffine_Maps_GN_Bmodel/
```
Specific EV changing:
```
CUDA_VISIBLE_DEVICES=0 python demo_recursive.py --source_EV=-2 --target_EV=-3 --resize
CUDA_VISIBLE_DEVICES=0 python demo_recursive_stack.py --source_EV=-3  --resize
```
