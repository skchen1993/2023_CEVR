# CEVR 
## Dataset
[Deep chain HDRI github](https://siyeong-lee.github.io/hdr_vds_dataset/)

## Training 
- cd to training_strategy folder
- Using following code to do the bright model training:
``` 
CUDA_VISIBLE_DEVICES=3 python -u train_warmupLoss.py --dataset_mode=increase  --check_path=inc --name=Standard_LNnoaffine_Maps_BmodelAug --model_name=affine2_pad_LNnoaffine_map --batch_size=4 --decode_name=mult_resizeUp_map --LR_CosAnneal --EV_info=1 --num_epoch=1250 --FloatLoss --Float_eff=0.1 --cache --wandb --augment 
```
- Using following code to do the dark model training:
```
CUDA_VISIBLE_DEVICES=9 python -u train_warmupLoss.py --dataset_mode=decrease  --check_path=dec --name=Standard_LNnoaffine_Maps_Dmodel --model_name=affine2_pad_LNnoaffine_map --batch_size=4 --decode_name=mult_resizeUp_map --LR_CosAnneal --EV_info=1 --num_epoch=1250  --FloatLoss --Float_eff=0.1 --cache --wandb
```
