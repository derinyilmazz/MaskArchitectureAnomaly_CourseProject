# Anomaly Segmentation for Road Scenes

This repository contains the full code for our course project. The project investigates anomaly segmentation in urban 
driving scenes, comparing a pixel-based model (ERFNet) with an encoder-only 
mask transformer (EoMT) across five benchmark datasets.

## What this project does

We evaluate how well segmentation models can detect out-of-distribution (OoD) 
objects, like animals or debris on the road, that were never seen during 
training. We apply four post-hoc anomaly scoring methods (MSP, MaxLogit, Max 
Entropy, and RbA) and extend the work by fine-tuning EoMT with Logit 
Normalization loss to reduce model overconfidence.

## Folders

- [eval](https://github.com/derinyilmazz/MaskArchitectureAnomaly_CourseProject/blob/main/eval) contains tools for evaluating ERFNet and EoMT on anomaly segmentation benchmarks. Includes unified evaluation scripts for both COCO and Cityscapes EoMT checkpoints, as well as the ERFNet baseline eval.
- [trained_models](https://github.com/derinyilmazz/MaskArchitectureAnomaly_CourseProject/blob/main/trained_models) contains the pretrained ERFNet weights for the baseline evaluation.
- [eomt](https://github.com/derinyilmazz/MaskArchitectureAnomaly_CourseProject/blob/main/eomt) is based on the original EoMT repository. Inside you will find code to train EoMT on Cityscapes, pretrained checkpoints, and our LogitNorm loss modification in `training/mask_classification_loss.py`.

## Running Anomaly Evaluation

**ERFNet:**
```bash
cd eval
python evalAnomaly.py \
  --input '../Validation_Dataset/RoadAnomaly21/images/*.png' \
  --method maxlogit
```

**EoMT-COCO:**
```bash
python eval/evalAnomaly_eomt.py \
  --model coco \
  --config eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --input 'Validation_Dataset/RoadAnomaly21/images/*.png' \
  --method rba
```

**EoMT-Cityscapes pretrained:**
```bash
python eval/evalAnomaly_eomt.py \
  --model cityscapes \
  --checkpoint /path/to/cityscapes_pretrained.ckpt \
  --input 'Validation_Dataset/RoadAnomaly21/images/*.png' \
  --method rba
```

**EoMT fine-tuned:**
```bash
python eval/evalAnomaly_eomt.py \
  --model cityscapes \
  --checkpoint /path/to/finetuned.ckpt \
  --input 'Validation_Dataset/RoadAnomaly21/images/*.png' \
  --method rba
```

**EoMT fine-tuned with LogitNorm loss:**
```bash
python eval/evalAnomaly_eomt.py \
  --model cityscapes \
  --checkpoint eomt/checkpoints/lognorm_epoch0.ckpt \
  --input 'Validation_Dataset/RoadAnomaly21/images/*.png' \
  --method rba
```

## Training with LogitNorm

The LogitNorm loss is already integrated in
`eomt/training/mask_classification_loss.py`.
To reproduce the fine-tuning run:

```bash
cd eomt
WANDB_MODE=disabled python main.py fit \
  --config configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --config configs/lognorm_train.yaml \
  --compile_disabled
```

## Team

Derin Yilmaz, Mehmet Basar Eroglu, Mert Akgun, Mert Deniz Eroglu  
Politecnico di Torino
