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

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Model Checkpoints

Due to GitHub's file size limits, the EoMT checkpoints must be downloaded separately. Please save all checkpoints into the `eomt/checkpoints/` directory.

1. **Cityscapes Pretrained Baseline**
   * **Download:** [Instructions in eomt/README.md](eomt/README.md)
   * **Save location:** `eomt/checkpoints/pytorch_model.bin`

2. **Standard Fine-tuned (No LogitNorm)**
   * **Download under finetuned_checkpoints:** [Finetuned Checkpoint](https://drive.google.com/drive/folders/1t2saVRc4_F54x_y_ErieNyN653riZXCA?usp=sharing)
   * **Save location:** `eomt/checkpoints/standard_epoch0.ckpt`

3. **Fine-tuned with LogitNorm**
   * **Download under logitnorm_checkpoint:** [LogitNorm Checkpoint](https://drive.google.com/drive/folders/1t2saVRc4_F54x_y_ErieNyN653riZXCA?usp=sharing)
   * **Save location:** `eomt/checkpoints/lognorm_epoch0.ckpt`
  
## Running Anomaly Evaluation

**ERFNet:**
```bash
python3 eval/evalAnomaly.py \
  --input '../Validation_Dataset/RoadAnomaly21/images/*.png' \
  --method maxlogit
```

**EoMT-COCO:**
```bash
python3 eval/evalAnomaly_eomt.py \
  --model coco \
  --config eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --input 'Validation_Dataset/RoadAnomaly21/images/*.png' \
  --method rba
```

**EoMT-Cityscapes pretrained:**
```bash
python3 eval/evalAnomaly_eomt.py \
  --model cityscapes \
  --checkpoint eomt/checkpoints/pytorch_model.bin \
  --input 'Validation_Dataset/RoadAnomaly21/images/*.png' \
  --method rba
```

**EoMT fine-tuned:**
```bash
python3 eval/evalAnomaly_eomt.py \
  --model cityscapes \
  --checkpoint eomt/checkpoints/standard_epoch0.ckpt \
  --input 'Validation_Dataset/RoadAnomaly21/images/*.png' \
  --method rba
```

**EoMT fine-tuned with LogitNorm loss:**
```bash
python3 eval/evalAnomaly_eomt.py \
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
WANDB_MODE=disabled python3 main.py fit \
  --config configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --config configs/lognorm_train.yaml \
  --compile_disabled
```

## Team

Derin Yilmaz, Mehmet Basar Eroglu, Mert Akgun, Mert Deniz Eroglu  
Politecnico di Torino
