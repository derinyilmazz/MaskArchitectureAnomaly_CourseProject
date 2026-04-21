import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from erfnet import ERFNet
from methods import msp_anomaly_score, maxlogit_anomaly_score, entropy_anomaly_score

NUM_CLASSES = 20

IMAGE_PATH = "../Validation_Dataset/RoadAnomaly21/images/0.png"
MASK_PATH = "../Validation_Dataset/RoadAnomaly21/labels_masks/0.png"
WEIGHTS_PATH = "../trained_models/erfnet_pretrained.pth"

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("Loading ERFNet...")
model = ERFNet(NUM_CLASSES).to(device)
model_state = torch.load(WEIGHTS_PATH, map_location=device)

new_state = {}
for k, v in model_state.items():
    if k.startswith('module.'):
        new_state[k[7:]] = v
    else:
        new_state[k] = v

model.load_state_dict(new_state, strict=False)
model.eval()

print(f"Processing Image: {IMAGE_PATH}")
original_img = Image.open(IMAGE_PATH).convert('RGB')
input_transform = Compose([Resize((512, 1024), Image.BILINEAR), ToTensor()])
tensor_img = input_transform(original_img).unsqueeze(0).float().to(device)

gt_mask = Image.open(MASK_PATH).resize((1024, 512), Image.NEAREST)

with torch.no_grad():
    result = model(tensor_img)
logits_np = result.squeeze(0).data.cpu().numpy()

print("Calculating Heatmaps...")
msp_score = msp_anomaly_score(logits_np)
maxlogit_score = maxlogit_anomaly_score(logits_np)
entropy_score = entropy_anomaly_score(logits_np)

plt.figure(figsize=(20, 5))

plt.subplot(1, 5, 1)
plt.title("Original Image")
plt.imshow(original_img.resize((1024, 512)))
plt.axis('off')

plt.subplot(1, 5, 2)
plt.title("Ground Truth Mask")
plt.imshow(gt_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.title("MSP Score")
plt.imshow(msp_score, cmap='magma')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.title("MaxLogit Score")
plt.imshow(maxlogit_score, cmap='magma')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.title("Entropy Score")
plt.imshow(entropy_score, cmap='magma')
plt.axis('off')

save_name = "comparison_plot.png"
plt.tight_layout()
plt.savefig(save_name, dpi=300)
print(f"Success! Image saved as {save_name} in the eval folder.")