from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


img = Image.open('C:/Users/initi/OneDrive/Documents/GitHub/facenet-pytorch/data/test_images/angelina_jolie/1.jpg')  # Replace with the path to your image file
img_cropped = mtcnn(img, save_path="C:/Users/initi/OneDrive/Documents/GitHub/facenet-pytorch/data/test_images/angelina_jolie/1_ cropped.jpg")
# Add batch dimension: [channels, height, width] -> [1, channels, height, width]
img_cropped = img_cropped.unsqueeze(0)
img_cropped = img_cropped.to(device)
# 或者，如果用于 VGGFace2 分类
resnet.classify = True
img_probs = resnet(img_cropped)


img_probs_cpu = img_probs.cpu()
img_probs_cpu = img_probs_cpu.detach().numpy()


plt.ion()
plt.plot(img_probs_cpu[0])
plt.show()
print(img_probs)



