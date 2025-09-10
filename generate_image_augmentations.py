import albumentations as A
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#Load the dataset
jhu_data_dir='/mnt/c/Users/swapnil/Downloads/XRAY_Dataset/XRAY_WashU/final_images'

images=[os.path.join(jhu_data_dir,img_folder,img) for img_folder in sorted(os.listdir(jhu_data_dir)) for img in os.listdir(os.path.join(jhu_data_dir,img_folder)) ]

transforms=A.Compose([A.Affine(scale=(0.98, 1.05), translate_percent=(0.05, 0.05), rotate=3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    A.ElasticTransform(p=0.5, alpha=40, sigma=40 * 0.07),#newly added
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(p=0.5)])

save_dir=os.path.join(jhu_data_dir,'augmented_images_resnet_10')
os.makedirs(save_dir,exist_ok=True)

for img_folder in sorted(os.listdir(jhu_data_dir)):
    save_img_folder=os.path.join(save_dir,img_folder)
    os.makedirs(save_img_folder,exist_ok=True)
    for img in os.listdir(os.path.join(jhu_data_dir,img_folder)):
        im=Image.open(os.path.join(jhu_data_dir,img_folder,img)).convert('RGB')
        im.save(os.path.join(save_img_folder,f'{img[:-4]}.jpg'))
        for i in range(10):
            transformed_image=transforms(image=np.array(im))['image']
            Image.fromarray(transformed_image).save(os.path.join(save_img_folder,f'{img[:-4]}_aug{i}.jpg'))
        
        
        
        
        
        