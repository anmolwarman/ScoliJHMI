## ScoliJHMI

This repository contains code and data for training and evaluation.

### Contents

- **`classifier.ipynb`**  
  Jupyter notebook with random forest training and heatmap generation.  

- **`jhu.npy`, `ucsf.npy`, `washu.npy`**  
  ODI labels extracted from the CSV files.  

- **`*aug_resnet_10.npy`**  
  Image embeddings for each dataset. Code for generating embeddings is included in `classifier.ipynb`.  

- **`generate_image_augmentations.py`**  
  Script to create the augmented dataset (10 new images per original image).  
