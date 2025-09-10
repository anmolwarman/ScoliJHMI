## ScoliJHMI

### `classifier.ipynb` contains code for random forest training and heatmap generation. 
### `jhu.npy`, `ucsf.npy` and `washu.npy` contain ODI labels extracted from the csv
### Image emebeddings for each dataset are provided, in the `*aug_resnet_10.npy` files. Code for generating the embeddings is part of `classifier.ipynb` 
### `generate_image_augmentations.py` is used to create the augmeneted dataset, with 10 new images from each image