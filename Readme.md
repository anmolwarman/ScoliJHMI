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

### Dockerized Inference

To build the inference container image, run the following command from the repository root:

```bash
docker build -t scoli-odi-infer .
```

This produces a Docker image that encapsulates the preprocessing steps, trained model weights, and scoring logic necessary to generate Optical Density Index (ODI) predictions from individual radiographs.

Run the container by mounting a directory that contains the X-ray files and passing the image path you wish to score. For example:

```bash
docker run --rm -v /path/to/xrays:/inputs scoli-odi-infer /inputs/example.jpg
```

The container expects a single-channel or RGB image file in a standard format such as JPEG (`.jpg`, `.jpeg`) or PNG (`.png`). The file is converted to the resolution required by the model before inference. The command prints two values: the predicted probability of the ODI being above the scoliosis threshold, and a corresponding binary label (1 for positive, 0 for negative).

**Limitations:**

- The binary classification output assumes the ODI decision threshold is fixed at 40; adjusting this cutoff requires rebuilding the image with updated logic.
- All inputs must be accessible on the mounted volume and provided as individual file paths. Batch processing or directory globbing is not supported out-of-the-box.
- Images outside of typical clinical radiograph formats (e.g., non-DICOM grayscale exports) may require manual conversion to JPEG or PNG prior to inference.
