# 🧠 Brain Tumor Classification from MRI Images

This project is a deep learning model for classifying brain MRI images into one of four categories using PyTorch and transfer learning with ResNet18.

## 📊 Dataset

- **Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Total Images**: 7023 human brain MRI scans
- **Categories**:
  - `Glioma`
  - `Meningioma`
  - `Pituitary`
  - `No Tumor` (normal scans without tumors)
- **Split**:
  - 5712 images used for training/validation
  - Remaining images for testing
  - Training/Validation ratio: **9:1**

## 🔄 Data Augmentation

Applied using `torchvision.transforms` to improve generalization:

- `Resize((224, 224))`
- `RandomHorizontalFlip()`
- `RandomRotation(10)`
- `ToTensor()`
- `Normalize(mean=[0.1845, 0.1845, 0.1846], std=[0.1992, 0.1992, 0.1992])`
