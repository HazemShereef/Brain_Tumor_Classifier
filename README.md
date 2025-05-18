# 🧠 Brain Tumor MRI Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)  
*A deep learning system for classifying brain tumors (Glioma, Meningioma, Pituitary) from MRI scans using PyTorch.*

---

## 📌 Key Features
- **98.5% validation accuracy** on 7,023 MRI scans
- **Dual-model approach**: Custom CNN + EfficientNet-B0 (transfer learning)
- **Clinical-grade preprocessing**: Normalization, augmentation, and DICOM compatibility
- **Optimized for deployment**: <70 mins training time on consumer GPUs

---

## 🏥 Dataset (WIP - Add your dataset source)
| Tumor Type      | Train Samples | Test Samples |
|----------------|--------------|--------------|
| Glioma         | 1,339        | 300          |
| Meningioma     | 1,595        | 306          |
| Pituitary      | 1,457        | 405          |
| Normal         | 1,321        | 300          |

**Preprocessing Pipeline**:
```python
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1845, 0.1845, 0.1846], 
                         std=[0.1992, 0.1992, 0.1992])
])

```
---
## ⚙️ Training Configuration
- Hyperparameter	Value 
- Batch Size	32
- Epochs (CNN/EffNet)	100 / 20
- Initial Learning Rate	1e-5 (dynamic adj.)
- Loss Function	CrossEntropyLoss
- Optimizer	Adam
## 📊 Results
Performance Comparison:
---

- Model	Accuracy	Training Time	Params
- Custom CNN	96.2%	80 mins	1.2M
- EfficientNet	98.5%	70 mins	5.3M
