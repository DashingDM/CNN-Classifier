# CNN Image Classification - MNIST & CIFAR-10

A PyTorch-based Convolutional Neural Network (CNN) implementation for image classification on MNIST and CIFAR-10 datasets. This project demonstrates deep learning fundamentals including model architecture design, training, testing, and filter visualization.

## Project Overview

This project implements a custom CNN architecture capable of classifying:
- **MNIST**: Handwritten digits (0-9)
- **CIFAR-10**: 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

The model achieves **99.39% accuracy on MNIST** and **78.47% accuracy on CIFAR-10** after 15 epochs of training.

##  Model Architecture
```
Input (32x32x3)
    ↓
Conv1 (32 filters, 5x5) → BatchNorm → ReLU → MaxPool (2x2)
    ↓
Conv2 (64 filters, 3x3) → BatchNorm → ReLU → MaxPool (2x2)
    ↓
Conv3 (128 filters, 3x3) → BatchNorm → ReLU → MaxPool (2x2)
    ↓
Flatten → FC1 (256 units) → ReLU → Dropout (0.5)
    ↓
FC2 (10 units) → Output
```

### Key Features:
- **3 Convolutional layers** with increasing filter depth (32 → 64 → 128)
- **Batch Normalization** for stable training
- **Dropout (0.5)** to prevent overfitting
- **MaxPooling** for spatial dimensionality reduction
- **~400K trainable parameters**

##  Getting Started

### Prerequisites
```bash
Python 3.7+
PyTorch 1.9+
torchvision
matplotlib
opencv-python
numpy
```



##  Usage

### Training

Train the model on MNIST:
```bash
python CNNclassify.py train --mnist
```

Train the model on CIFAR-10:
```bash
python CNNclassify.py train --cifar
```

Both models will be saved in the `model/` directory as:
- `cnn_mnist.pth`
- `cnn_cifar.pth`

### Testing

Classify an image using both trained models:
```bash
python CNNclassify.py test path/to/your/image.png
```

**Example Output:**
```
Prediction using MNIST model → 8
Prediction using CIFAR model → horse
```

The script will also generate:
- `CONV_filters_mnist.png` - Visualization of Conv1 learned filters
- `CONV_filters_cifar.png` - Visualization of Conv1 learned filters
- `CONV_rslt_mnist.png` - Feature maps after Conv1 layer
- `CONV_rslt_cifar.png` - Feature maps after Conv1 layer

##  Performance Results

### MNIST Results
| Metric | Training | Testing |
|--------|----------|---------|
| **Accuracy** | 99.55% | 99.39% |
| **Loss** | 0.0137 | 0.0237 |

### CIFAR-10 Results
| Metric | Training | Testing |
|--------|----------|---------|
| **Accuracy** | 85.93% | 78.47% |
| **Loss** | 0.3935 | 0.6764 |

### Training Progress

**MNIST** (15 epochs):
```
Epoch 01: Train Loss=0.1550, Train Acc=95.21%, Test Acc=98.72%
Epoch 15: Train Loss=0.0137, Train Acc=99.55%, Test Acc=99.39%
```

**CIFAR-10** (15 epochs):
```
Epoch 01: Train Loss=1.4373, Train Acc=47.61%, Test Acc=59.56%
Epoch 15: Train Loss=0.3935, Train Acc=85.93%, Test Acc=78.47%
```

##  Visualization Features

The code automatically generates visualizations:

1. **Convolutional Filters** (`CONV_filters_*.png`)
   - Displays the 32 learned 5x5 filters from the first convolutional layer
   - Shows what patterns the network detects (edges, textures, shapes)

2. **Feature Maps** (`CONV_rslt_*.png`)
   - Shows the 32 activation maps after Conv1 for your test image
   - Demonstrates which filters activate for different image features

##  Project Structure
```
cnn-classification/
├── CNNclassify.py          # Main script for training and testing
├── model/                  # Saved model weights
│   ├── cnn_mnist.pth
│   └── cnn_cifar.pth
├── data/                   # Auto-downloaded datasets (MNIST, CIFAR-10)
├── CONV_filters_mnist.png  # Filter visualizations
├── CONV_filters_cifar.png
├── CONV_rslt_mnist.png     # Feature map visualizations
└── CONV_rslt_cifar.png
```

##  Technical Details

### Data Preprocessing
- **MNIST**: Resized to 32x32, converted to 3-channel grayscale, normalized to [-1, 1]
- **CIFAR-10**: Already 32x32 RGB, normalized to [-1, 1]

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 128 (training), 100 (testing)
- **Epochs**: 15
- **Device**: CUDA (GPU) if available, else CPU

### Model Details
```python
Total Parameters: ~400,000
- Convolutional layers: ~150,000
- Fully connected layers: ~250,000
```

##  Learning Objectives

This project demonstrates:
- Building CNNs from scratch using PyTorch
- Image classification on standard benchmarks
- Batch normalization and dropout for regularization
- Model training, evaluation, and hyperparameter tuning
- Visualizing learned convolutional filters
- Transfer of model architecture across different datasets

##  Potential Improvements

- [ ] Data augmentation (rotation, flipping, cropping)
- [ ] Learning rate scheduling
- [ ] More epochs for CIFAR-10 training
- [ ] Deeper architectures (ResNet, VGG-style)
- [ ] Cross-validation
- [ ] Ensemble methods





