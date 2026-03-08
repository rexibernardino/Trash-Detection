Smart Trash Recognition System

Overview
Waste management is a critical global challenge. This project aims to assist in sorting waste by classifying images into 6 categories: **cardboard, glass, metal, paper, plastic, and trash**.

ech Stack
- **Deep Learning Framework**: PyTorch
- **Architecture**: ResNet50 (Transfer Learning)
- **Data Source**: TrashNet Dataset
- **Tools**: Kaggle Kernels, Python

Methodology & Challenges
This project wasn't just about training a model; it involved addressing common Computer Vision challenges:
1. **Shortcut Learning (Bias)**: Initially, the model learned to associate background colors with waste types. I addressed this using aggressive **Data Augmentation** (RandomErasing, Grayscale, ColorJitter) to force the model to focus on object shapes.
2. **Architecture Upgrade**: Migrated from ResNet18 to **ResNet50** to capture finer textural details, significantly improving accuracy on visually similar classes like paper vs. cardboard.
3. **Fine-Tuning**: Performed full parameter fine-tuning with a low learning rate to adapt the pre-trained weights to specific waste textures.

Test Performance Example

<img width="456" height="464" alt="image" src="https://github.com/user-attachments/assets/5f0ccd15-ee7b-4da5-b7b2-bad5bf01a71c" />
