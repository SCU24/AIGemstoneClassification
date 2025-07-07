# Diamond Shape Classification with Deep Learning

This repository presents the implementation and comparative analysis of deep learning models for the automated classification of diamond shapes. The project was developed as part of a Master's dissertation at "Dunărea de Jos" University of Galați, Romania.

## 🧠 Project Overview

The main goal of this project is to develop an automated image classification system capable of identifying the shape of diamonds based on their visual features. This has significant applications in the jewelry industry, where accurate shape classification directly affects both aesthetic value and commercial pricing.

The models used in this study include:
- A custom Convolutional Neural Network (CNN)
- ResNet50 (with transfer learning)
- EfficientNetB0
- Vision Transformer (ViT)

The dataset contains over **48,000 labeled images** grouped into **8 distinct diamond shapes**: *cushion, emerald, heart, marquise, oval, pear, princess, and round*.

## 📊 Results Summary

| Model        | Accuracy | Strengths                         | Weaknesses                        |
|--------------|----------|-----------------------------------|-----------------------------------|
| CNN (custom) | 0.93     | Lightweight and interpretable     | Struggles with certain shapes     |
| ResNet50     | 1.00     | Excellent generalization          | Requires more resources           |
| EfficientNet | 0.13     | Lightweight, efficient architecture | Severe underperformance on task |
| ViT          | 1.00     | Robust to complex patterns        | High training and data demand     |


## 📁 Dataset

- Total images: **48,765**
- Format: RGB images per shape, labeled into directories
- Data split: 80% training, 10% validation, 10% testing
- Shapes: `cushion`, `emerald`, `heart`, `marquise`, `oval`, `pear`, `princess`, `round`

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy / Pandas**
- **Matplotlib / Seaborn**
- **Scikit-learn**
- **Transfer Learning (ImageNet weights)**

## 🔄 Workflow

1. **Data Preprocessing**:
   - Image resizing (224x224 for CNN/ResNet/EfficientNet, 384x384 for ViT)
   - Normalization and augmentation (rotation, zoom, flip)

2. **Model Training**:
   - All models trained with categorical cross-entropy and Adam optimizer
   - Fine-tuning for ResNet and EfficientNet
   - Transfer learning for pretrained models

3. **Evaluation**:
   - Accuracy, precision, recall, F1-score, confusion matrix
   - Visualizations of training/validation curves

## 🔮 Future Work

- Increase dataset diversity and add new diamond shapes
- Test newer architectures: EfficientNetV2, MobileNetV3, Swin Transformer
- Develop a mobile app for real-time diamond classification
- Combine image classification with structured data (e.g., gemstone size, cut grade)

## 📄 License

This project is part of an academic dissertation and is provided for educational and research purposes.

## 👨‍🎓 Contact

**Cristian Ursu**  
For questions or collaboration, please contact the project maintainer.
