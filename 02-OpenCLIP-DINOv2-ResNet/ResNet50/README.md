# Vehicle Model Classification

This project performs vehicle model classification using a fine-tuned ResNet50 model with PyTorch. It allows training on custom-labeled datasets and predicting new images by outputting the predicted model name along with confidence scores.

---

## Project Overview

- Custom dataset is split into training, validation, and test sets automatically.
- Model is trained using transfer learning with ResNet50.
- After training, the model can predict vehicle models from images placed in the `input/` directory.
- The prediction output includes the class name and the confidence percentage.

---

## How to Use

### 1. Prepare Dataset

- Place your vehicle images in subfolders inside the `dataset/` directory.
- Create a `data.json` mapping folder names (e.g. `"000"`) to model names (e.g. `"audi-a4"`).
- Run the following script to split the dataset:
  ```bash
    python 1_resnet50_training_bpolat.py

### 2. Make Predictions 
 - Add images to the input/ folder, then run:
  ```bash
    python 2_resnet50_predict_bpolat.py
