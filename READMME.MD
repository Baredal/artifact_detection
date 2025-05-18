# Image Classification and Analysis with Vision Transformer (ViT)

## Description

Binary image classification for the presence of artifacts in AI-generated images. The solution consist of two parts: classification model training on full images and on croped images (faces). Then by using ensembling weighted the final prediction for artifact detection is making. Alongside, different preparation and processing were done.
### Features:
- EfficientNetV2m model for image classification.
- Face detection with `MTCNN` from `facenet-pytorch`.
- Evaluation using metrics F1 score, class-wise accuracy.
- Single image prediction for the presence of artifacts.

## Requirements

To run this solution, the following are required:

- **Python 3.10+**
- **CUDA** Ensure you have CUDA support and correspond pytorch version or run notebook on Kaggle/Google Colab with GPU
- **PyTorch** (Ensure PyTorch is installed with GPU support)

## Setup and Installation

### 1. Unzip into folder
### 2. Run following command
```bash
pip install -r requirements.txt
```
### 3. Open Jupyter Notebook and walkthrough
---

### How to Use:
Notebook is already saved with all results and metrics. You can either navigate to **Inference section** to run models inference and predict image class (you have to load import all necessary libraries before in first cell and follow instructions in comments). Notebook provides also comments for clarification of process and steps which were done. 

## Results
The best micro F1-score for full image model was 97% and for face image model 95%. Per class accuracy on enembled method was 100% for artifactless images and 70% for artifact images. More detailed metrics you can view in Notebook (metrics per epoch, etc)
