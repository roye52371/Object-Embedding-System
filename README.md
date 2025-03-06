<div align="center">

# Object Embedding System

</div>

## Overview
This project implements an object embedding system that detects objects in an image, extracts their bounding boxes, and generates embeddings for each detected object. The embeddings allow clustering, retrieval, and comparison of similar objects.

## Features
- **Object Detection**: Uses YOLOv12x to detect objects in an image.
- **Object Embedding Extraction**:
  - **Cropping with ResNet50 (Main Algorithm)**: Crops detected objects and extracts embeddings using a ResNet50 backbone.
  - **CLIP Encoder (Alternative Option)**: Can be used instead of ResNet50 for embedding extraction.
  - **ROI Align Approach (Tested but Not Effective)**: Implemented but found to produce lower-quality embeddings.
- **Visualization**:
  - **t-SNE (2D)**: Projects embeddings into a 2D space for visualization.
  - **PCA (3D)**: Projects embeddings into a 3D space for better class separation.

## 📁 Project Structure
- `Dogs_and_cats_folder`: The dataset used for evaluate the object embedding system.
- `ObjectEmbedding.ipynb`: The main notebook containing the implementation.
- `requirements.txt`: The list of dependencies.
- `README.md`: This documentation file.

## Setup Instructions

### Requirements
The project was tested on Google Colab using a **T4 GPU**.
It should work on any system with a CUDA-compatible GPU.
Make sure you have Python 3.11.11+ installed.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/roye52371/Object-Embedding-System.git
   cd object-embedding
   ```

2. If not working on Google Colab - install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Object Embedder
The embedder can be initialized and tested as follows:
```python
from object_embedder import ObjectEmbedder

detector = "yolo12x"
embed_method = "crop"  # Can be "roi_align" but is not recommended
crop_backbone = "resnet50"  # Can be "clip" as an alternative
conf_threshold = 0.6  # Confidence threshold for detection

embedder = ObjectEmbedder(detection_model=detector, embed_method=embed_method, crop_backbone=crop_backbone, conf_threshold=conf_threshold, target_classes=["dog", "cat"])

# Evaluate on a folder of images
evaluate_data("path/to/your/image/folder", embedder)
```

In addition, the ObjectEmbedding.ipynb file contains the class and evaluation examples and can be run straightforwardly (with relevant data folder path).

### Evaluation Metrics
- **t-SNE (2D)** is used to evaluate local structure and how well embeddings of similar objects cluster.
- **PCA (3D)** is used to analyze global structure and class separability.

## Notes on ROI Align
While ROI Align was implemented for embedding extraction, the results were not satisfactory. The method is available in the code for reference but is not recommended as the primary approach.

## Contact
For questions, reach out via GitHub Issues or email at roye.katzav@gmail.com.

