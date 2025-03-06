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

## **Expected Output**
- **Bounding boxes**: The model detects objects and returns bounding boxes with confidence scores.
- **Feature embeddings**: A numerical vector representing each detected object.
- **Visualization**:
  - **t-SNE 2D plot**: Clusters similar objects (e.g., all dogs together, all cats together).
  - **PCA 3D plot**: Provides another view of class separation in a reduced space.

## Example log output (from my crop+ResNet50 algorithm)
```
detector: yolo12x
Embedding Method: crop
Crop Backbone: resnet50
 Confidence Threshold: 0.6
YOLOv12x summary (fused): 283 layers, 59,135,744 parameters, 0 gradients, 199.0 GFLOPs
Processed 138 embeddings after filtering.
Processed 35 images.
 Average processing time per image: 114.92 ms
```

## Expected Results (from my crop+ResNet50 algorithm)

Below are example outputs of the object embedding process:

### t-SNE Visualization of Embeddings
![t-SNE Plot](Results_image/Result_crop_resnet50_tSNE_2D.png)

### PCA 3D Visualization of Embeddings
![PCA 3D Plot](Results_image/Result_crop_resnet50_PCA_3D.png)


## **Notes on RoIAlign**
- RoIAlign was implemented as an alternative embedding method but was found to be suboptimal for high-quality embeddings. The cropping method with **ResNet50** provides the best results.

## **Acknowledgments**
- **YOLOv12x** for object detection
- **ResNet50 / CLIP** for feature extraction
- **t-SNE & PCA** for visualization

## **License**
This project is released under the MIT License.

---

**Author:** Your Name

## Notes on ROI Align
While ROI Align was implemented for embedding extraction, the results were not satisfactory. The method is available in the code for reference but is not recommended as the primary approach.

## Contact
For questions, reach out via GitHub Issues or email at roye.katzav@gmail.com.

