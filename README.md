Olivetti Faces Clustering
This notebook explores unsupervised learning techniques by applying clustering algorithms to the Olivetti Faces dataset using scikit-learn. The goal is to group similar face images together without using their identity labels.

📊 Project Overview
Dataset: Olivetti Faces (400 grayscale facial images of 40 people, 10 images per person).

Goal: Perform dimensionality reduction and clustering on high-dimensional face image data.

Techniques Used:

PCA (Principal Component Analysis)

KMeans Clustering

t-SNE for visualization

Stratified data splitting

📁 Dataset Details
Each image: 64x64 pixels (flattened into 4096-dimensional vectors).

The dataset is loaded via sklearn.datasets.fetch_olivetti_faces.

🔍 Workflow
Data Loading: Fetch and preprocess Olivetti faces dataset.

Visualization: Display a grid of face images.

Data Splitting: Stratified splitting into training, validation, and test sets.

PCA: Reduce the dimensionality to visualize the data structure.

Clustering: Apply KMeans clustering.

Evaluation: Cluster visualization and comparison with actual labels.

🛠️ Requirements
Python 3.x

numpy

pandas

matplotlib

seaborn

scikit-learn

📌 Notes
The code uses PCA for dimensionality reduction before clustering.

Clustering is done without using labels (unsupervised), but labels are used for evaluation.

Stratified sampling ensures class balance in splits.
