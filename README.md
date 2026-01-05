# Federated Learning with Double-CNN for Alzheimer’s Stage Classification

This repository implements a **Double-CNN architecture** integrated into a **Federated Learning (FL)** framework to classify Alzheimer’s disease stages using MRI data. The project addresses the challenge of building clinically deployable deep learning systems where patient data cannot be pooled due to strict privacy regulations. By simulating a multi-hospital environment, this framework evaluates the viability of privacy-preserving diagnostic models under realistic, non-IID data constraints.

## Project Overview

Alzheimer’s disease progresses through a continuous neurodegenerative trajectory, from cognitively normal status to Alzheimer dementia. While deep learning can detect subtle structural changes like cortical thinning and hippocampal atrophy, real-world deployment is hindered by data silos and privacy laws.

This project combines two major components:
1.  **Federated Learning:** A framework where hospitals (clients) train locally on their own MRI collections while a central server aggregates model updates via **FedAvg**, preserving patient privacy.
2.  **Double-CNN:** A dual-path architecture designed to capture both fine-grained and global anatomical patterns.

## Data Preparation

We utilized the public **Alzheimer’s Dataset** from Kaggle, which contains 2D axial MRI slices classified into four stages: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented.

**Preprocessing Pipeline:**
* **Augmentation:** Applied geometric transformations (rotations, flips, affine shifts, elastic distortions) using the Albumentations library.
* **Resizing:** Images were resized to 224×224 pixels with intensity scaled to the [0, 1] range.
* **Skull Stripping:** Performed to filter non-brain tissues and focus on relevant structures.
* **Partitioning:** Data was split into training (81%), validation (9%), and testing (10%) to prevent data leakage.

## Model Architecture: Double-CNN

The model adopts a dual-branch convolutional design inspired by El-Assy et al. (2024) to extract multi-scale features from the same input image:

* **Branch 1 (CNN1):** Uses small **3×3 filters** to capture fine-grained local structures like hippocampal atrophy.
* **Branch 2 (CNN2):** Uses larger **5×5 filters** to capture global anatomical patterns and spatial context.
* **Fusion:** Both branches output 128-dimensional vectors, which are concatenated into a 256-dimensional joint representation before passing through a dense layer for classification.

## Federated Learning Framework

To simulate a realistic clinical environment, we divided the dataset among four clients using three different distribution settings:

1.  **IID (Independent and Identically Distributed):** Approximates centralized training with balanced data.
2.  **Dirichlet Non-IID ($\alpha=0.3$):** Introduces label imbalance, simulating hospitals that specialize in different stages of the disease.
3.  **Quantity Skew:** Clients receive different amounts of data but similar class proportions.

**Training Configuration:**
* **Algorithm:** Federated Averaging (FedAvg).
* **Rounds:** 25 communication rounds.
* **Local Updates:** 5 epochs per round.
* **Optimizer:** Adam (learning rate = 1e-4).

## Results

The model was evaluated on a held-out global test set of 3,402 MRI slices.

| Setting | Global Accuracy | Macro F1 | Key Observations |
| :--- | :--- | :--- | :--- |
| **IID** | **91.48%** | 0.9208 | Performance closely approximates centralized training; stable convergence across all clients. |
| **Quantity Skew** | **90.62%** | 0.9125 | Moderate performance loss; class representation remained stable despite volume imbalance. |
| **Dirichlet** | **89.24%** | 0.8999 | Most challenging setting due to label imbalance; significant performance drops in "Very Mild" and "Mild" classes due to client specialization. |

## Project Structure

```text
data_preprocess.py   # Dataset resizing, skull stripping, and client partitioning
fed_main.py          # FL training loop using FedAvg aggregation
model.py             # Double-CNN architecture definition
pipeline.py          # Training, validation, and testing pipelines
utils.py             # Helper functions for metrics and logging
config.json          # Experiment parameters (e.g., IID vs Dirichlet)
main.py              # Entry point for running experiments
