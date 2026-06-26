# SignVox

### Graph-Based ASL Recognition using Deep Learning

---

## Overview

SignVox is a research project focused on efficient American Sign Language (ASL) recognition using computer vision and graph-based deep learning. The project explores both static fingerspelling recognition and dynamic phrase recognition using MediaPipe Holistic landmarks and spatio-temporal neural networks.

The repository contains the complete training pipeline for the recognition models developed during the project, including landmark extraction, phrase classification, and exploratory sequence decoding experiments.

---

## Project Highlights

* MediaPipe Holistic landmark extraction pipeline
* Static ASL fingerspelling recognition
* Dynamic phrase recognition using graph neural networks
* Exploratory sequence decoding for future near-CSLR research
* Lightweight models suitable for deployment on consumer-grade hardware
* Complete training notebooks for reproducibility

---

## Repository Structure

```text
SignVox/

├── notebooks/
│   ├── dataset_extraction.ipynb
│   ├── train_phrase_classifier.ipynb
│   └── train_bigru_decoder.ipynb
│
├── models/
│   ├── mlp/
│   ├── phrase_classifier/
│   └── phrase_decoder/
│
├── requirements.txt
└── README.md
```

---

## Recognition Pipeline

The repository contains three independent research components.

### 1. Static Fingerspelling Recognition

A lightweight Multilayer Perceptron (MLP) trained on the IEEE DataPort ASL Fingerspelling Dataset for recognizing individual hand signs.

---

### 2. Dynamic Phrase Classification

Dynamic phrase recognition is performed using a pretrained Spatio-Temporal Graph Convolutional Network (ST-GCN) encoder followed by a Bidirectional Gated Recurrent Unit (BiGRU) classifier.

The model is trained on a custom ASL dataset consisting of frequently used online meeting phrases collected and annotated by the project team.

---

### 3. Sequence Decoding Research

The repository also contains an experimental sequence decoder built upon the same pretrained ST-GCN encoder.

Instead of directly predicting phrase classes, the decoder generates token-level phrase representations using a BiGRU-GRU Seq2Seq architecture.

This work represents an initial step toward future near-Continuous Sign Language Recognition (CSLR). Owing to the limited size of the current dataset, these experiments should be viewed as exploratory research rather than a complete CSLR solution.

---

## Dataset

Two datasets are used throughout this project.

### IEEE DataPort ASL Fingerspelling Dataset

Used for training the static MLP fingerspelling recognizer.

### Custom ASL Phrase Dataset

A custom dataset created by the project team containing commonly used ASL phrases frequently encountered during online meetings.

MediaPipe Holistic landmarks are extracted from each recording and used as the input representation for all dynamic recognition models.

---

## Training Workflow

The notebooks are intended to be executed in the following order.

1. **dataset_extraction.ipynb**

   Extracts MediaPipe Holistic landmarks from the raw videos.

2. **train_phrase_classifier.ipynb**

   Trains the ST-GCN + BiGRU phrase classifier using the extracted landmarks.

3. **train_bigru_decoder.ipynb**

   Trains the experimental sequence decoder using the same extracted landmark dataset.

---

## Future Work

The sequence decoder included in this repository serves as a research baseline toward near-Continuous Sign Language Recognition.

Future work will investigate:

* larger sign language datasets
* transformer-based sequence modeling
* richer linguistic representations
* scalable sentence-level recognition
* real-time deployment

---

## License

This repository is released under the MIT License.
