# Training Notebooks

This directory contains the notebooks used to train the dynamic recognition models developed as part of the SignVox project.

## Notebook Order

### 1. `dataset_extraction.ipynb`

Extracts MediaPipe Holistic landmarks from raw ASL videos.

The generated landmark files serve as the input for all subsequent training notebooks.

---

### 2. `train_phrase_classifier.ipynb`

Trains the dynamic phrase classifier.

Architecture:

* Pretrained ST-GCN Encoder
* Bidirectional GRU
* Attention Pooling
* Phrase Classification Head

Dataset:

Custom ASL Meeting Phrase Dataset

---

### 3. `train_bigru_decoder.ipynb`

Trains the experimental sequence decoder.

Architecture:

* Pretrained ST-GCN Encoder
* Bidirectional GRU
* GRU Seq2Seq Decoder

Purpose:

Exploratory research towards near-Continuous Sign Language Recognition (CSLR).

---

## Workflow

```text
Raw Videos
      │
      ▼
dataset_extraction.ipynb
      │
      ▼
Extracted Landmarks
      │
      ├──────────────► Phrase Classifier
      │
      └──────────────► Phrase Decoder
```
