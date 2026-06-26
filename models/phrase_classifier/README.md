# Phrase Classifier

## Overview

This directory contains the pretrained weights for the dynamic ASL phrase classifier used in SignVox.

The classifier recognizes complete ASL phrases from MediaPipe Holistic landmark sequences.

---

## Architecture

```text
MediaPipe Holistic

        │

Pretrained ST-GCN Encoder

        │

Bidirectional GRU

        │

Attention Pooling

        │

Phrase Classification
```

---

## Dataset

Custom ASL Meeting Phrase Dataset

* Created by the project team
* Frequently used online meeting phrases
* MediaPipe Holistic landmark representation

---

## Training

The complete training notebook is available here:

`../../notebooks/train_phrase_classifier.ipynb`

---

## Weights

Place pretrained checkpoints inside the `weights/` directory.

Example:

```text
weights/

best_phrase_classifier.pth
```
