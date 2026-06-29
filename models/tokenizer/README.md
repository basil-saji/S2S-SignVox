# Phrase Decoder

## Overview

This directory contains the experimental sequence decoder developed as part of the SignVox research project.

Unlike the phrase classifier, this model predicts token-level phrase representations using sequence decoding techniques.

The decoder is intended as an exploratory step toward future near-Continuous Sign Language Recognition (CSLR).

---

## Architecture

```text
MediaPipe Holistic

        │

Pretrained ST-GCN Encoder

        │

Bidirectional GRU

        │

GRU Seq2Seq Decoder

        │

Token Sequence
```

---

## Motivation

Current isolated sign recognition systems perform phrase classification effectively but cannot model longer temporal language structures.

This decoder explores sequence modeling on a custom ASL meeting phrase dataset and serves as a research baseline for future sentence-level recognition.

The present implementation is constrained primarily by dataset size rather than model capacity. Larger and more diverse datasets could enable more expressive sequence models, including transformer-based decoder architectures.

---

## Training

The complete training notebook is available here:

`../../notebooks/train_bigru_decoder.ipynb`

---

## Weights

Place pretrained checkpoints inside the `weights/` directory.

Example:

```text
weights/

best_phrase_decoder.pth
```
