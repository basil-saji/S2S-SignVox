<div align="center">

# 🤟 SignVox

### **Graph-Based American Sign Language Recognition using Deep Learning**

*Lightweight Graph Neural Networks for Static and Dynamic ASL Recognition*

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-4285F4?style=for-the-badge\&logo=google\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)
![Research](https://img.shields.io/badge/Status-Research-blueviolet?style=for-the-badge)

</div>

---

## 📖 Overview

**SignVox** is a research project exploring efficient **American Sign Language (ASL)** recognition using **MediaPipe Holistic** and modern graph-based deep learning architectures.

This repository contains the complete training pipeline developed during the project, including:

* MediaPipe landmark extraction
* Static fingerspelling recognition
* Dynamic phrase classification
* Experimental sequence decoding
* Pretrained model weights
* End-to-end training notebooks

The primary objective of this repository is to provide a reproducible research pipeline for lightweight ASL recognition while laying the groundwork for future sentence-level sign language understanding.

---

# ✨ Highlights

* 🖐️ Static ASL Fingerspelling Recognition
* 🎥 Dynamic Phrase Recognition
* 🕸️ Graph-based Spatial-Temporal Modeling
* 🧠 ST-GCN + BiGRU Architecture
* 📍 MediaPipe Holistic Landmark Pipeline
* ⚡ Lightweight CPU-Friendly Inference
* 📓 Complete Training Notebooks
* 🔬 Preliminary Research towards Near-CSLR

---

# 🏗 Repository Structure

```text
SignVox
│
├── notebooks/
│   ├── README.md
│   ├── dataset_extraction.ipynb
│   ├── train_phrase_classifier.ipynb
│   └── train_bigru_decoder.ipynb
│
├── models/
│   ├── README.md
│   ├── mlp/
│   ├── phrase_classifier/
│   └── phrase_decoder/
│
├── requirements.txt
└── README.md
```

---

# 🧠 Recognition Models

| Model                 | Purpose                           | Architecture                                  |
| --------------------- | --------------------------------- | --------------------------------------------- |
| **MLP**               | Static Fingerspelling Recognition | Multilayer Perceptron                         |
| **Phrase Classifier** | Dynamic Phrase Classification     | Pretrained ST-GCN Encoder + BiGRU             |
| **Phrase Decoder**    | Sequence Modeling Research        | Pretrained ST-GCN Encoder + BiGRU-GRU Seq2Seq |

---

# 📊 Datasets

### IEEE DataPort ASL Fingerspelling Dataset

Used for training the static fingerspelling recognizer.

---

### Custom ASL Meeting Phrase Dataset

A custom dataset collected and annotated by the project team consisting of frequently used American Sign Language phrases commonly encountered during online meetings.

MediaPipe Holistic landmarks are extracted from every recording and used as the input representation for all dynamic recognition models.

---

# ⚙️ Training Pipeline

Execute the notebooks in the following order.

| Step  | Notebook                        |
| ----- | ------------------------------- |
| **1** | `dataset_extraction.ipynb`      |
| **2** | `train_phrase_classifier.ipynb` |
| **3** | `train_bigru_decoder.ipynb`     |

The extraction notebook generates MediaPipe Holistic landmark sequences from the raw recordings. These extracted landmarks are subsequently used by both the phrase classifier and the experimental sequence decoder.

---

# 🔬 Research Direction

The sequence decoder included in this repository is an exploratory effort toward **near-Continuous Sign Language Recognition (CSLR)**.

While the current implementation demonstrates promising sequence modeling capabilities, the available dataset is relatively small for training data-intensive language models. Consequently, the decoder should be viewed as a research baseline rather than a complete CSLR solution.

Future work will explore:

* Transformer-based sequence decoders
* Larger sign language datasets
* Richer linguistic representations
* Sentence-level recognition
* Real-time continuous inference

---

# 📚 Citation

If you find this repository useful in your research, please consider citing the project.

```bibtex
@software{signvox2026,
  title={SignVox: Graph-Based American Sign Language Recognition using Deep Learning},
  author={Basil Saji and Contributors},
  year={2026}
}
```

---

# 📄 License

Released under the **MIT License**.

---

<div align="center">

**⭐ If you find this project useful, consider giving it a star.**

</div>
