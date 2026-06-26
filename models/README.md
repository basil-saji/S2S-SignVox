# Models

This directory contains the trained recognition models developed during the SignVox project.

## Directory Structure

```text
models/

├── mlp/
├── phrase_classifier/
└── phrase_decoder/
```

---

## MLP

Static ASL fingerspelling recognition model.

See:

`models/mlp/README.md`

---

## Phrase Classifier

Dynamic ASL phrase recognition model based on a pretrained ST-GCN encoder and a Bidirectional GRU classifier.

See:

`models/phrase_classifier/README.md`

---

## Phrase Decoder

Experimental sequence decoder developed as an initial step toward future sentence-level sign language recognition.

See:

`models/phrase_decoder/README.md`
