# Multiple Instance Learning for Histopathology: An Introduction to torchmil

A comprehensive tutorial on Multiple Instance Learning (MIL) for histopathology using the `torchmil` library. This tutorial demonstrates how to train attention-based MIL models on Whole Slide Images (WSIs) from the CAMELYON16 dataset to detect breast cancer metastases.

This tutorial is part of the [RISE MICCAI Tutorial Series](https://rise.miccai.org/tutorials/), which provides in-depth guides on various topics in medical image computing.

## Overview

This repository contains:
- **tutorial.ipynb** - Interactive Jupyter notebook with complete tutorial and code examples
- **tutorial.html** - HTML version of the tutorial
- **extract_labels_camelyon16.py** - Script to extract patch-level labels from CAMELYON16 annotations
- **train.csv / test.csv** - Pre-split WSI labels for training and testing
- **wsi_labels.csv** - CSV file containing WSI-level labels for the CAMELYON16 dataset

## Key Topics

1. **Data Preparation** - Working with Whole Slide Images and patch extraction.
2. **Multiple Instance Learning** - Understanding MIL concepts and the ABMIL model architecture.
3. **Model Training** - Implementing and training attention-based MIL models with torchmil.
4. **Evaluation & Visualization** - Assessing model performance and localizing predictions in WSIs.

## Data

Download the CAMELYON16 dataset from [here](https://camelyon16.grand-challenge.org/).

## References

- torchmil library: [https://torchmil.readthedocs.io/](https://torchmil.readthedocs.io/)
- Tutorial slides: [https://franblueee.github.io/assets/pdf/slides/2026_rise_miccai.pdf](https://franblueee.github.io/assets/pdf/slides/2026_rise_miccai.pdf)
