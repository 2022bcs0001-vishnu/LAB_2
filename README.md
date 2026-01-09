# LAB_2 – Automated Machine Learning Experiments using GitHub Actions

This repository contains **Lab 2** for the course **CSS 426**, focusing on automating machine learning experiment execution and tracking using **GitHub Actions**.

The objective of this lab is to demonstrate how Continuous Integration (CI) can be applied to machine learning workflows to improve **reproducibility, traceability, and consistency** compared to manual experiment tracking.

---

## Overview

In this lab:
- The **Wine Quality Dataset** from the UCI Machine Learning Repository is used.
- Multiple regression experiments are performed by **manually editing training scripts**.
- Each experiment is executed automatically using **GitHub Actions**.
- Evaluation metrics and trained models are stored as workflow artifacts.

Each Git commit represents a **single experiment**, ensuring that results can be traced back to the exact code used.

---

## Repository Structure

```text
LAB_2/
├── .github/workflows/
│   └── train.yml          # GitHub Actions workflow
├── dataset/
│   └── winequality-red.csv
├── outputs/
│   ├── model/             # Trained model artifacts
│   └── results/           # Evaluation metrics (JSON)
├── train.py               # Initial training script
├── train_2.py             # Modified training script for another experiment
├── requirements.txt       # Python dependencies
└── README.md
