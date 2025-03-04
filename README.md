# NLP Classifier for Patronizing and Condescending Language Detection

## Overview

This is code used for the NLP coursework submission at Imperial College London, implementing a Natural Language Processing (NLP) classifier to detect patronizing or condescending language in text, trained on the [_Don't Patronize Me!_ dataset](https://aclanthology.org/2020.coling-main.518/). All of the experiments and hyperparameter tuning can be found in `models/` directory.

## Model

The best-performing model in this project is a finetuned DeBERTa model, incorporating:

- Synonym replacement for data augmentation.
- Class-weighted sampling to handle data imbalance.
- Preprocessing (punctuation removal and lemmatization)

## Repository Structure

The repository is organized as follows:

```
📂 analysis       # Code for analyzing the dataset and the final trained model
📂 dataset        # Scripts for reading and splitting the dataset into training and validation sets
📂 models         # Implementation of different models and experiments
    ├── Baseline models: BoW and TF-IDF with logistic regression
    ├── DeBERTa finetuning with hyperparameter tuning
    ├── Data augmentation, sampling, and preprocessing techniques
📄 dev.txt        # Final predictions for the development dataset
📄 test.txt       # Final predictions for the test dataset
```

## Installation and Requirements

To run the code in this repository, install the required dependencies:

```
python3 -m venv venv
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```
This was trained on the GPU lab machines found at Imperial College London

## Acknowledgments

- The dataset used in this project: Don't Patronize Me! [(Dataset Link)](https://github.com/CRLala/NLPLabs-2024/tree/main/Dont_Patronize_Me_Trainingset)
- DeBERTa model from Microsoft for state-of-the-art NLP performance. [(DeBERTa Paper)](https://arxiv.org/abs/2006.03654)

This repository is part of a coursework submission of the NLP course at Imperial College London. Any unauthorized use, reproduction, or submission of this code as original work may result in academic misconduct or plagiarism consequences.
