# P05: What Makes a Movie Review Negative?

**Course:** Applied Machine Learning (CSAI2017P)  
**Academic Year:** 2026–27  
**Theme:** P05, Review Sentiment Analysis  
**Group:** G6

## 1. Project Overview

This project investigates whether a linear model using bag-of-words features can outperform nonlinear alternatives when classifying movie reviews as positive or negative.

The project uses the IMDB 50K Movie Reviews dataset and studies which words contribute most strongly to sentiment predictions.

## 2. Team Members

| Name | SAP-ID | Enrollment Number | GitHub Username |
|---|---|---|---|
| Shivang Parashar | 590022267 | 25010100737 | ShivangParashar |
| Arshdeep Singh | 590028294 | 25010103645 | Arshdeep-Singh-Dev |
| Vivek Kuliyal | 590028449 | 25010103872 | VIVEKUK19 |
| Garvit Jain | 590026685 | 25010101700 | Garvitjain0100 |
| Arjun Gupta | 590028000 | 25010103045 | arjungupta006 |
| Member 6 | To be added | To be added | To be added |
| Member 7 | To be added | To be added | To be added |
| Member 8 | To be added | To be added | To be added |


## 3. Dataset

**Dataset:** IMDB 50K Movie Reviews  
**Source:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Download the dataset through Kaggle using a browser. Check whether a permitted copy is available through the course LMS first.

The dataset is approximately 66 MB. It will not be committed to Git. Follow the dataset instructions in this README and place the downloaded CSV in `data/raw/`.

## 4. Planned Methodology

1. Inspect the dataset and class distribution.
2. Clean text, including HTML stripping and tokenisation.
3. Evaluate the effect of stop-word handling and other cleaning steps through ablation experiments.
4. Extract TF-IDF features.
5. Compare Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
6. Evaluate models using accuracy and ROC/AUC.
7. Examine the top positive and negative weighted words.
8. Plot a learning curve showing accuracy against training-set size.

## 5. Repository Structure

- `data/raw/`: locally stored original dataset, excluded from Git.
- `data/processed/`: generated intermediate data.
- `notebooks/`: numbered experiment notebooks.
- `src/`: reusable Python functions.
- `results/figures/`: generated plots.
- `results/metrics.csv`: recorded evaluation metrics.
- `report/`: final project report.
- `REPRODUCIBILITY.md`: environment and execution instructions.

## 6. Environment Setup

Python version and dependency versions will be documented in `requirements.txt` and `REPRODUCIBILITY.md`.

## 7. Results

Results will be reported after the experiments are run. No model performance claims are made at this stage.

## 8. Contributions

Each member's responsibilities and contributions will be documented here.

## 9. Acknowledgements and Tools

External tutorials, AI tools, libraries, and other resources used will be acknowledged with appropriate references.
