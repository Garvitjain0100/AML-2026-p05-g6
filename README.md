# P05: What Makes a Movie Review Negative?

**Course:** Applied Machine Learning (CSAI2017P)  
**Academic Year:** 2026–27  
**Theme:** P05, Review Sentiment Analysis  
**Group:** G6

## 1. Project Overview

This project investigates whether a linear model using bag-of-words features can outperform nonlinear alternatives when classifying movie reviews as positive or negative.It also examines which words contribute most strongly to sentiment predictions.

The project uses the IMDB 50K Movie Reviews dataset. It evaluates text preprocessing choices, compares count-based and TF-IDF representations, and tests Logistic Regression, Multinomial Naive Bayes, Linear SVM and Random Forest classifiers.

## 2. Team Members

| Name | SAP-ID | Enrollment Number | GitHub Username |
|---|---|---|---|
| Shivang Parashar | 590022267 | 25010100737 | ShivangParashar |
| Arshdeep Singh | 590028294 | 25010103645 | Arshdeep-Singh-Dev |
| Vivek Kuliyal | 590028449 | 25010103872 | VIVEKUK19 |
| Garvit Jain | 590026685 | 25010101700 | Garvitjain0100 |
| Arjun Gupta | 590028000 | 25010103045 | arjungupta006 |
| Satvik Indoria | 590023384 | 25010101710 | Satvik-indoria |
|Prince Bisht | 590027473 | 25010104012 | Prince-Bisht |
| Priyanshu Kumar | 590025327 | 25010103182 |priyanshukumar-bot|


## 3. Dataset

**Dataset:** IMDB 50K Movie Reviews  
**Source:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

The dataset contains 50,000 labelled movie reviews, with 25,000 positive and 25,000 negative examples.

### Download and placement instructions

1. Open the Kaggle dataset URL in a web browser.
2. Sign in to Kaggle using a free account.
3. Click the Download button to download the dataset archive.
4. Extract the downloaded archive.
5. Locate the original file named `IMDB Dataset.csv`.
6. Copy this file without modifying it into the repository's `data/raw/` directory.
7. Confirm that the file is located at `data/raw/IMDB Dataset.csv` and contains the columns `review` and `sentiment`.

The raw dataset is approximately 66 MB and is excluded from Git because it exceeds the 50 MB repository limit. It must be downloaded separately before running notebook 01.

## 4. Planned Methodology
1. **Exploratory data analysis:** Examined dataset size, class balance, missing values, review lengths and duplicate reviews.
2. **Duplicate-aware splitting:** Used `GroupShuffleSplit` with a 20% held-out test size and random seed 42 to ensure identical review texts do not cross the training and test sets.
3. **Text preprocessing:** Evaluated HTML stripping, lowercasing, punctuation removal, stop-word removal and stemming through ablation experiments.
4. **Feature representation:** Compared raw count vectors, unigram TF-IDF and unigram-plus-bigram TF-IDF features.
5. **Model comparison:** Evaluated Logistic Regression, Multinomial Naive Bayes, Linear SVM and Random Forest using five-fold stratified group cross-validation.
6. **Final evaluation:** Compared models on the held-out test set using accuracy, precision, recall, F1-score and ROC-AUC.
7. **Model interpretation and analysis:** Examined positive and negative feature weights, learning curves, prediction confidence, robustness to formatting changes and a sample of misclassified reviews.

## 5. Repository Structure

```text
AML-2026-p05-g6/
├── README.md
├── requirements.txt
├── .gitignore
├── REPRODUCIBILITY.md
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-models.ipynb
│   └── 04-evaluation.ipynb
├── src/
├── results/
│   ├── figures/
│   ├── metrics.csv
│   ├── cv_results.csv
│   ├── model_comparison.csv
│   ├── confidence_summary.csv
│   ├── high_confidence_errors.csv
│   ├── misclassified_reviews.csv
│   └── robustness_check.csv
└── report/
    └── report.pdf
```

## 6. Environment Setup and Execution

The reported environment uses Python 3.14.2.

Clone the repository:

```bash
git clone https://github.com/Garvitjain0100/AML-2026-p05-g6.git
cd AML-2026-p05-g6
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it using:

```powershell
.venv\Scripts\activate
```

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Download and place the raw dataset as described in Section 3.

Launch Jupyter Lab from the notebooks directory:

```bash
cd notebooks
jupyter lab
```

Run the notebooks in this exact order:

1. `01-eda.ipynb`
2. `02-preprocessing.ipynb`
3. `03-models.ipynb`
4. `04-evaluation.ipynb`

Notebook 01 generates the processed training and test CSV files. The subsequent notebooks use these generated files.

For further environment details, random seeds, runtime notes and reproduction instructions, see `REPRODUCIBILITY.md`.

## 7. Results

The current saved held-out test results are shown below. The test set contains 9,994 reviews, produced by the duplicate-aware 80:20 split.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.9130 | 0.9052 | 0.9227 | 0.9139 | 0.9701 |
| Logistic Regression | 0.9115 | 0.9020 | 0.9233 | 0.9126 | 0.9704 |
| Multinomial Naive Bayes | 0.8862 | 0.8802 | 0.8941 | 0.8871 | 0.9545 |
| Random Forest | 0.8364 | 0.8385 | 0.8333 | 0.8359 | 0.9156 |

The majority-class baseline achieved an accuracy of 0.4999.

Linear SVM achieved the highest saved test accuracy and F1-score. Logistic Regression achieved a marginally higher ROC-AUC and was selected for probability-based confidence analysis and coefficient interpretation.

**Important comparison limitation:** Random Forest used a reduced feature configuration during cross-validation because of memory constraints. Its cross-validation configuration used 2,000 unigram features and 20 trees, while its final test configuration used 20,000 unigram features and 100 trees. Its CV result is therefore not a like-for-like comparison with the bigram TF-IDF linear models.

Regenerate and verify this table from `results/metrics.csv` after the duplicate-aware cross-validation correction in notebooks 02 and 03. Update the values here if the final model configurations or test results change.

## 8. Contributions

The following is a draft based on the changes recorded in the repository's Git history. Each member should verify that the description accurately reflects their actual work before submission.

| Member | Work recorded in the repository |
|---|---|
| Garvit Jain | Initial repository setup, dependency configuration, processed data and duplicate-aware EDA split. |
| Arshdeep Singh | Initial EDA notebook and preprocessing notebook refactoring. |
| Vivek Kuliyal | Initial addition of the model-comparison notebook. |
| Arjun Gupta | EDA notebook additions and updates. |
| Satvik Indoria | Evaluation extensions, confidence analysis, robustness checks, result outputs and final report. |
| Prince Bisht | Initial addition of the model-evaluation notebook. |
| Priyanshu Kumar | Preprocessing pipeline corrections, import consolidation and HTML-handling improvements. |
| Shivang Parashar | Model-comparison result tables and reproducibility documentation. |

## 9. Acknowledgements and Tools

The project uses Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, BeautifulSoup and SnowballStemmer.
