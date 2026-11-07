# REPRODUCIBILITY: aml-2026-p05-g6

Theme P05: review sentiment on the IMDB 50K movie reviews. Applied Machine
Learning (CSAI2017P), Autumn 2026, Group G6.

## 1. Environment

- Python: 3.14.2
- Hardware used: GitHub Codespaces, 2 CPU cores, 7.8 GB RAM, no GPU.
- Versions used for the reported results:

| Package | Version |
|---|---|
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| joblib | 1.6.0 |
| beautifulsoup4 | 4.15.0 |
| snowballstemmer | 3.1.1 |
| jupyter | 1.1.1 |
| ipykernel | 7.4.0 |

Recreate the environment from a clean checkout:

    git clone https://github.com/Garvitjain0100/AML-2026-p05-g6.git
    cd AML-2026-p05-g6
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

`requirements.txt` lists the packages without version numbers, so pip installs
the newest releases. To get exactly the versions in the table above, install
them with `pip install package==version`.

On the UPES network `pip install` can fail because of the campus TLS
appliance. If it does, install from home or mobile data. Certificate
verification is never disabled anywhere in this repository.

## 2. Manual step: the raw data

The IMDB 50K file (about 66 MB) is over the 50 MB limit, so `data/raw/` is
gitignored and the file must be downloaded by hand.

1. Open https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
   in a browser and sign in to Kaggle (a free account is enough).
2. Click Download and unzip the archive.
3. Put the CSV, unmodified, in `data/raw/` (Kaggle's default name is
   `IMDB Dataset.csv`; `01-eda.ipynb` must point at whatever name is used).
4. Check: 50,000 rows, columns `review` and `sentiment`, 25,000 of each class.

The train/test split generated from it is stored in `data/processed/`
(`train.csv`: 40,006 rows, `test.csv`: 9,994 rows). Notebooks 02, 03 and 04
read these two files, so they can be run without the raw file as long as
the two CSVs are present.

## 3. Random seeds

Every seeded step uses the value 42.

| Where | Seed | Purpose |
|---|---|---|
| `notebooks/01-eda.ipynb` (`random_state=42`, two places) | 42 | train/test split and the other seeded step in the EDA notebook |
| `notebooks/02-preprocessing.ipynb` | 42 | `StratifiedKFold` and the baseline logistic regression |
| `notebooks/03-models.ipynb` | 42 | `StratifiedKFold` for all four grid searches, `LinearSVC`, `RandomForestClassifier` |
| `notebooks/04-evaluation.ipynb` (`random_state=42`, four places) | 42 | model settings and cross-validation in the evaluation notebook |

Logistic regression and Multinomial Naive Bayes have no random component in
the way they are used here.

## 4. Run order

Run from the repository root with the environment active, using Restart & Run All
in each notebook:

1. `notebooks/01-eda.ipynb`: explores the data and writes `data/processed/train.csv` and `test.csv`
2. `notebooks/02-preprocessing.ipynb`: baseline and cleaning ablation (HTML stripping, lowercasing, punctuation, stop-words)
3. `notebooks/03-models.ipynb`: tunes Logistic Regression, Naive Bayes, Linear SVM and Random Forest with 5-fold stratified CV; writes `results/model_comparison.csv` and `results/cv_results.csv`
4. `notebooks/04-evaluation.ipynb`: test-set metrics, ROC curves, top weighted words, learning curve; writes `results/metrics.csv` and the figures in `results/figures/`

Command-line alternative:

    jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb

## 5. Runtime and memory

- Notebook 03 is the slow one because it runs four grid searches (60 model
  fits in total, plus the forest). On 2 cores expect several minutes.
- Memory matters. On the 2-core, 7.8 GB codespace, an additional bigram TF-IDF
  sweep run with `n_jobs=-1` killed the worker processes
  (`TerminatedWorkerError`, SIGTERM) and later crashed the Jupyter kernel.
  If you add experiments, use `n_jobs=1` or restart the kernel first.

## 6. What may not reproduce exactly

- Metrics should match to about three decimals with the library versions
  above. Different scikit-learn or BeautifulSoup versions can change the
  last digits.
- Wall-clock times depend on the machine.
- The Random Forest in notebook 03 uses a reduced feature set (2,000
  unigrams, 20 trees) to fit in memory, so its score is not a like-for-like
  comparison with the bigram TF-IDF models.
- `results/metrics.csv` is written by notebook 04 only. It must be generated
  by running that notebook and committing the result.

## 7. Tools and references

AI tools and external references used are credited in `README.md`.