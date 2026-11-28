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

Install the pinned dependencies using:

    python -m pip install -r requirements.txt

The requirements file specifies the package versions used for
the reported experiments. This helps maintain a consistent
environment when reproducing the project.

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

The project uses random seed 42 wherever a randomised operation
requires a reproducible result. The same seed is used consistently
across the notebooks so that the data split, cross-validation folds,
model configurations and sampled error examples can be reproduced.

The locations of the seed settings are listed below.

| Notebook | Location | Purpose |
|---|---|---|
| `01-eda.ipynb` | `GroupShuffleSplit` | Reproducible 80:20 train/test split |
| `01-eda.ipynb` | `StratifiedGroupKFold` | Reproducible five-fold cross-validation on the training data |
| `02-preprocessing.ipynb` | Baseline `LogisticRegression` pipeline | Sets the classifier random state |
| `02-preprocessing.ipynb` | `StratifiedKFold` | Reproducible five-fold cross-validation |
| `02-preprocessing.ipynb` | Logistic Regression pipelines in the preprocessing ablation and feature-representation experiments | Consistent model configuration across experiments |
| `03-models.ipynb` | `StratifiedKFold` | Reproducible cross-validation for model selection |
| `03-models.ipynb` | `LinearSVC` | Sets the Linear SVM random state |
| `03-models.ipynb` | `RandomForestClassifier` | Reproducible random forest training |
| `04-evaluation.ipynb` | `LogisticRegression` and `RandomForestClassifier` | Sets the random state in the final model configurations |
| `04-evaluation.ipynb` | `learning_curve` | Reproducible shuffling of training samples |
| `04-evaluation.ipynb` | `DataFrame.sample` | Reproducible selection of 30 misclassified reviews for manual inspection |

All the random-state settings listed above use the value `42`.
Logistic Regression and Multinomial Naive Bayes are otherwise
deterministic in the configurations used in this project.

The seed settings are defined directly in the relevant notebook
cells rather than through a separate global seed variable.

## 4. Run order

Run the notebooks from the `notebooks/` directory so that their
relative data and results paths resolve correctly.

From the repository root, activate the virtual environment and run:

    cd notebooks
    jupyter lab

Use Restart & Run All in this exact order:

1. `01-eda.ipynb`
   Loads the raw IMDB dataset, performs EDA, creates the
   duplicate-aware train/test split and saves the processed CSV files.

2. `02-preprocessing.ipynb`
   Runs the cleaning ablation and compares count and TF-IDF
   representations.

3. `03-models.ipynb`
   Performs five-fold cross-validation and model selection.

4. `04-evaluation.ipynb`
   Evaluates the selected models on the held-out test set and
   generates the metrics, figures and error-analysis outputs.

Notebook 01 requires the manually downloaded raw dataset.
Notebooks 02–04 can use the processed train and test CSV files
already stored in `data/processed/`.

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