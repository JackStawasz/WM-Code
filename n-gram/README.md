# N-Gram Language Model for Code

A statistical n-gram language model trained on tokenized Java/code functions. The model uses add-alpha smoothing and greedy prediction to estimate token probabilities and compute perplexity over a held-out corpus.

---

## Project Structure

```
ngram/
├── src/
    ├── n-gram_constructor.ipynb            # N-gram predictor model training
    └── Ngram_dataset_Collection.ipynb      # Clones Github repos and tokenizes java methods
├── models/
    ├── model1        # Saved model (pickle) — trained on ~15k samples
    ├── model2        # Saved model (pickle) — trained on ~25k samples
    └── model3        # Saved model (pickle) — trained on ~35k samples
├── data/
    ├── train.txt                 # 75,499 tokenized functions (training data)
    ├── val.txt                   # 1,000 functions (validation)
    ├── test.txt                  # 1,000 functions (test)
    ├── provided_test_set.txt     # Provided evaluation test set
    ├── results-xxxxxx.json       # Predictions on provided test set
    └── results-yyyyyy.json       # Predictions on local test set
```

## Dependencies

The notebooks run on **Google Colab** and require Python 3. Standard library packages (`json`, `pickle`, `collections`, `pathlib`, `time`) are pre-installed. The following packages are auto-installed at runtime if missing:

```bash
pip install lizard humanize
```

| Package | Version | Use |
|---|---|---|
| `numpy` | any | Log-probability computation |
| `matplotlib` | any | Perplexity vs. n plots |
| `lizard` | any | Code complexity / tokenization utilities |
| `humanize` | any | Human-readable training time output |

> **Google Drive** is required for file I/O. Both notebooks call `drive.mount('/content/drive')` at startup. Update `REPO_PATH` in the Load Files cell to match your Drive layout:
> ```python
> REPO_PATH = Path("/content/drive/MyDrive/<your-path>/ngram_dataset")
> ```

---

## Running the Notebooks

### 1. Dataset Collection

Open and run `Ngram_Dataset_Collection.ipynb` top-to-bottom to scrape and write `train.txt`, `val.txt`, and `test.txt` into `REPO_PATH`. Skip this step if the dataset files already exist.

### 2. Training

Open `n-gram_constructor.ipynb` and run cells in order.

**Load data** — The Load Files cell reads all three splits and prints record counts:
```
train.txt: 75499 functions
val.txt:   1000 functions
test.txt:  1000 functions
```

**Split training data** — The corpus is automatically partitioned into three subsets for comparative experiments:

| Subset | Max size |
|---|---|
| `train1_corpus` | 15,000 functions |
| `train2_corpus` | 25,000 functions |
| `train3_corpus` | 35,000 functions |

**Train models** — Three models are trained and saved to `REPO_PATH`:
```python
n = 3  # set context window size here before running
model1 = train_model("model1", model, train1_corpus, n, save=True)
model2 = train_model("model2", model, train2_corpus, n, save=True)
model3 = train_model("model3", model, train3_corpus, n, save=True)
```

Each model prints elapsed training time and confirms the save path.

### 3. Validation

Run the perplexity evaluation cells to score each model against `val.txt`. Results are accumulated in a `perplexities` dictionary keyed by model name and `n`:

```
========== Perplexities for n=3 ==========
Model 1 Perplexity: <value>
Model 2 Perplexity: <value>
Model 3 Perplexity: <value>
```

A plot of **Perplexity vs. n** is generated automatically once you have run evaluations across multiple values of `n`.

### 4. Testing & Output Generation

Run the **Save Metadata** cell to generate a JSON results file:

```python
# Provided test set
test_fname = "provided_test_set.txt"
out_fname  = "results-xxxxxx.json"

# Uncomment for local test set
# test_fname = TEST_FNAME
# out_fname  = "results-yyyyyy.json"

output = build_output(test_fname, test_corpus, model3)
```

The file is written to `REPO_PATH / out_fname`.

---

## Hyperparameters

| Parameter | Variable | Default | Description |
|---|---|---|---|
| Context window size | `n` | `3` | Number of tokens in each n-gram (1 = unigram, 2 = bigram, 3 = trigram, …). Controls how much history the model conditions on. Larger `n` captures longer dependencies but increases data sparsity. |
| Add-alpha smoothing | `alpha` | `0.1` | Laplace smoothing coefficient added to all counts. Prevents zero probabilities for unseen n-grams. A value of `1.0` is full Laplace smoothing; smaller values (e.g. `0.01`) apply lighter smoothing and stay closer to the raw MLE. |

After hyperparameter tuning on the validation set `val.txt`, the following optimal values were reached:

```python
n     = 3
alpha = 0.1
```

---

## Smoothing Details

The model applies **add-alpha (Lidstone) smoothing** during probability estimation:

```
P(w | context) = (count(context, w) + α) / (count(context) + α × |V|)
```

Where `|V|` is the vocabulary size. For unseen contexts at inference time, the model falls back to a **uniform distribution** `1 / |V|`.
