# Assignment 2: LSTM Code Summarization
**CSCI 455/555 — Spring 2026 | Prof. Antonio Mastropaolo**

---

## Overview

This project implements a Seq2Seq LSTM model that generates natural language summaries for Java methods, using pretrained CodeT5+ embeddings (32,100 × 768).

---

## Environment

This notebook is intended to run in **Google Colab** with a **T4 GPU** runtime (`Runtime → Change runtime type → T4 GPU`). It will run on CPU but training will be extremely slow.

---

## Files

The following files must be present in Colab's working directory (`/content/`) before running:

- `get_codet5_embeddings.py` — provided by the instructor; used to tokenize data and extract the CodeT5+ embedding matrix
- `test_dataset_tokenized.csv` — instructor-provided test set (99 samples)

The `dataset/all_pairs.pkl` file is a backup of the custom-mined GitHub dataset. Importing this file allows skipping cells 1.1 through 1.5.

---

## Dataset

Two datasets are available and referenced in the notebook:

### 1. Custom-Mined GitHub Dataset (`/content/data/`)
Java method–summary pairs mined from 59 public GitHub repositories using the GitHub API and `javalang`. Approximately 6,700 pairs were extracted after filtering test/example/generated directories.

> **Note:** Data mining was performed in a separate Colab session. Due to Google Drive sync failures during that session, the mined pairs were saved locally and exported as `all_pairs.pkl` (contained in this repo). The notebook loads this pickle in place of re-running the mining pipeline. The mining code is included in the notebook for reproducibility but its saved outputs display results from that prior session.

### 2. CodeSearchNet Alternative Dataset (`/content/code_search_net/`)
As a documented alternative, the notebook also includes a corpus construction pipeline using the HuggingFace `code_search_net` Java split — the reference dataset for this task (CodeXGLUE). This produces ~50,000 train pairs and 1,000 validation pairs, satisfying the assignment's corpus size requirements. This was used as a fallback due to the Drive persistence issues described above.

---

## Installing Dependencies

Run the first two cells of the notebook in order. Key packages:

```
torch==2.0.1
transformers==4.46.0
sentencepiece==0.2.1
tokenizers==0.20.3
numpy==1.26.4
datasets==2.14.0
huggingface_hub==0.19.4
sacrebleu
evaluate
bert_score
javalang
```

The SIDE metric repository is cloned directly from GitHub (not pip-installable):
```
git clone https://github.com/antonio-mastropaolo/code-summarization-metric.git
```

> **Note:** Due to version conflicts between `transformers`, `huggingface_hub`, and `datasets`, a specific pinned combination is required. Do not upgrade these packages independently.

---

## Reproducing the Results

1. Open `assignment-2-LSTM.ipynb` in Google Colab with a T4 GPU runtime
2. Place `get_codet5_embeddings.py`, `test_dataset_tokenized.csv`, and `all_pairs.pkl` in `/content/`
3. Run all cells top to bottom
4. The notebook will skip re-mining and load from `all_pairs.pkl`
5. `.pt` embedding files will be generated in `/content/data/` or `/content/code_search_net/`

The saved cell outputs in the notebook display results from the best trained model checkpoint.

---

## Outputs during Runtime

All outputs are written to `/content/` (Colab local storage):

| File | Description |
|---|---|
| `data/` | Tokenized `.txt` and `.pt` files from mined dataset |
| `code_search_net/` | Tokenized `.txt` and `.pt` files from CodeSearchNet |
| `checkpoints/lstm_best.pt` | Best model checkpoint (highest validation BLEU-1) |
| `test_predictions.json` | Final predictions on the instructor test set |
| `training_curves.png` | Train loss and validation BLEU-1 curves |

---

## Evaluation Results (Test Set)

| Metric | Score |
BLEU-1       : 34.67
BLEU-2       : 26.86
BLEU-3       : 12.34
BLEU-4       : 5.24
METEOR       : 0.13
BERTScore-F1 : 0.5621
SIDE         : 0.2954

---

## Model Architecture

Seq2Seq LSTM adapted from the class bug-fixing notebook:
- Shared CodeT5+ embedding layer (32,100 × 768), fine-tuned
- Linear projection: 768 → 512 before LSTM
- 2-layer stacked encoder + decoder LSTM (hidden dim: 512)
- Teacher forcing during training
- Early stopping on validation BLEU-1 (patience = 3, evaluated every 500 steps)