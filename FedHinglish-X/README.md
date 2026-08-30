# FedHinglish-X

Privacy-preserving personalized federated transformer for Hindi-English code-mixed sentiment analysis.

## What this project adds

This is a research extension of the original `Negative-Tweet-Detector` repository. The old system uses TF-IDF + Logistic Regression for six toxicity labels. This module changes the research task to 3-class sentiment analysis and adds:

- MuRIL/XLM-R transformer backbone
- LoRA-style lightweight client adapters
- non-IID client simulation
- personalized client models
- client-update clipping + Gaussian differential privacy
- weighted asynchronous federated aggregation
- client-level evaluation and fairness statistics
- Flask inference API

The implementation is intentionally a reproducible research prototype. It does **not** claim experimental improvement until you run the supplied experiments on the chosen dataset.

## Directory

```text
FedHinglish-X/
├── app.py
├── config.py
├── data.py
├── model.py
├── privacy.py
├── federated.py
├── train_centralized.py
├── train_federated.py
├── evaluate.py
├── requirements.txt
└── data/README.md
```

## Dataset

The code accepts a local CSV with two columns:

```csv
text,label
"movie bahut achi hai",positive
"ye product bakwas hai",negative
"theek tha, nothing special",neutral
```

Labels can be `positive`, `neutral`, `negative` or `2`, `1`, `0`.

For a research experiment, use a properly licensed Hindi-English/code-mixed sentiment dataset and keep a held-out test set. A suitable public starting point is the Code-Mixed Sentiment Analysis dataset referenced in the project documentation. Do not commit a large dataset or private user text to GitHub.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
cd FedHinglish-X
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare data

Put your CSV at:

```text
data/hinglish_sentiment.csv
```

or set:

```powershell
$env:DATA_PATH="C:\path\to\hinglish_sentiment.csv"
```

## Centralized baseline

```bash
python train_centralized.py
```

This establishes a transformer baseline before federated training.

## Federated experiment

```bash
python train_federated.py
```

Default experiment:

- 8 clients
- 10 rounds
- 2 local epochs
- non-IID Dirichlet partitioning
- personalized LoRA adapters
- client update clipping
- optional Gaussian noise
- weighted asynchronous-style aggregation

For a quick CPU smoke run:

```powershell
$env:NUM_CLIENTS="3"
$env:ROUNDS="2"
$env:LOCAL_EPOCHS="1"
$env:MAX_TRAIN_SAMPLES="300"
python train_federated.py
```

## Evaluation

```bash
python evaluate.py
```

Outputs are written to `artifacts/`:

- global model
- client metrics
- confusion matrix
- training history
- communication statistics

## Web application

After training a model:

```bash
python app.py
```

Open `http://127.0.0.1:5000` and submit Hinglish/English/Hindi text.

## Research experiments

Run at least these ablations:

1. centralized MuRIL
2. FedAvg without personalization
3. personalized LoRA federation
4. personalized federation + DP
5. personalized federation + DP + asynchronous aggregation
6. low-resource clients
7. imbalanced clients
8. increasing code-mixing intensity
9. privacy budgets/noise levels

Report Accuracy, Macro-F1, per-client F1, personalization gap, client variance, convergence rounds, communication volume, and privacy parameters.

## Important reproducibility note

The federated simulator is deliberately written without requiring a real multi-machine deployment. It models independent clients locally so that the research can be reproduced on one workstation. For deployment, the same model/update logic can be moved to Flower; Flower currently provides PyTorch, Transformers, federated strategy and differential-privacy examples.
