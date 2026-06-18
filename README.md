# Hate Speech vs Offensive Language Detection

> Warning: this project works with highly offensive text, including racist, sexist, homophobic, and otherwise abusive language. The models are imperfect, and short insults or context-dependent sentences can still be misclassified.

This repository detects one of three labels in short social-media text:

- `hate_speech`
- `offensive_language`
- `neither`

It includes:

- a flat classical baseline in [src/train.py](src/train.py)
- a two-stage hierarchical model in [src/train_hierarchical.py](src/train_hierarchical.py)
- a transformer model in [src/train_transformer.py](src/train_transformer.py)
- a Streamlit demo in [src/app.py](src/app.py)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/train_hierarchical.py --data-dir data --models-dir models
streamlit run src/app.py
```

If you want the transformer instead of the hierarchical model, train `src/train_transformer.py` first and then use the app.

## Models

### Flat baseline

- TF-IDF word n-grams
- TF-IDF character n-grams
- lexicon features
- logistic regression and Naive Bayes candidates

### Two-stage hierarchical model

- stage 1: `harmful` vs `neither`
- stage 2: `hate_speech` vs `offensive_language`
- TF-IDF features + lexicon features + logistic regression

### Transformer model

- fine-tuned encoder model such as DistilBERT
- direct 3-class prediction
- optional external rows and calibration rows during training

## Results

Current logs show that the stronger models are clearly better than the flat baseline.

| Model | Accuracy | Macro F1 | Notes |
|---|---:|---:|---|
| Flat logistic-regression baseline | 0.8569 | 0.7259 | Classical baseline |
| Two-stage hierarchical model | 0.8983 | 0.7427 | Best classical model in the repo |
| DistilBERT transformer | 0.9102 | 0.7541 | Best overall run in the shared logs |

The weakest class across all models is usually `hate_speech`, especially on recall.

## Repository Layout

```text
data/
logs/
models/
reports/figures/
src/
```

Key scripts:

- [src/train.py](src/train.py) trains the classical baseline.
- [src/train_hierarchical.py](src/train_hierarchical.py) trains the two-stage model.
- [src/train_transformer.py](src/train_transformer.py) fine-tunes a transformer classifier.
- [src/predict.py](src/predict.py) predicts with the saved classical or hierarchical bundle.
- [src/transformer_predict.py](src/transformer_predict.py) predicts with a transformer checkpoint.
- [src/app.py](src/app.py) runs the Streamlit UI.

## Data Used

The main dataset is the Davidson hate speech and offensive language corpus.

### Core files

- `data/labeled_data.csv`
  - main Davidson-style dataset used by the flat baseline and the transformer
- `data/olid-training-v1.0.tsv`
  - OLID / OffensEval training file used by the hierarchical model
- `data/testset-levela.tsv` and `data/labels-levela.csv`
  - OLID / OffensEval level-A test data used by the hierarchical model
- `data/test_a_tweets_all.tsv` and `data/test_a_labels_all.csv`
  - additional project-mapped level-A files used by the hierarchical model
- `data/external_gate_project_mapped.csv`
  - project-processed extra rows for the gate classifier
- `data/external_severity_project_mapped.csv`
  - project-processed extra rows for the severity classifier
- `data/manual_calibration_project_mapped.csv`
  - small manually curated calibration set for hard edge cases

### Label mapping

| Original class | Project label |
|---:|---|
| `0` | `hate_speech` |
| `1` | `offensive_language` |
| `2` | `neither` |

The transformer scripts use the same project labels so results stay comparable with the classical baseline.

## Citations


- Davidson et al. 2017, *Automated Hate Speech Detection and the Problem of Offensive Language*
  - https://arxiv.org/abs/1703.04009
- Zampieri et al. 2019, *SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval)*
  - https://arxiv.org/abs/1903.08983
- Sanh et al. 2019, *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*
  - https://arxiv.org/abs/1910.01108
- Caselli et al. 2020, *HateBERT: Retraining BERT for Abusive Language Detection in English*
  - https://arxiv.org/abs/2010.12472

The project-specific mapped CSV files in `data/` are derived and curated within this repository.

## Setup

Use Python 3.10+ if possible.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you are on a CPU-only machine and `torch` pulls CUDA packages, install the CPU wheel for PyTorch instead of a CUDA wheel.

## How To Use The Project

Run all commands from the repository root.

### Train the flat baseline

```bash
python src/train.py \
  --data data/labeled_data.csv \
  --text-col tweet \
  --label-col class \
  --label-map davidson \
  --models-dir models \
  --figures-dir reports/figures
```

Outputs:

- `models/best_model.joblib`
- `reports/figures/confusion_matrix.png`
- timestamped `logs/results_*.txt`

### Train the two-stage hierarchical model

```bash
python src/train_hierarchical.py \
  --data-dir data \
  --models-dir models
```

Outputs:

- `models/hierarchical_model.joblib`
- timestamped `logs/hierarchical_results_*.txt`

Optional flags:

- `--no-external-data`
- `--no-calibration-data`
- `--calibration-weight 10`

### Train a transformer model

```bash
python src/train_transformer.py \
  --data data/labeled_data.csv \
  --text-col tweet \
  --label-col class \
  --label-map davidson \
  --model-name distilbert-base-uncased \
  --output-dir models/distilbert_hsvol
```

Outputs:

- `models/distilbert_hsvol/`
- `models/distilbert_hsvol/training_metadata.json`
- `models/distilbert_hsvol/metrics.json`
- timestamped `logs/transformer_results_*.txt`

Optional flags:

- `--external-gate-data data/external_gate_project_mapped.csv`
- `--external-severity-data data/external_severity_project_mapped.csv`
- `--calibration-data data/manual_calibration_project_mapped.csv`
- `--calibration-weight 10`
- `--no-external-data`
- `--no-calibration-data`

### Predict one text

Classical or hierarchical bundle:

```bash
python src/predict.py \
  --model models/hierarchical_model.joblib \
  --text "I strongly disagree with this decision."
```

Transformer checkpoint:

```bash
python src/transformer_predict.py \
  --model-dir models/distilbert_hsvol \
  --text "I strongly disagree with this decision."
```

### Run the Streamlit demo

```bash
streamlit run src/app.py
```

The app lets you choose between:

- `Logistic regression (2 stage, baseline)`
- `DistilBERT`
- `HateBERT`

It writes prediction logs to timestamped files in `logs/` and shows the entered text plus probabilities in the UI. You do not need to retrain every time you open the app.

## Evaluation

The project reports:

- accuracy
- macro F1
- per-class precision, recall, and F1
- confusion matrix

Macro F1 matters because the classes are imbalanced and `hate_speech` is much rarer than `offensive_language`.

When you retrain, compare models on the same split and inspect:

- overall accuracy
- macro F1
- `hate_speech` recall and F1
- the confusion matrix for class imbalance errors

## Generated Files

Generated artifacts are intentionally kept out of version control where possible.

Common files and folders that should stay untracked:

- `.venv/`
- `venv/`
- `__pycache__/`
- `*.pyc`
- `logs/`
- `prediction_log.txt`
- `models/best_model.joblib`
- `models/hierarchical_model.joblib`
- `models/distilbert_hsvol/`
- `models/hatebert_hsvol/`
- `models/*/checkpoints/`

The `data/` directory also contains legacy or duplicate files that are not required by the current code. They are ignored through `.gitignore` and are not needed for a normal run.

## References

- [Davidson et al. 2017](https://arxiv.org/abs/1703.04009)
- [OffensEval / OLID 2019](https://arxiv.org/abs/1903.08983)
- [DistilBERT 2019](https://arxiv.org/abs/1910.01108)
- [HateBERT 2020](https://arxiv.org/abs/2010.12472)
