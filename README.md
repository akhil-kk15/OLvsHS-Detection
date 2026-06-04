# Hate Speech vs Offensive Language Detection

This project is a supervised NLP text-classification system for detecting whether a short text is:

- `hate_speech`
- `offensive_language`
- `neither`

It includes preprocessing, TF-IDF feature extraction, classic machine-learning classifiers, evaluation metrics, a saved model, command-line prediction, and a Streamlit web interface.

## Project Structure

```text
NLP_mini-Project/
  data/
    labeled_data.csv
  models/
    best_model.joblib
    results.txt
  reports/
    figures/
      confusion_matrix.png
  src/
    app.py
    lexiconfeatures.py
    model.py
    predict.py
    preprocess.py
    train.py
  requirements.txt
  README.md
```

Generated runtime files:

```text
prediction_log.txt
```

`prediction_log.txt` is created when users submit text through the Streamlit app.

## Dataset

The main dataset used here is the Davidson Hate Speech and Offensive Language dataset.

Expected CSV location:

```text
data/labeled_data.csv
```

Expected columns:

```text
tweet,class
```

Davidson label mapping:

| Original class | Project label |
|---:|---|
| `0` | `hate_speech` |
| `1` | `offensive_language` |
| `2` | `neither` |

The dataset file may not be included in the GitHub repository if it is large or license-restricted. If it is missing, download it separately and place it at `data/labeled_data.csv`.

## Setup

Clone the repository and move into the project folder:

```bash
git clone <your-repository-url>
cd NLP_mini-Project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` causes version issues on another machine, install the main libraries manually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
```

## Train The Model

Run this from the repository root:

```bash
python src/train.py \
  --data data/labeled_data.csv \
  --text-col tweet \
  --label-col class \
  --label-map davidson \
  --models-dir models \
  --figures-dir reports/figures
```

One-line version:

```bash
python src/train.py --data data/labeled_data.csv --text-col tweet --label-col class --label-map davidson --models-dir models --figures-dir reports/figures
```

The training script:

- loads the dataset
- maps labels to `hate_speech`, `offensive_language`, and `neither`
- preprocesses text
- splits data into train, development, and test sets
- trains baseline and improved classifiers
- chooses the best model by development macro F1
- evaluates on the test set
- saves the model and evaluation artifacts

Expected outputs:

```text
models/best_model.joblib
models/results.txt
reports/figures/confusion_matrix.png
```

## Predict From The Command Line

After training, run:

```bash
python src/predict.py \
  --model models/best_model.joblib \
  --text "I strongly disagree with this decision."
```

One-line version:

```bash
python src/predict.py --model models/best_model.joblib --text "I strongly disagree with this decision."
```

Example output:

```text
Prediction: neither
hate_speech: 0.0920
neither: 0.7281
offensive_language: 0.1799
```

Probability values will differ if the model is retrained.

## Run The Streamlit App

After `models/best_model.joblib` exists, start the app:

```bash
streamlit run src/app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Paste text into the textbox and click **Check text**.

The app shows:

- predicted class
- confidence/probability scores when available
- a saved log entry for each submitted text

## Prediction Logging

The Streamlit app writes predictions to:

```text
prediction_log.txt
```

This file is saved in the project root.

Each entry includes:

- timestamp
- input text
- predicted label
- display label
- hate-speech threshold
- class probabilities

View the log:

```bash
cat prediction_log.txt
```

Example log entry:

```text
================================================================================
Timestamp: 2026-06-04 12:30:00
Input text: I strongly disagree with this decision.
Predicted label: neither
Display label: Neither / Clean
Hate threshold: 0.5
Probabilities:
  - neither: 0.7281
  - offensive_language: 0.1799
  - hate_speech: 0.0920
```

## Recommended Run Order

Use this order when running the project from scratch:

```bash
cd NLP_mini-Project
source .venv/bin/activate

python src/train.py --data data/labeled_data.csv --text-col tweet --label-col class --label-map davidson --models-dir models --figures-dir reports/figures

python src/predict.py --model models/best_model.joblib --text "I strongly disagree with this decision."

streamlit run src/app.py
```

## Testing Examples

Neutral examples:

```bash
python src/predict.py --model models/best_model.joblib --text "I strongly disagree with this decision."
python src/predict.py --model models/best_model.joblib --text "The weather is nice today."
```

Offensive-language examples:

```bash
python src/predict.py --model models/best_model.joblib --text "You are stupid."
python src/predict.py --model models/best_model.joblib --text "Shut up, you idiot."
```

Identity-targeted examples:

```bash
python src/predict.py --model models/best_model.joblib --text "I hate all immigrants."
python src/predict.py --model models/best_model.joblib --text "Women should not be allowed to speak here."
```

Model predictions may not always match human judgment. Short insults are especially likely to be confused with hate speech or offensive language.

## Evaluation

The project uses:

- accuracy
- macro F1
- per-class precision, recall, and F1
- confusion matrix

Macro F1 is important because the dataset is imbalanced. The hate-speech class is much smaller than the offensive-language class.

After training, check:

```bash
cat models/results.txt
```

Open the confusion matrix:

```text
reports/figures/confusion_matrix.png
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'joblib'`

You are probably using the wrong Python interpreter. Activate the virtual environment:

```bash
source .venv/bin/activate
```

Then run:

```bash
python -c "import sys; print(sys.executable)"
```

It should point to `.venv/bin/python`.

### `train.py: error: the following arguments are required: --data`

Make sure there is a space after `--data`:

```bash
python src/train.py --data data/labeled_data.csv --text-col tweet --label-col class --label-map davidson
```

Incorrect:

```bash
python src/train.py --data/labeled_data.csv
```

### Streamlit Cannot Find The Model

Train the model first:

```bash
python src/train.py --data data/labeled_data.csv --text-col tweet --label-col class --label-map davidson --models-dir models --figures-dir reports/figures
```

Then run:

```bash
streamlit run src/app.py
```

### Streamlit Is Still Showing Old Errors

Stop the old app process with `Ctrl+C`, then restart:

```bash
streamlit run src/app.py
```

## Limitations

This is a course-level hate/offensive-language detection system, not a production moderation tool.

Known limitations:

- Offensive insults may be misclassified as hate speech.
- Hate speech can be implicit and difficult to detect.
- The model depends strongly on the dataset's annotation style.
- Social-media text contains slang, misspellings, sarcasm, and context gaps.
- The model should not be used as the only basis for real moderation decisions.

## Notes For GitHub

Before pushing, consider whether to include large or sensitive generated files.

Common files to exclude in `.gitignore`:

```text
.venv/
venv/
__pycache__/
*.pyc
prediction_log.txt
```

Depending on dataset license and file size, you may also need to exclude:

```text
data/labeled_data.csv
models/best_model.joblib
```

If you exclude the dataset or model, mention in the README how users can recreate them by downloading the dataset and running `src/train.py`.
