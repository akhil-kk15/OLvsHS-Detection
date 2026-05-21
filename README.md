# Hate Speech vs Offensive Language Detection

This is a runnable starter implementation for a 3-class classifier:

- `hate_speech`
- `offensive_language`
- `neither`

It follows the course pipeline: preprocessing, BoW/TF-IDF features, Naive Bayes baseline, Logistic Regression model, precision/recall/F1 evaluation, and confusion matrix analysis.

## 1. Install Libraries

From `/home/akhilkk/Desktop/NLP`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## 2. Add Dataset

Download the Davidson dataset and place the CSV here:

```text
hate_offensive_detection/data/raw/labeled_data.csv
```

Expected Davidson columns:

```text
tweet,class
```

Davidson label mapping:

| Original class | System label |
|---:|---|
| `0` | `hate_speech` |
| `1` | `offensive_language` |
| `2` | `neither` |

## 3. Train

Run from `/home/akhilkk/Desktop/NLP`:

```bash
python3 hate_offensive_detection/src/train.py \
  --data hate_offensive_detection/data/raw/labeled_data.csv \
  --text-col tweet \
  --label-col class \
  --label-map davidson
```

The script trains:

- `CountVectorizer` + `MultinomialNB`
- word+character `TfidfVectorizer` + `LogisticRegression`

It chooses the best model by development-set macro F1.

## 4. Outputs

```text
hate_offensive_detection/models/best_model.joblib
hate_offensive_detection/models/results.txt
hate_offensive_detection/reports/figures/confusion_matrix.png
```

## 5. Predict

```bash
python3 hate_offensive_detection/src/predict.py \
  --text "I strongly disagree with this decision."
```

## 6. Different CSV Columns

If your dataset already has columns named `text` and `label` with labels exactly equal to `hate_speech`, `offensive_language`, and `neither`, use:

```bash
python3 hate_offensive_detection/src/train.py \
  --data hate_offensive_detection/data/raw/dataset.csv \
  --text-col text \
  --label-col label \
  --label-map none
```

