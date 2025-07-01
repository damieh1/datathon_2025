## 🐍 Example Code for Model Training

This guide walks you through how to fine-tune a Hugging Face transformer model (e.g., RoBERTa, BERTweet, HateBERT) using our official annotated datasets for antisemitism detection.  
You'll use Google Colab to train and evaluate a classifier on tweets labeled as antisemitic or non-antisemitic.

---

### Guide Agenda

- [Open in Colab](#fine-tune-a-transformer-model-using-colab)
- [Dataset Overview](#official-gold-standard-datasets)
- [Training Requirements](#dataset-requirements)
- [Example Training Code](#training-script-python-in-colab)
- [Evaluation Guidelines](#model-evaluation-what-you-must-submit)

---

## Fine-Tune a Transformer Model Using Colab

You can fine-tune a Hugging Face transformer model using your annotated dataset directly in **Google Colab**.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/damieh1/datathon_2025/blob/main/example_code_challenge_2.ipynb)

---

## Official Gold Standard Datasets

### Dataset 1: Antisemitism on Twitter (2023–2024)  
[Zenodo Record – Antisemitism on Twitter](https://zenodo.org/records/14448399)

### Dataset 2: Trends in Antisemitism and Counter-Speech (Before & After Oct 7)  
[Zenodo Record – Antisemitism on X (Oct 7 Analysis)](https://zenodo.org/records/15025646)


<summary> Dataset Fields</summary>

| Column      | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `TweetID`   | Unique identifier for the tweet                                             |
| `Username`  | Account that published the tweet                                            |
| `Text`      | Full, unprocessed tweet text                                                |
| `CreateDate`| Date the tweet was posted                                                   |
| `Biased`    | Binary label indicating if tweet is antisemitic (`1`) or not (`0`)          |
| `Keyword`   | The keyword used in the query (text, mentions, or username)                 |
| `CallingOut`| (Only in Dataset 2) Whether the tweet is reporting/calling out antisemitism |


---

## Dataset Requirements

- File must contain `Text` and `Biased` columns
- Label values must be binary: `0` = not antisemitic, `1` = antisemitic
- File should be uploaded to your Google Drive

---

## Training Script (Python in Colab)

```python
!pip install transformers datasets scikit-learn --quiet

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/your_dataset.csv')  # change path

# Basic cleaning
df = df[['Text', 'Biased']].dropna()
df['Biased'] = df['Biased'].astype(int)

# Split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Biased'], random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Model choice
model_name = "cardiffnlp/twitter-roberta-base-offensive"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def preprocess(example):
    return tokenizer(example["Text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

# Training Arguments and Hyperparameters for fine-tuning
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)
```

---

## Model Evaluation: What You Must Submit

Training a model is only part of the challenge — careful **evaluation and interpretation** are essential.

Your submission must include the following:

### Required Evaluation Metrics
- **Precision**, **Recall**, and **F1-Score** for both classes (antisemitic / non-antisemitic)
- A **confusion matrix** to show the distribution of predictions
- (Optional but recommended) A brief explanation of class imbalance handling, if applicable

### Error Analysis
- Describe at least **3–5 examples** of tweets your model misclassified.
- Explain possible reasons (e.g., satire, irony, subtle references, poor keyword fit).
- Reflect on where your model struggles and what might improve performance.

### Hyperparameters
- Document your training setup:
  - Model used (e.g., `twitter-roberta-base-offensive`)
  - Learning rate, number of epochs, batch size, weight decay, etc.
  - Train/test split ratio and random seed

> 💡 The goal is not only high performance — but also transparency and insight into the **limitations** of automated antisemitism detection.

---

> See our model recommendations: [Recommended Models](https://github.com/AnnotationPortal/DatathonandHackathon.github.io/blob/main/NLP-Tools%20and%20Guides.md#recommended-transformer-models-for-hate-speech--antisemitism-detection)

---

📁 **Reminder:** Always download or save your model checkpoints if you plan to continue training later.  
