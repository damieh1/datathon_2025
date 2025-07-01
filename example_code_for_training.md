## Fine-Tune a Transformer Model Using Colab

You can fine-tune a Hugging Face transformer model using your annotated dataset directly in **Google Colab**.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/damieh1/datathon_2025/blob/main/Fine_tune_HF_Model.ipynb)


---
## Use the Official Gold Standard Dataset

For Challenge #2 (Modeling & Evaluation), participants must use our pre-annotated dataset:

🔗 [Antisemitism on Twitter: A Dataset for Machine Learning and Text Analytics (Zenodo)](https://zenodo.org/records/14448399)

---

## Official Gold Standard Datasets

You must use one or both of the following annotated datasets to train and evaluate your antisemitism detection model.

---

### Dataset 1: Antisemitism on Twitter (2023–2024)

🔗 [Zenodo Record – Antisemitism on Twitter](https://zenodo.org/records/14448399)

| Column      | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `TweetID`   | Unique identifier for the tweet                                             |
| `Username`  | Account that published the tweet                                            |
| `Text`      | Full, unprocessed tweet text                                                |
| `CreateDate`| Date the tweet was posted                                                   |
| `Biased`    | Binary label indicating if tweet is antisemitic (`1`) or not (`0`)          |
| `Keyword`   | The keyword used in the query (can appear in text, hashtags, mentions, etc.)|

---

### Dataset 2: Trends in Antisemitism and Counter-Speech (Before & After Oct 7)

🔗 [Zenodo Record – Antisemitism on X (Oct 7 Analysis)](https://zenodo.org/records/15025646)

Includes all fields from Dataset 1, plus:

| Additional Column | Description                                                                              |
|-------------------|------------------------------------------------------------------------------------------|
| `CallingOut`      | Binary label indicating whether the tweet is actively calling out antisemitism (`1`)     |

---

### How to Use These Datasets

- Use `Text` as model input.
- Use `Biased` as the main label for antisemitism classification.
- If using Dataset 2, you may optionally explore `CallingOut` as a secondary task or control variable.
- You may choose either dataset or combine both — just be clear in your report how you used them.

> 💡 Tip: Use stratified train/test splits and document class distributions to ensure balanced evaluation.


### Instructions for Model Training

- Use the `Text` column as input for your model.
- Use the `Biased` column as the classification label (binary: `1` or `0`).
- You may split the data using an 80/20 train-test split or with cross-validation.
- You are free to choose any transformer model (see our recommendations [here](https://github.com/AnnotationPortal/DatathonandHackathon.github.io/blob/main/NLP-Tools%20and%20Guides.md#recommended-transformer-models-for-hate-speech--antisemitism-detection).

> ⚠️ Please do not modify or filter the labels unless otherwise justified in your report.

### What You’ll Need

1. **Choose a pretrained model from Hugging Face**  
   We recommend using one of the following:

   - [`cardiffnlp/twitter-roberta-base-offensive`](https://huggingface.co/cardiffnlp/twitter-roberta-base-offensive)
   - [`GroNLP/hateBERT`](https://huggingface.co/GroNLP/hateBERT)
   - [`vinai/bertweet-base`](https://huggingface.co/docs/transformers/en/model_doc/bertweet)
   - [`microsoft/mdeberta-v3-base`](https://huggingface.co/microsoft/mdeberta-v3-base)

---

### How It Works

- The Colab notebook will mount your Google Drive
- Load your `.csv` from the drive
- Tokenize your text using the selected model’s tokenizer
- Fine-tune the model using the Hugging Face `Trainer` API
- Evaluate on a test split (e.g., precision, recall, F1)

---
## Example: Fine-Tune a Transformer Model in Colab

This example shows how to fine-tune a Hugging Face transformer model using our official gold standard dataset (`Biased` column) as a binary classification task.

You can paste this into your own Colab notebook, or run it from our pre-built notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/damieh1/datathon_2025/blob/main/Fine_tune_HF_Model.ipynb)

---

### Dataset Requirements
- Must contain `Text` and `Biased` columns
- CSV should be uploaded to Google Drive
- Label values: `0` = non-antisemitic, `1` = antisemitic

---

### Training Script (Python in Colab)

```python
!pip install transformers datasets scikit-learn --quiet

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd

# Mount Google Drive to load the file
from google.colab import drive
drive.mount('/content/drive')

# Load the labeled dataset
df = pd.read_csv('/content/drive/MyDrive/your_dataset.csv')  # change this path
df = df[['Text', 'Biased']].dropna()
df['Biased'] = df['Biased'].astype(int)

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Biased'], random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Choose your model
model_name = "cardiffnlp/twitter-roberta-base-offensive"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
def preprocess(example):
    return tokenizer(example["Text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

# Training args for fine-tuning
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

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train!
trainer.train()

# Evaluate
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

---

> 💡 The goal is not only high performance — but also transparency and insight into the **limitations** of automated antisemitism detection.

---

📁 **Reminder:** Always download or save your model checkpoints if you plan to continue training later.  
