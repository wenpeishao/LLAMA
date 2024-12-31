#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    LlamaTokenizer,
    LlamaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.calibration import calibration_curve
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the comments dataset
df = pd.read_parquet('comb_comments.parquet')
df = df.dropna(subset=['body'])
df = df.sample(n=1000000, random_state=42)  # Set random_state for reproducibility

# Encode labels
label_encoder = LabelEncoder()
df['subreddit_encoded'] = label_encoder.fit_transform(df['subreddit'])
NUM_LABELS = df['subreddit_encoded'].nunique()

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    df['body'], df['subreddit_encoded'], test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

local_dir = "./llama_model"

if not os.path.exists(local_dir):
    raise FileNotFoundError(f"The directory {local_dir} does not exist. Ensure the model is saved locally.")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(local_dir)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    local_dir,
    num_labels=NUM_LABELS,
    torch_dtype=torch.bfloat16,  # or torch.float16 depending on hardware support
)

model.to(device)
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenization function
def tokenize_function(texts):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

# Tokenize data
train_encodings = tokenize_function(X_train.tolist())
val_encodings = tokenize_function(X_val.tolist())
test_encodings = tokenize_function(X_test.tolist())

train_labels = torch.tensor(y_train.values)
val_labels = torch.tensor(y_val.values)
test_labels = torch.tensor(y_test.values)

train_dataset = TensorDataset(
    train_encodings['input_ids'], train_encodings['attention_mask'], train_labels
)
val_dataset = TensorDataset(
    val_encodings['input_ids'], val_encodings['attention_mask'], val_labels
)
test_dataset = TensorDataset(
    test_encodings['input_ids'], test_encodings['attention_mask'], test_labels
)

batch_size = 4
train_dataloader = DataLoader(
    train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
)
val_dataloader = DataLoader(
    val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size
)
test_dataloader = DataLoader(
    test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size
)

optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# Lists to store metrics
train_losses = []
val_accuracies = []
best_val_accuracy = 0.0

for epoch in range(epochs):
    # Training
    model.train()
    total_train_loss = 0

    for batch in train_dataloader:
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_attention_mask,
            labels=b_labels
        )

        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_eval_accuracy = 0

    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels
            )

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        total_eval_accuracy += (preds == b_labels).cpu().numpy().mean()

    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    val_accuracies.append(avg_val_accuracy)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Average Training Loss: {avg_train_loss}")
    print(f"Validation Accuracy: {avg_val_accuracy}")

    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        output_dir = './best_llama_model'
        os.makedirs(output_dir, exist_ok=True)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Saved Best Model to best_llama_model directory")

print("Training complete!")

# Save results to CSV
results_df = pd.DataFrame({
    'epoch': list(range(1, epochs + 1)),
    'train_loss': train_losses,
    'val_accuracy': val_accuracies
})
results_df.to_csv('training_results.csv', index=False)
print("Saved training metrics to training_results.csv")

# -------------------- Evaluation on Test Set with Best Model --------------------
best_model_dir = './best_llama_model'
best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
best_model.to(device)
best_model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        outputs = best_model(
            input_ids=b_input_ids,
            attention_mask=b_attention_mask
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(b_labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Compute Macro-average AUC
auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
print(f"Macro-average AUC: {auc_score}")

# Save predictions and probabilities to CSV
prob_columns = [f'prob_class_{i}' for i in range(NUM_LABELS)]
test_results_df = pd.DataFrame({
    'true_label': all_labels,
    'predicted_label': all_preds,
})
for i in range(NUM_LABELS):
    test_results_df[prob_columns[i]] = all_probs[:, i]

test_results_df.to_csv('test_predictions_with_probs.csv', index=False)
print("Saved test predictions and probabilities to test_predictions_with_probs.csv")

# Generate calibration data (one-vs-rest for each class)
calibration_data = []
for class_id in range(NUM_LABELS):
    class_probs = all_probs[:, class_id]
    class_true = (all_labels == class_id).astype(int)
    fraction_pos, mean_pred_value = calibration_curve(class_true, class_probs, n_bins=10, strategy='uniform')
    calib_df = pd.DataFrame({
        'class_id': class_id,
        'fraction_of_positives': fraction_pos,
        'mean_predicted_value': mean_pred_value
    })
    calibration_data.append(calib_df)

calibration_all_classes_df = pd.concat(calibration_data, ignore_index=True)
calibration_all_classes_df.to_csv('calibration_data.csv', index=False)
print("Saved calibration data to calibration_data.csv")

# Save AUC and other metrics
metrics_df = pd.DataFrame({
    'metric': ['macro_auc'],
    'value': [auc_score]
})
metrics_df.to_csv('test_metrics.csv', index=False)
print("Saved AUC metric to test_metrics.csv")

print("Done! You can now plot calibration curves and AUC using the saved CSV files.")
