import pandas as pd
import pickle
import numpy as np
import torch

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments


# Initialize variables and data structures
DATASET_NAME = "haspeede"
class_weights = {}
df_raw, df_train, df_valtest, df_val, df_test = {}, {}, {}, {}, {}
train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = {}, {}, {}, {}, {}, {}
train_encodings, val_encodings, test_encodings = {}, {}, {}
train_dataset, val_dataset, test_dataset = {}, {}, {}
training_args, trainer = {}, {}

# PyTorch dataset class
class HateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Trainer class with weighted loss for skewed distributions
class WeightedTrainerHaSpeeDe(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        weighted_loss = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights[DATASET_NAME])).to(device)
        return weighted_loss(logits, labels)

# Function for model initialization
def model_init():
    model = BertForSequenceClassification.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
    model.resize_token_embeddings(len(tokenizer))
    return model

# Load the haspeede dataset and write it to a dictionary
training_data = pd.read_pickle('data/training/haspeede.pkl')
df_train[DATASET_NAME] = training_data[DATASET_NAME + ".train"].copy()
df_val[DATASET_NAME] = training_data[DATASET_NAME + ".dev"].copy()
df_test[DATASET_NAME] = training_data[DATASET_NAME + ".test"].copy()

# Divide dataframe columns into series
train_texts[DATASET_NAME] = df_train[DATASET_NAME].text.astype("string").tolist()
val_texts[DATASET_NAME] = df_val[DATASET_NAME].text.astype("string").tolist()
test_texts[DATASET_NAME] = df_test[DATASET_NAME].text.astype("string").tolist()
train_labels[DATASET_NAME] = df_train[DATASET_NAME].label.tolist()
val_labels[DATASET_NAME] = df_val[DATASET_NAME].label.tolist()
test_labels[DATASET_NAME] = df_test[DATASET_NAME].label.tolist()
    
# Compute the dataset class weights based on the skewed label distribution in the training data
class_weights[DATASET_NAME] = compute_class_weight(
    "balanced", classes=np.unique(train_labels[DATASET_NAME]), y=train_labels[DATASET_NAME])
# print(f"Class weights: {class_weights}.")

# Load the BERT tokenizer, add special tokens to it, and tokenize the text series
tokenizer = BertTokenizerFast.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0", model_max_length=512)
special_tokens_dict = {'additional_special_tokens': ['[USER]','[EMOJI]','[URL]','<user>','<url>','<money>','<time>','<date>','<number>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
train_encodings[DATASET_NAME] = tokenizer(
    train_texts[DATASET_NAME], truncation=True, padding=True)
val_encodings[DATASET_NAME] = tokenizer(
    val_texts[DATASET_NAME], truncation=True, padding=True)
test_encodings[DATASET_NAME] = tokenizer(
    test_texts[DATASET_NAME], truncation=True, padding=True)

# Create the PyTorch HateDataset objects for the splits
train_dataset[DATASET_NAME] = HateDataset(
    train_encodings[DATASET_NAME], train_labels[DATASET_NAME])
val_dataset[DATASET_NAME] = HateDataset(
    val_encodings[DATASET_NAME], val_labels[DATASET_NAME])
test_dataset[DATASET_NAME] = HateDataset(
    test_encodings[DATASET_NAME], test_labels[DATASET_NAME])

# Check for the GPU availability, o.w. set the CPU as the preferred device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the training arguments
training_args[DATASET_NAME] = TrainingArguments(
    save_steps=2500,
    output_dir="./model/BERT_haspeede/checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy='epoch',
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    seed=123
)

# Create the weighted trainer object
trainer[DATASET_NAME] = WeightedTrainerHaSpeeDe(                        
    args=training_args[DATASET_NAME],                  
    train_dataset=train_dataset[DATASET_NAME],         
    eval_dataset=val_dataset[DATASET_NAME],            
    model_init=model_init
)

# Fine-tune weighted BERT on the dataset
# print(f"Fine-tuning a weighted BERT model on {DATASET_NAME} data.")
trainer[DATASET_NAME].train()

# Save the model and the tokenizer
trainer[DATASET_NAME].save_model("./model/BERT_haspeede")
tokenizer.save_pretrained("./model/BERT_haspeede")

# Re-load the fine-tuned model and instantiate the trainer
models, trainer, results, pred_labels = {}, {}, {}, {}
models[DATASET_NAME] = BertForSequenceClassification.from_pretrained("./model/BERT_haspeede")
trainer[DATASET_NAME] = Trainer(
    model=models[DATASET_NAME],         
    args=TrainingArguments(
        output_dir="./model/BERT_haspeede/test",
        per_device_eval_batch_size=64)
)

# Predict and evaluate on test set data
# results[DATASET_NAME] = trainer[DATASET_NAME].predict(test_dataset[DATASET_NAME])
# for metric in results[DATASET_NAME].metrics:
#     print(metric, results[DATASET_NAME].metrics['{}'.format(metric)])
# print()

# preds=[]
# for row in results[DATASET_NAME][0]:
#     preds.append(int(np.argmax(row)))
# pred_labels[DATASET_NAME] = pd.Series(preds)

# for average in ["micro", "macro"]:
#     print('{} F1 score: {:.2%}'.format(average, f1_score(
#         test_labels[DATASET_NAME], pred_labels[DATASET_NAME], average=average)))
# print()
