import pandas as pd
import numpy as np
import re
import evaluate
import wandb
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from pysentimiento.preprocessing import preprocess_tweet
import os

from sklearn.metrics import (
    precision_recall_fscore_support,
    jaccard_score,
    accuracy_score,
)

from transformers import EvalPrediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_PROJECT"] = "Tweets_clasificacion_ROBERTUITO"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_MODE"] = "online"

data = pd.read_parquet('./datos_pln/tweets.parquet', engine='pyarrow')
data = data[['partido', 'tweet']]

def limpiar_texto(texto):
    #eliminar caracteres
    texto_limpio = re.sub(r'http\S+|www\S+|@\w+|[^\w\s]', ' ', texto).lower().strip()
    texto_limpio = ' '.join(texto_limpio.split())
    texto_limpio = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ]+', ' ', texto_limpio)
    texto_limpio = ' '.join(texto_limpio.split())
    return texto_limpio

data['tweet_clean'] = data['tweet'].apply(limpiar_texto)

partidos = ['pp', 'psoe', 'ciudadanos', 'vox', 'podemos']
ptoi = {p: i for i, p in enumerate(partidos)}
itop = {v: k for k, v in ptoi.items()}

def apply_ptoi(label):
    return ptoi[label]

data['labels'] = data['partido'].apply(apply_ptoi)

#model_name = "pysentimiento/robertuito-base-uncased" # RUN 1
#model_name = "dccuchile/bert-base-spanish-wwm-uncased" # RUN 2
model_name = "PlanTL-GOB-ES/roberta-base-bne" # RUN 3

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,
    return_dict=True
).to(device)  # Enviar el modelo a la GPU


tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces = True)


def tokenize_text(text):
    preprocessed_text = preprocess_tweet(text)

    tok = tokenizer.tokenize(preprocessed_text)
    return tok

data['tweet_clean_token'] = data['tweet_clean'].apply(tokenize_text)

tweets = list(data['tweet_clean'])

tokenized_output = tokenizer(tweets, padding="max_length", truncation=True, max_length = 36)
input_ids = tokenized_output["input_ids"]
attention_mask = tokenized_output["attention_mask"]

data['input_ids'] = input_ids
data['attention_mask'] = attention_mask

dataset = Dataset.from_pandas(data)

train_test_split = dataset.train_test_split(test_size = 0.2, seed = 23)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

def compute_metrics_xuli(
    p: EvalPrediction
):
    """
    Compute metrics for a given prediction from HF Trainer,
    It takes:
     - An `EvalPrediction` object (a namedtuple with a predictions and label_ids field)
    Returns:
     - a dictionary string to float.
    """
    predictions, ref_labels = p.predictions, p.label_ids
    pred_labels = predictions.argmax(-1)
    #print(predictions)
    #print(ref_labels)
    """
    Compute the evaluation metrics for given pred_labels, references
      - Precision
      - Recall
      - F1
      - Jaccard accuracy
    Averaging={micro, macro, weighted}
    """

    results = {}
    ## Metrics with averaging
    for metric in [precision_recall_fscore_support, jaccard_score]:
        for average in ["micro", "macro", "weighted"]:
            if metric == precision_recall_fscore_support:
                precision, recall, f1, _ = metric(
                    ref_labels, pred_labels, average=average, zero_division=0
                )                
                results[f'{average}_precision'] = precision
                results[f'{average}_recall'] = recall
                results[f'{average}_f1'] = f1
            elif metric == jaccard_score:
                results['jaccard'] = metric(ref_labels, pred_labels, average=average)

    ## Metrics w/o averaging
    ### Accuracy: given (pred, ref), 1 if perfect match, 0 otherwise
    results['accuracy'] = accuracy_score(ref_labels, pred_labels)
    return results


training_args = TrainingArguments(
    run_name = "roberta_tardis_b4",
    output_dir = "tweets_roberta_run",
    do_eval = True,
    do_train = True,
    save_strategy="epoch",
    eval_strategy = "epoch",
    num_train_epochs = 10,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    group_by_length = True,
    remove_unused_columns = True,
    metric_for_best_model = 'macro_f1',
    save_total_limit = 2,
    load_best_model_at_end = True,
    greater_is_better = True,
    lr_scheduler_type = 'linear',
    seed = 23,
    report_to = 'wandb',
    logging_steps=50,  # how often to log to W&B

    #no_cuda=True, # en CPU
    
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics_xuli,
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience = 2, early_stopping_threshold = 0.0005)]  # Early stopping
)

print("Inicia entrenamiento")
trainer.train()

print("Inicia evaluación")
trainer.evaluate()
trainer.save_model("./saves/modelos_beto")

wandb.finish()
