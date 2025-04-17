#!/usr/bin/env python3

"""
relation_extraction_pubmedbert.py

Fine-tune PubMedBERT for SDOH-biomedical relationship extraction with
an additional MLP head as described in the paper.

Usage:
    python relation_extraction_pubmedbert.py \
      --train_file train.json --val_file valid.json \
      --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
      --output_dir ./relation_model --epochs 3 --batch_size 16 --lr 2e-5
"""

import argparse
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer
)

logging.basicConfig(level=logging.INFO)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RelationExtractionModel(nn.Module):
    def __init__(self, model_name, num_labels, hidden_dim=256, dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size
        # MLP head: take two entity embeddings concatenated
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(bert_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        # input_ids: [batch, seq_len]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [batch, seq_len, bert_dim]
        # find marker token ids
        # assume tokenizer has added special tokens [E1], [E2]
        # marker ids in vocab
        e1_id = self.bert.config.tokenizer_class.from_pretrained(self.bert.name_or_path).convert_tokens_to_ids('[E1]')
        e2_id = self.bert.config.tokenizer_class.from_pretrained(self.bert.name_or_path).convert_tokens_to_ids('[E2]')
        # gather entity embeddings: take first occurrence in each example
        batch_size, seq_len, bert_dim = hidden.size()
        ent1 = []
        ent2 = []
        for i in range(batch_size):
            ids = input_ids[i]
            # find first marker positions
            e1_pos = (ids == e1_id).nonzero(as_tuple=True)[0]
            e2_pos = (ids == e2_id).nonzero(as_tuple=True)[0]
            # default to CLS if not found
            if len(e1_pos) == 0:
                e1_pos = torch.tensor([0], device=ids.device)
            if len(e2_pos) == 0:
                e2_pos = torch.tensor([0], device=ids.device)
            # take hidden at first marker
            ent1.append(hidden[i, e1_pos[0], :])
            ent2.append(hidden[i, e2_pos[0], :])
        ent1 = torch.stack(ent1)  # [batch, bert_dim]
        ent2 = torch.stack(ent2)
        x = torch.cat([ent1, ent2], dim=1)  # [batch, bert_dim*2]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.relu(x)
        logits = self.classifier(x)
        return logits


def tokenize_and_encode(examples, tokenizer, max_length):
    # insert markers
    texts = []
    for text, e1, e2 in zip(examples['text'], examples['entity1'], examples['entity2']):
        # wrap entities
        # ensure e1 appears first
        t_lower = text.lower()
        i1 = t_lower.find(e1.lower())
        i2 = t_lower.find(e2.lower())
        if i2 < i1:
            # swap
            e1, e2 = e2, e1
            i1, i2 = i2, i1
        # insert markers
        marked = text[:i1] + '[E1]' + text[i1:i1+len(e1)] + '[/E1]' + \
                 text[i1+len(e1):i2] + '[E2]' + text[i2:i2+len(e2)] + '[/E2]' + text[i2+len(e2):]
        texts.append(marked)
    # tokenize
    enc = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length
    )
    enc['labels'] = [label2id[l] for l in examples['label']]
    return enc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    parser.add_argument('--output_dir', type=str, default='./relation_model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # load data
    train_json = json.load(open(args.train_file))
    if args.val_file:
        val_json = json.load(open(args.val_file))
    else:
        # split
        random.shuffle(train_json)
        split = int(0.8 * len(train_json))
        val_json = train_json[split:]
        train_json = train_json[:split]

    # label mapping
    labels = sorted({ex['label'] for ex in train_json + val_json})
    global label2id; label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}

    # datasets
    train_ds = Dataset.from_list(train_json)
    val_ds = Dataset.from_list(val_json)
    dataset = DatasetDict({'train': train_ds, 'validation': val_ds})

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # add markers
    tokenizer.add_special_tokens({'additional_special_tokens': ['[E1]','[/E1]','[E2]','[/E2]']})

    # map
    dataset = dataset.map(
        lambda ex: tokenize_and_encode(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=['text','entity1','entity2','label']
    )

    # PrimeKGIntegration
    num_labels = len(labels)
    model = RelationExtractionModel(args.model_name, num_labels)
    model.bert.resize_token_embeddings(len(tokenizer))

    # metrics
    accuracy = load_metric('accuracy')
    f1 = load_metric('f1')
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {'accuracy': accuracy.compute(predictions=preds, references=p.label_ids)['accuracy'],
                'f1': f1.compute(predictions=preds, references=p.label_ids, average='weighted')['f1']}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.lr,
        load_best_model_at_end=True,
        metric_for_best_model='f1'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    # plot loss
    import matplotlib.pyplot as plt
    history = trainer.state.log_history
    steps = [h['step'] for h in history if 'loss' in h]
    losses = [h['loss'] for h in history if 'loss' in h]
    plt.plot(steps, losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f"{args.output_dir}/loss.png")
    print("Done.")
