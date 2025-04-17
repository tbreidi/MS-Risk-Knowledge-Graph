#!/usr/bin/env python3

"""
Fine-tune PubMedBERT for SDoH-biomedical relationship extraction with
an additional MLP head and co-occurrence score integration.

Usage:
    python RelationExtractionCooccurence.py \
      --articles abstracts.json \
      --train train.json --val valid.json \
      --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
      --corpus abstracts.json \
      --output_dir ./relation_model --epochs 3 --batch_size 16 --lr 2e-5
"""

import argparse
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter, defaultdict
from datasets import Dataset, DatasetDict
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer
)

# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------
class RelationExtractionModel(nn.Module):
    def __init__(self, model_name, num_labels, hidden_dim=256, dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(bert_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask, e1_pos=None, e2_pos=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        # gather entities at marker positions
        batch_size, seq_len, dim = hidden.size()
        ent1 = hidden[torch.arange(batch_size), e1_pos]
        ent2 = hidden[torch.arange(batch_size), e2_pos]
        x = torch.cat([ent1, ent2], dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.relu(x)
        logits = self.classifier(x)
        return logits

# ----------------------------------------------------------------------------
# Co-occurrence Computation
# ----------------------------------------------------------------------------
def compute_cooccurrence(corpus, entity_pairs):
    """
    Count co-occurrence and individual frequencies for entity pairs.
    corpus: list of abstracts (strings)
    entity_pairs: list of (e1, e2) tuples
    Returns dict of composite scores.
    """
    # count occurrences
    freq = Counter()
    joint = Counter()
    for abstract in corpus:
        text = abstract.lower()
        for e1, e2 in entity_pairs:
            if e1 in text: freq[e1] += text.count(e1)
            if e2 in text: freq[e2] += text.count(e2)
            if e1 in text and e2 in text:
                # simple sentence-level count
                joint[e1, e2] += sum(1 for sent in text.split('.') if e1 in sent and e2 in sent)
    coocc = {}
    for e1, e2 in entity_pairs:
        nA = freq[e1]
        nB = freq[e2]
        nJoint = joint[e1, e2]
        if nA and nB:
            coocc[(e1,e2)] = nJoint / np.sqrt(nA * nB)
        else:
            coocc[(e1,e2)] = 0.0
    return coocc

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--articles', type=str, required=True, help='JSON of abstracts')
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--val', type=str)
    parser.add_argument('--model_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    parser.add_argument('--corpus', type=str, required=True, help='Same as articles for co-occ')
    parser.add_argument('--output_dir', type=str, default='./relation_model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()

    # load abstracts for co-occ
    abstracts = json.loads(Path(args.corpus).read_text())

    # load train/val data
    train_data = json.loads(Path(args.train).read_text())
    if args.val:
        val_data = json.loads(Path(args.val).read_text())
    else:
        np.random.shuffle(train_data)
        split = int(0.8 * len(train_data))
        val_data = train_data[split:]
        train_data = train_data[:split]

    # build entity_pairs for co-occ
    pairs = [(ex['entity1'].lower(), ex['entity2'].lower()) for ex in train_data + val_data]
    coocc_scores = compute_cooccurrence(abstracts, pairs)

    # label mapping
    labels = sorted({ex['label'] for ex in train_data + val_data})
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}
    for ex in train_data + val_data:
        ex['label'] = label2id[ex['label']]

    # prepare datasets
    from datasets import Dataset, DatasetDict
    ds = DatasetDict({'train': Dataset.from_list(train_data), 'validation': Dataset.from_list(val_data)})

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<e1>','</e1>','<e2>','</e2>']})

    # tokenization and marker positions
    def preprocess(batch):
        texts, e1pos, e2pos = [], [], []
        for text, e1, e2 in zip(batch['text'], batch['entity1'], batch['entity2']):
            # insert markers and record positions
            low = text.lower()
            i1 = low.find(e1.lower())
            i2 = low.find(e2.lower())
            if i2 < i1: i1, i2 = i2, i1; e1, e2 = e2, e1
            marked = text[:i1] + '<e1>' + text[i1:i1+len(e1)] + '</e1>' + text[i1+len(e1):i2] + '<e2>' + text[i2:i2+len(e2)] + '</e2>' + text[i2+len(e2):]
            enc = tokenizer(marked, padding='max_length', truncation=True, max_length=args.max_length, return_offsets_mapping=True)
            # find offsets for e1 and e2 tokens
            offsets = enc.pop('offset_mapping')
            ids = enc['input_ids']
            # find token positions
            pos1 = next(i for i,(s,e) in enumerate(offsets) if s>0 and marked[s-3:s] == '<e1>')
            pos2 = next(i for i,(s,e) in enumerate(offsets) if s>0 and marked[s-4:s] == '<e2>')
            texts.append(enc)
            e1pos.append(pos1)
            e2pos.append(pos2)
        batch_out = {k: [t[k] for t in texts] for k in texts[0]}
        batch_out['e1_pos'] = e1pos
        batch_out['e2_pos'] = e2pos
        batch_out['labels'] = batch['label']
        return batch_out

    ds = ds.map(preprocess, batched=True, remove_columns=['text','entity1','entity2','label'])

    # model init
    model = RelationExtractionModel(args.model_name, len(labels))
    model.bert.resize_token_embeddings(len(tokenizer))

    # metrics
    accuracy = load_metric('accuracy')
    f1 = load_metric('f1')
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            'accuracy': accuracy.compute(predictions=preds, references=p.label_ids)['accuracy'],
            'f1': f1.compute(predictions=preds, references=p.label_ids, average='weighted')['f1']
        }

    # training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model='f1'
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(args.output_dir)

    # Inference with composite score
    logging.info('Computing composite scores on validation set')
    val = ds['validation']
    encs = {k: torch.tensor(v) for k,v in val.items() if k in ['input_ids','attention_mask','e1_pos','e2_pos']}
    with torch.no_grad():
        logits = model(encs['input_ids'], encs['attention_mask'], encs['e1_pos'], encs['e2_pos'])
        probs = torch.softmax(logits, dim=1)
    # assume label2id maps positive class for each relation
    for i,(e1,e2) in enumerate(pairs[len(train_data):]):
        transformer_score = probs[i, val['label'][i]].item()
        co = coocc_scores.get((e1,e2),0.0)
        composite = 0.6*transformer_score + 0.4*co
        print(f"Pair ({e1},{e2}): transformer={transformer_score:.3f}, coocc={co:.3f}, composite={composite:.3f}")

if __name__ == '__main__':
    main()
    main()
