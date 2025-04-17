
"""
Performs:
 1. Biomedical NER on PubMed abstracts using transformer-based model.
 2. Refined SDoH mapping via cosine similarity against a given ontology.
 3. Outputs processed articles with 'entities' and 'sdoh_mentions'.

Usage:
    python NERLabelling.py \
        --input abstracts.json \
        --ontology_file sdoh_ontology.json \
        --output processed.json \
        --ner_model d4data/biomedical-ner-all \
        --embed_model all-MiniLM-L6-v2 \
        [--threshold 0.5] [--no-plot]
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import torch
import spacy
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util

# Ensure punkt is available
nltk.download('punkt', quiet=True)


def init_ner(ner_model: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(ner_model)
    model = AutoModelForTokenClassification.from_pretrained(ner_model)
    return pipeline(
        'ner', model=model, tokenizer=tokenizer,
        aggregation_strategy='simple', device=device
    )


def load_ontology(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_ontology(ontology: dict) -> dict:
    flat = {}
    for main_cat, subcats in ontology.items():
        for subcat, phrases in subcats.items():
            key = f"{main_cat}::{subcat}"
            flat[key] = [p.lower() for p in phrases]
    return flat


def main():
    parser = argparse.ArgumentParser(description='NER and SDoH mapping')
    parser.add_argument('--input', required=True, help='Path to abstracts JSON')
    parser.add_argument('--ontology_file', required=True, help='Path to refined SDoH ontology JSON')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--ner_model', default='d4data/biomedical-ner-all', help='NER model name')
    parser.add_argument('--embed_model', default='all-MiniLM-L6-v2', help='SentenceTransformer model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold')
    parser.add_argument('--no_plot', action='store_true', help='Skip plotting counts')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load abstracts
    logging.info(f"Loading abstracts from {args.input}")
    articles = json.loads(Path(args.input).read_text(encoding='utf-8'))

    # Initialize NER
    logging.info(f"Initializing NER model: {args.ner_model}")
    ner_pipe = init_ner(args.ner_model, device)

    # Perform NER
    logging.info("Performing NER on abstracts...")
    texts = [art.get('abstract','') or '' for art in articles]
    ner_results = ner_pipe(texts)
    all_labels = []
    for art, ents in zip(articles, ner_results):
        art['entities'] = ents
        all_labels.extend([e['entity_group'] for e in ents])

    # Log entity counts
    label_counts = Counter(all_labels)
    logging.info("Entity counts:")
    for label, cnt in label_counts.items():
        logging.info(f"  {label}: {cnt}")

    # Load and flatten ontology
    logging.info(f"Loading ontology from {args.ontology_file}")
    ontology = load_ontology(Path(args.ontology_file))
    flat_ont = flatten_ontology(ontology)

    # Prepare embeddings
    logging.info(f"Loading embedder: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)
    ont_embeddings = {cat: embedder.encode(phs, convert_to_tensor=True) for cat, phs in flat_ont.items()}

    # Initialize spaCy for noun-chunks
    logging.info("Loading spaCy model en_core_web_sm")
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner','textcat'])

    # Map SDoH mentions
    logging.info("Extracting and mapping SDoH mentions...")
    all_sdoh = []
    for art in articles:
        abstract = art.get('abstract','') or ''
        sdoh_mentions = []
        if abstract:
            doc = nlp(abstract)
            phrases = {chunk.text.strip() for chunk in doc.noun_chunks}
            for phrase in phrases:
                p_text = phrase.lower()
                emb = embedder.encode(p_text, convert_to_tensor=True)
                best_cat, best_score = None, 0.0
                for cat, cat_emb in ont_embeddings.items():
                    sim = util.cos_sim(emb, cat_emb).max().item()
                    if sim > best_score:
                        best_cat, best_score = cat, sim
                if best_score >= args.threshold:
                    sdoh_mentions.append({'phrase': phrase, 'category': best_cat, 'score': float(best_score)})
                    all_sdoh.append(best_cat)
        art['sdoh_mentions'] = sdoh_mentions

    # Count SDoH mentions
    sdoh_counts = Counter(all_sdoh)
    logging.info("SDoH mention counts:")
    for cat, cnt in sdoh_counts.most_common():
        logging.info(f"  {cat}: {cnt}")

    # Optional plotting
    if not args.no_plot and sdoh_counts:
        try:
            import matplotlib.pyplot as plt
            labels, counts = zip(*sdoh_counts.most_common(10))
            plt.figure(figsize=(10,6))
            plt.bar(labels, counts)
            plt.xticks(rotation=45, ha='right')
            plt.title('Top 10 SDoH Categories')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.warning(f"Plotting skipped: {e}")

    # Save output
    logging.info(f"Saving processed data to {args.output}")
    Path(args.output).write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding='utf-8')
    logging.info("Done.")

if __name__ == '__main__':
    main()
