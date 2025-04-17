"""
Efficient NER and Social Determinants of Health (SDoH) mapping from PubMed abstracts.
Performs:
 1. Biomedical NER using a transformer-based model.
 2. SDoH phrase extraction via spaCy noun-chunking.
 3. SDoH category assignment using SentenceTransformer embeddings.
 4. Optional LLM verification for borderline cases.

Usage:
    python NERMapping.py --input abstracts.json --output processed.json \
        --ner_model d4data/biomedical-ner-all --embed_model all-MiniLM-L6-v2 \
        [--use_llm] [--llm_model meta-llama/Llama-3.2-1B] [--hf_token YOUR_TOKEN]
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize

# ----------------------------------------------------------------------------
# Configuration defaults
# ----------------------------------------------------------------------------
LOW_THRESHOLD = 0.5
HIGH_THRESHOLD = 0.7
BATCH_SIZE = 16

# ----------------------------------------------------------------------------
# SDoH Ontology Loader from Correlation File
# ----------------------------------------------------------------------------

def load_sdoh_ontology(correlations_file: str):
    """
    Load SDoH-to-Risk correlations as ontology mappings.
    Expects a JSON file with list of edges:
      [{"source": "SDOH_term", "target": "Risk_term", ...}, ...]
    Returns dict: { "SDOH_term": ["Risk_term1", ...], ... }
    """
    import json
    from collections import defaultdict

    with open(correlations_file, 'r', encoding='utf-8') as f:
        edges = json.load(f)

    ontology = defaultdict(list)
    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        if src and tgt:
            ontology[src].append(tgt)
    return dict(ontology)

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(data, path):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

# ----------------------------------------------------------------------------
# Initialize pipelines and models
# ----------------------------------------------------------------------------

def init_ner_pipeline(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        'ner', model=model, tokenizer=tokenizer,
        aggregation_strategy='simple', device=device,
        batch_size=BATCH_SIZE
    )


def init_embed_model(model_name):
    return SentenceTransformer(model_name)


def init_llm_pipeline(model_id, hf_token):
    return pipeline(
        'text-generation', model=model_id,
        trust_remote_code=True,
        device_map='auto',
        use_auth_token=hf_token,
        max_new_tokens=5,
        do_sample=False
    )

# ----------------------------------------------------------------------------
# Main extraction functions
# ----------------------------------------------------------------------------

def perform_ner(articles, ner_pipeline):
    texts = [a.get('abstract','') or '' for a in articles]
    ner_results = ner_pipeline(texts)
    for art, ents in zip(articles, ner_results):
        art['entities'] = ents or []
    return articles


def prepare_sdoh_embeddings(ontology, embed_model):
    embeddings = {}
    for cat, phrases in ontology.items():
        embeddings[cat] = embed_model.encode(phrases, convert_to_tensor=True)
    return embeddings


def map_sdoh(articles, ontology_embeddings, embed_model, use_llm=False, llm_pipeline=None):
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner','textcat'])
    for art in articles:
        sdoh_mentions = []
        abstract = art.get('abstract','') or ''
        if not abstract:
            art['sdoh_mentions'] = []
            continue
        # Extract noun phrases
        doc = nlp(abstract)
        phrases = list({chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()})
        if not phrases:
            art['sdoh_mentions'] = []
            continue
        # Batch embed phrases
        phrase_embs = embed_model.encode(phrases, convert_to_tensor=True)
        # For LLM fallback, prepare sentences
        sentences = sent_tokenize(abstract)
        for phrase, emb in zip(phrases, phrase_embs):
            best_cat, best_score = None, LOW_THRESHOLD
            # find highest similarity category
            for cat, cat_embs in ontology_embeddings.items():
                sim = util.cos_sim(emb, cat_embs).max().item()
                if sim > best_score:
                    best_cat, best_score = cat, sim
            if best_score < LOW_THRESHOLD:
                continue
            if best_score >= HIGH_THRESHOLD or not use_llm:
                mention = {'phrase': phrase, 'category': best_cat, 'score': best_score}
                if use_llm and best_score < HIGH_THRESHOLD:
                    mention['score'] = best_score
                sdoh_mentions.append(mention)
            else:
                # LLM verify borderline
                sent = next((s for s in sentences if phrase in s), abstract)
                prompt = (
                    f"Context: {sent}\nPhrase: '{phrase}'\nCategory: '{best_cat}'\n" +
                    "Is this classification correct? Answer 'yes' or 'no'."
                )
                resp = llm_pipeline(prompt)[0]['generated_text'].lower()
                if 'yes' in resp and 'no' not in resp:
                    sdoh_mentions.append({'phrase': phrase, 'category': best_cat, 'score': best_score, 'verified': True})
        art['sdoh_mentions'] = sdoh_mentions
    return articles

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input JSON of articles')
    parser.add_argument('--output', required=True, help='Output JSON with entities')
    parser.add_argument('--ner_model', default='d4data/biomedical-ner-all')
    parser.add_argument('--embed_model', default='all-MiniLM-L6-v2')
    parser.add_argument('--use_llm', action='store_true')
    parser.add_argument('--llm_model', default=None)
    parser.add_argument('--hf_token', default=None)
    args = parser.parse_args()

    articles = load_json(args.input)
    logging.info(f'Loaded {len(articles)} articles')

    ner_pipe = init_ner_pipeline(args.ner_model)
    logging.info('Performing NER...')
    articles = perform_ner(articles, ner_pipe)

    ontology = load_sdoh_ontology()
    embed_model = init_embed_model(args.embed_model)
    logging.info('Preparing SDoH embeddings...')
    ontology_embeddings = prepare_sdoh_embeddings(ontology, embed_model)

    llm_pipe = None
    if args.use_llm and args.llm_model and args.hf_token:
        llm_pipe = init_llm_pipeline(args.llm_model, args.hf_token)
        logging.info('Initialized LLM for verification')

    logging.info('Mapping SDoH mentions...')
    articles = map_sdoh(articles, ontology_embeddings, embed_model, args.use_llm, llm_pipe)

    save_json(articles, args.output)
    logging.info(f'Processed data saved to {args.output}')

if __name__ == '__main__':
    main()
