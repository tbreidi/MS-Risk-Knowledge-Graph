"""
Compute ontology mappings between Social Determinants of Health (SDOH) and Risk Factor terms
based on SNOMED CT nearest-neighbor (shared ancestors) metric.

Usage:
    python SDOH_RISK_SNOMETCT.py \
      --sdoh_csv sdoh_snomed.csv \
      --risk_csv risk_snomed.csv \
      --snomed_owl SNOMEDCT.owl \
      --output mapping.json \
      [--topk 5] [--threshold 0.1]

Inputs:
  sdoh_snomed.csv: CSV with columns [term,snomed_id]
  risk_snomed.csv: CSV with columns [term,snomed_id]
  SNOMEDCT.owl: SNOMED CT ontology OWL file

Output:
  mapping.json: JSON list of mappings {sdoh_term, sdoh_id, risk_term, risk_id, score}
"""
import argparse
import json
import pandas as pd
import heapq
from owlready2 import get_ontology, Thing


def load_csv(path):
    df = pd.read_csv(path, dtype=str)
    if 'term' not in df.columns or 'snomed_id' not in df.columns:
        raise ValueError(f"CSV {path} must contain 'term' and 'snomed_id' columns.")
    return df[['term','snomed_id']].dropna()


def find_class_by_id(onto, snomed_id):
    # SNOMED CT classes typically have IRI ending with /<id>
    matches = list(onto.search(iri=f"*{snomed_id}"))
    return matches[0] if matches else None


def compute_ancestors(klass):
    # Exclude Thing
    return set(a for a in klass.ancestors() if a is not Thing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdoh_csv', required=True, help='SDOH terms CSV')
    parser.add_argument('--risk_csv', required=True, help='Risk factors CSV')
    parser.add_argument('--snomed_owl', required=True, help='SNOMED CT OWL file')
    parser.add_argument('--output', required=True, help='Output JSON mapping file')
    parser.add_argument('--topk', type=int, default=5, help='Top K risk factors per SDOH')
    parser.add_argument('--threshold', type=float, default=0.0, help='Minimum score to include')
    args = parser.parse_args()

    # Load SDOH and risk data
    sdoh_df = load_csv(args.sdoh_csv)
    risk_df = load_csv(args.risk_csv)

    # Load SNOMED CT ontology
    print(f"Loading SNOMED CT ontology from {args.snomed_owl}...")
    onto = get_ontology(args.snomed_owl).load()
    print("Ontology loaded.")

    # Map terms to classes and compute ancestors
    sdoh_anc = {}
    for _, row in sdoh_df.iterrows():
        term, sid = row['term'], row['snomed_id']
        cls = find_class_by_id(onto, sid)
        if cls:
            sdoh_anc[(term, sid)] = compute_ancestors(cls)
        else:
            print(f"Warning: SDOH SNOMED class {sid} not found for term '{term}'")

    risk_anc = {}
    for _, row in risk_df.iterrows():
        term, sid = row['term'], row['snomed_id']
        cls = find_class_by_id(onto, sid)
        if cls:
            risk_anc[(term, sid)] = compute_ancestors(cls)
        else:
            print(f"Warning: Risk SNOMED class {sid} not found for term '{term}'")

    # Compute nearest neighbors
    mapping = []
    for (sterm, sid), s_anc in sdoh_anc.items():
        heap = []  # max-heap via negative
        for (rterm, rid), r_anc in risk_anc.items():
            shared = s_anc.intersection(r_anc)
            union = s_anc.union(r_anc)
            score = len(shared) / len(union) if union else 0.0
            if score >= args.threshold:
                heapq.heappush(heap, (-score, rterm, rid))
        # take topk
        for _ in range(min(args.topk, len(heap))):
            neg_score, rterm, rid = heapq.heappop(heap)
            mapping.append({
                'sdoh_term': sterm,
                'sdoh_id': sid,
                'risk_term': rterm,
                'risk_id': rid,
                'score': round(-neg_score, 4)
            })

    # Save mapping
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    print(f"Mapping written to {args.output} ({len(mapping)} pairs)")

if __name__ == '__main__':
    main()
