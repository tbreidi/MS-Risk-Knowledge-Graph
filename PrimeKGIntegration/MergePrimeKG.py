"""
merge_kg_with_primekg.py

Merge a custom KG (from text extraction) with PrimeKG subgraph focused on Multiple Sclerosis (MS).

Steps:
 1. Load custom KG triples (CSV with columns: head, relation, tail, head_id, tail_id, head_type, tail_type).
 2. Load PrimeKG triples (CSV with same schema).
 3. Extract MS-relevant subgraph from PrimeKG (nodes within 2 hops of MS node).
 4. Entity matching & merging:
    - Use MeSH/OMIM-to-MONDO and MeSH-to-DrugBank mapping files.
    - For unmatched biomedical nodes, optionally lookup UMLS CUI (stubbed).
 5. Combine custom triples + filtered PrimeKG triples, merging nodes by identifier.
 6. Save merged triples to output CSV.

Usage:
    python merge_kg_with_primekg.py \
      --custom custom_triples.csv \
      --prime primekg_triples.csv \
      --disease_map mesh_omim_to_mondo.csv \
      --drug_map mesh_to_drugbank.csv \
      --ms_id MONDO:0005301 \
      --output merged_triples.csv
"""
import argparse
import csv
import networkx as nx
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)


def load_triples(path, subject_col, pred_col, object_col, sid_col, oid_col, stype_col=None, otype_col=None):
    """Load triples from CSV into list of dicts."""
    triples = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            triples.append({
                'head': row[subject_col],
                'relation': row[pred_col],
                'tail': row[object_col],
                'head_id': row[sid_col],
                'tail_id': row[oid_col],
                'head_type': row.get(stype_col, ''),
                'tail_type': row.get(otype_col, '')
            })
    return triples


def build_graph(triples):
    """Build a NetworkX directed multigraph from triples."""
    G = nx.MultiDiGraph()
    for t in triples:
        # add nodes with id and type attributes
        G.add_node(t['head_id'], name=t['head'], type=t['head_type'])
        G.add_node(t['tail_id'], name=t['tail'], type=t['tail_type'])
        # add edge
        G.add_edge(t['head_id'], t['tail_id'], relation=t['relation'])
    return G


def extract_ms_subgraph(G, ms_id, max_hops=2):
    """Return subgraph of nodes within max_hops of ms_id (undirected BFS)."""
    visited = {ms_id: 0}
    queue = deque([ms_id])
    while queue:
        node = queue.popleft()
        depth = visited[node]
        if depth >= max_hops:
            continue
        # neighbors in both directions
        for nbr in set(G.successors(node)) | set(G.predecessors(node)):
            if nbr not in visited:
                visited[nbr] = depth + 1
                queue.append(nbr)
    sub_nodes = list(visited.keys())
    return G.subgraph(sub_nodes).copy()


def load_mapping(path):
    """Load two-column CSV mapping source_id to target_id."""
    m = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for src, tgt in reader:
            m[src] = tgt
    return m


def merge_graphs(G_custom, G_prime, disease_map, drug_map):
    """Merge two graphs by node identifier, applying mappings."""
    G_merged = nx.MultiDiGraph()
    # Add custom graph completely
    for u, v, data in G_custom.edges(data=True):
        G_merged.add_node(u, **G_custom.nodes[u])
        G_merged.add_node(v, **G_custom.nodes[v])
        G_merged.add_edge(u, v, relation=data['relation'])

    # Add prime subgraph, mapping identifiers
    for u, v, data in G_prime.edges(data=True):
        u_map = disease_map.get(u, drug_map.get(u, u))
        v_map = disease_map.get(v, drug_map.get(v, v))
        # (stub) UMLS lookup could be inserted here for unmatched nodes
        G_merged.add_node(u_map, **G_prime.nodes[u])
        G_merged.nodes[u_map]['original_id'] = u  # retain original
        G_merged.add_node(v_map, **G_prime.nodes[v])
        G_merged.nodes[v_map]['original_id'] = v
        G_merged.add_edge(u_map, v_map, relation=data['relation'])

    return G_merged


def export_triples(G, output_path):
    """Write merged graph to CSV triples."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['head_id','head_name','head_type','relation','tail_id','tail_name','tail_type'])
        for u, v, data in G.edges(data=True):
            h = G.nodes[u]
            t = G.nodes[v]
            writer.writerow([u, h.get('name',''), h.get('type',''), data['relation'], v, t.get('name',''), t.get('type','')])


def main():
    parser = argparse.ArgumentParser(description='Merge custom KG with PrimeKG subgraph')
    parser.add_argument('--custom', required=True, help='Custom triples CSV')
    parser.add_argument('--prime', required=True, help='PrimeKG triples CSV')
    parser.add_argument('--disease_map', required=True, help='MeSH/OMIM to MONDO mapping CSV')
    parser.add_argument('--drug_map', required=True, help='MeSH to DrugBank mapping CSV')
    parser.add_argument('--ms_id', required=True, help='MS node identifier in PrimeKG (e.g., MONDO:0005301)')
    parser.add_argument('--output', required=True, help='Output merged triples CSV')
    args = parser.parse_args()

    # Load and build graphs
    logging.info('Loading custom KG...')
    custom_triples = load_triples(args.custom, 'head','relation','tail','head_id','tail_id','head_type','tail_type')
    G_custom = build_graph(custom_triples)

    logging.info('Loading PrimeKG...')
    prime_triples = load_triples(args.prime, 'head','relation','tail','head_id','tail_id','head_type','tail_type')
    G_prime_full = build_graph(prime_triples)

    logging.info('Extracting MS-relevant subgraph...')
    G_prime_sub = extract_ms_subgraph(G_prime_full, args.ms_id, max_hops=2)

    logging.info('Loading mapping files...')
    disease_map = load_mapping(args.disease_map)
    drug_map = load_mapping(args.drug_map)

    logging.info('Merging graphs...')
    G_merged = merge_graphs(G_custom, G_prime_sub, disease_map, drug_map)

    logging.info('Exporting merged triples...')
    export_triples(G_merged, args.output)
    logging.info(f'Merged KG saved to {args.output}')

if __name__ == '__main__':
    main()
