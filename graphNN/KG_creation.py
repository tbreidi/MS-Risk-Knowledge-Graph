"""
kg_construction.py

Script to construct a knowledge graph from entity-relation triplets.
Reads triplets from a CSV file and builds a PyTorch Geometric Data object,
saving the graph data for downstream GNN training.
"""

import argparse
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder


def load_triplets(file_path):
    """Load triplets CSV with columns: head, relation, tail."""
    df = pd.read_csv(file_path)
    if not {'head','relation','tail'}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'head','relation','tail' columns")
    return df


def encode_nodes(df):
    """Encode node names to integer indices."""
    nodes = pd.concat([df['head'], df['tail']]).unique()
    le = LabelEncoder()
    le.fit(nodes)
    head_idx = le.transform(df['head'])
    tail_idx = le.transform(df['tail'])
    return head_idx, tail_idx, le


def build_data(head_idx, tail_idx, num_nodes):
    """Build PyG Data object from edge indices."""
    edge_index = torch.tensor([head_idx, tail_idx], dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    return data


def main():
    parser = argparse.ArgumentParser(description="Construct KG Data from triplets CSV.")
    parser.add_argument('--triplets', type=str, required=True,
                        help="Path to CSV file containing triplets: head,relation,tail")
    parser.add_argument('--output', type=str, default='kg_data.pt',
                        help="Output path for saved graph data")
    args = parser.parse_args()

    # Load and process
    df = load_triplets(args.triplets)
    head_idx, tail_idx, le = encode_nodes(df)
    data = build_data(head_idx, tail_idx, num_nodes=len(le.classes_))

    # Save graph data and encodings
    torch.save({
        'data': data,
        'node_classes': le.classes_.tolist(),
        'relations': df['relation'].tolist()
    }, args.output)
    print(f"Saved processed KG data to {args.output}")


if __name__ == "__main__":
    main()
