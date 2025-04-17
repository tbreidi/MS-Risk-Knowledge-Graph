#!/usr/bin/env python3

"""
exploratory_analysis.py

Script to reproduce key results from the MS-SDOH KG paper:
- Print node and edge counts
- Compute test AUC for link prediction
- Display top 3 predicted links
"""

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import Node2Vec
from gnn_link_prediction import GraphSAGE, LinkPredictor


def main():
    # Load preprocessed KG data
    ckpt = torch.load('kg_data.pt')
    data = ckpt['data']
    node_classes = ckpt['node_classes']

    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.size(1)}")

    # Split edges for link prediction
    data = train_test_split_edges(data)

    # Load trained PrimeKGIntegration checkpoint
    model_ckpt = torch.load('PrimeKGIntegration.pt')

    # Determine embedding dimension from Node2Vec state dict
    if 'node2vec_state_dict' in model_ckpt and 'embedding.weight' in model_ckpt['node2vec_state_dict']:
        hidden = model_ckpt['node2vec_state_dict']['embedding.weight'].shape[1]
    else:
        hidden = 64  # fallback

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Node2Vec, GraphSAGE, and LinkPredictor
    node2vec = Node2Vec(
        data.train_pos_edge_index,
        embedding_dim=hidden,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True
    ).to(device)

    model = GraphSAGE(in_channels=hidden, hidden_channels=hidden).to(device)
    predictor = LinkPredictor(in_channels=hidden).to(device)

    # Load saved weights
    node2vec.load_state_dict(model_ckpt['node2vec_state_dict'])
    model.load_state_dict(model_ckpt['model_state_dict'])
    predictor.load_state_dict(model_ckpt['predictor_state_dict'])

    node2vec.eval()
    model.eval()
    predictor.eval()

    # Move data to device
    data = data.to(device)

    # Compute node embeddings and edge embeddings
    with torch.no_grad():
        z = node2vec()
        emb = model(z, data.train_pos_edge_index)

    # Compute scores for positive and negative test edges
    pos_scores = predictor(emb, data.test_pos_edge_index).cpu().numpy()
    neg_scores = predictor(emb, data.test_neg_edge_index).cpu().numpy()

    # Build DataFrame of test positive edges and their scores
    src, dst = data.test_pos_edge_index
    df_test = pd.DataFrame({
        'head': [node_classes[i] for i in src.cpu().numpy()],
        'tail': [node_classes[i] for i in dst.cpu().numpy()],
        'score': pos_scores
    })

    top3 = df_test.sort_values('score', ascending=False).head(3)
    print("\nTop 3 predicted edges in test set:")
    print(top3.to_string(index=False))

    # Calculate and print AUC
    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores = list(pos_scores) + list(neg_scores)
    auc = roc_auc_score(y_true, y_scores)
    print(f"\nTest AUC: {auc:.4f}")

if __name__ == '__main__':
    main()
