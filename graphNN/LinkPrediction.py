"""
gnn_link_prediction.py

Script to train a GNN for link prediction on the constructed knowledge graph.
Leverages Node2Vec and GraphSAGE using PyTorch Geometric.
"""
import argparse
import torch
import torch.nn.functional as f
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import Node2Vec, SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin = torch.nn.Linear(in_channels * 2, 1)

    def forward(self, z, edge_index):
        src, dst = edge_index
        out = torch.cat([z[src], z[dst]], dim=1)
        return torch.sigmoid(self.lin(out)).view(-1)


def train_epoch(node2vec, model, predictor, data, optimizer):
    model.train()
    predictor.train()
    node2vec.train()
    optimizer.zero_grad()

    # Compute embeddings
    z = node2vec()  # shape [num_nodes, hidden]
    # GraphSAGE on training edges
    emb = model(z, data.train_pos_edge_index)

    # Positive and negative link scores
    pos_score = predictor(emb, data.train_pos_edge_index)
    neg_score = predictor(emb, data.train_neg_edge_index)

    # Loss
    loss = -torch.log(pos_score + 1e-15).mean() - torch.log(1 - neg_score + 1e-15).mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def test(predictor, emb, edge_index_pos, edge_index_neg):
    predictor.eval()
    with torch.no_grad():
        pos_pred = predictor(emb, edge_index_pos)
        neg_pred = predictor(emb, edge_index_neg)
        preds = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0)
        return ((preds > 0.5).float() == labels).sum().item() / labels.size(0)


def main():
    parser = argparse.ArgumentParser(description="Train GNN link prediction PrimeKGIntegration.")
    parser.add_argument('--input', type=str, default='kg_data.pt',
                        help="Path to preprocessed KG data (.pt file)")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs for GNN")
    parser.add_argument('--hidden', type=int, default=64,
                        help="Dimension of hidden embeddings")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument('--output', type=str, default='PrimeKGIntegration.pt',
                        help="Path to save the trained PrimeKGIntegration checkpoint")
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    ckpt = torch.load(args.input)
    data = ckpt['data']
    data = train_test_split_edges(data)
    data = data.to(device)

    # Node2Vec pre-training
    node2vec = Node2Vec(
        data.train_pos_edge_index, embedding_dim=args.hidden,
        walk_length=20, context_size=10, walks_per_node=10,
        num_negative_samples=1, p=1, q=1, sparse=True,
    ).to(device)

    model = GraphSAGE(in_channels=args.hidden, hidden_channels=args.hidden).to(device)
    predictor = LinkPredictor(in_channels=args.hidden).to(device)

    optimizer = torch.optim.Adam(
        list(node2vec.parameters()) + list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr
    )

    # Pre-train Node2Vec
    print("Pre-training Node2Vec embeddings...")
    for epoch in range(1, 51):
        loss = node2vec.loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Node2Vec Epoch {epoch:02d}, Loss: {loss:.4f}")
    node2vec.eval()

    # GNN training
    print("Training GraphSAGE link predictor...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(node2vec, model, predictor, data, optimizer)
        if epoch % 10 == 0:
            z = node2vec()
            emb = model(z, data.train_pos_edge_index)
            train_acc = test(predictor, emb, data.train_pos_edge_index, data.train_neg_edge_index)
            val_acc = test(predictor, emb, data.val_pos_edge_index, data.val_neg_edge_index)
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Final test
    z = node2vec()
    emb = model(z, data.train_pos_edge_index)
    test_acc = test(predictor, emb, data.test_pos_edge_index, data.test_neg_edge_index)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'predictor_state_dict': predictor.state_dict(),
        'node2vec_state_dict': node2vec.state_dict(),
        'node_classes': ckpt['node_classes']
    }, args.output)
    print(f"Model saved to {args.output}")

if __name__ == '__main__':
    main()
