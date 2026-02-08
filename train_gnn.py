"""
Minimal GNN training script for diffusion dynamics.
Trains a 2-layer GCN to predict x_{t+1} from x_t.
"""

import os
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# -----------------------
# Graph loading (same as pipeline)
# -----------------------
def load_email_eu_core_graph():
    """Load Email-Eu-Core graph, matching the pipeline's logic."""
    # Check for user-provided dataset first (multiple possible locations/names)
    possible_datasets = [
        os.path.join("dataset", "email-Eu-core.txt"),
        os.path.join("dataset", "Email-EuAll.txt"),
        os.path.join("dataset", "email-EuAll.txt"),
        os.path.join("data", "email-Eu-core.txt"),
    ]
    
    edge_file = None
    for dataset_path in possible_datasets:
        if os.path.exists(dataset_path):
            edge_file = dataset_path
            print(f"Using dataset from: {edge_file}")
            break
    
    if edge_file is None:
        raise FileNotFoundError(
            "Missing dataset file. Expected one of: dataset/email-Eu-core.txt, "
            "dataset/Email-EuAll.txt, or data/email-Eu-core.txt"
        )

    edges = []
    with open(edge_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if u != v:
                edges.append((u, v))

    G = nx.Graph()
    G.add_edges_from(edges)

    # Largest connected component
    lcc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc).copy()

    # Remap nodes to 0..n-1
    mapping = {node: idx for idx, node in enumerate(sorted(G_lcc.nodes()))}
    G_mapped = nx.relabel_nodes(G_lcc, mapping)

    # PyG edge_index (store both directions for undirected graph)
    edge_index = torch.tensor(list(G_mapped.edges()), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    data = Data(edge_index=edge_index, num_nodes=G_mapped.number_of_nodes())
    return G_mapped, data


# -----------------------
# Minimal GNN
# -----------------------
class DiffusionGCN(nn.Module):
    """2-layer GCN for predicting next-step node states."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, x, edge_index):
        # x: [N, 1] float
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h  # logits [N, 1]


def main():
    # ---- Load trajectories ----
    traj_path = "trajectories.npy"
    if not os.path.exists(traj_path):
        raise FileNotFoundError("Missing trajectories.npy. Run diffusion_pipeline.py first.")

    X = np.load(traj_path)  # shape: (T+1, N), int {0,1}
    T1, N = X.shape
    T = T1 - 1
    print(f"Loaded trajectories: {X.shape} (time_steps, nodes)")

    # ---- Rebuild graph ----
    G, data = load_email_eu_core_graph()
    assert data.num_nodes == N, f"Node mismatch: data.num_nodes={data.num_nodes} vs traj N={N}"

    # Degree for analysis plot
    deg = np.array([d for _, d in sorted(G.degree(), key=lambda x: x[0])], dtype=np.int32)

    # ---- Build input-output pairs (time pairs) ----
    # pair t: input = x^t, target = x^{t+1}
    inputs = torch.tensor(X[:-1], dtype=torch.float32).unsqueeze(-1)   # [T, N, 1]
    targets = torch.tensor(X[1:], dtype=torch.float32).unsqueeze(-1)   # [T, N, 1]

    # ---- Train/test split over time ----
    # Hold out last 20% of time steps
    split = int(0.8 * T)
    train_inputs, test_inputs = inputs[:split], inputs[split:]
    train_targets, test_targets = targets[:split], targets[split:]
    print(f"Train steps: {train_inputs.size(0)}, Test steps: {test_inputs.size(0)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DiffusionGCN(hidden_dim=32).to(device)
    edge_index = data.edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    # ---- Training loop ----
    epochs = 200
    train_losses, test_losses = [], []

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss = 0.0
        for t in range(train_inputs.size(0)):
            logits = model(train_inputs[t], edge_index)
            loss = loss + bce(logits, train_targets[t])
        loss = loss / train_inputs.size(0)

        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            t_loss = 0.0
            for t in range(test_inputs.size(0)):
                logits = model(test_inputs[t], edge_index)
                t_loss = t_loss + bce(logits, test_targets[t])
            t_loss = t_loss / test_inputs.size(0)

        train_losses.append(loss.item())
        test_losses.append(t_loss.item())

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train BCE={loss.item():.4f} | test BCE={t_loss.item():.4f}")

    # ---- Save plots folder ----
    os.makedirs("figures", exist_ok=True)

    # (1) Diffusion curve
    frac_active = X.mean(axis=1)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(T1), frac_active, linewidth=2)
    plt.xlabel("time step t")
    plt.ylabel("fraction active")
    plt.title("Markov diffusion curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/diffusion_curve.png", dpi=200)
    plt.close()

    # (2) Train/test loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, epochs + 1), train_losses, label="train", linewidth=2)
    plt.plot(np.arange(1, epochs + 1), test_losses, label="test", linewidth=2)
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title("GNN training curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/loss_curve.png", dpi=200)
    plt.close()

    # (3) Error vs degree on test steps
    model.eval()
    with torch.no_grad():
        # Predict next-step probabilities for each held-out time step
        probs_list = []
        y_list = []
        for t in range(test_inputs.size(0)):
            logits = model(test_inputs[t], edge_index)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()  # [N]
            y = test_targets[t].squeeze(-1).cpu().numpy()            # [N]
            probs_list.append(probs)
            y_list.append(y)

    # Per-node absolute error aggregated over test steps
    abs_err_per_node = np.zeros(N, dtype=np.float64)
    for i in range(N):
        # collect predictions for node i across test steps
        pi = np.array([probs_list[t][i] for t in range(len(probs_list))])
        yi = np.array([y_list[t][i] for t in range(len(y_list))])
        abs_err_per_node[i] = np.mean(np.abs(pi - yi))

    # Bin nodes by degree
    bins = np.quantile(deg, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(int)
    # Ensure strictly increasing bins
    bins = np.unique(bins)
    if bins.size < 3:
        # fallback if degrees are too concentrated
        bins = np.array([deg.min(), int(np.median(deg)), deg.max()])

    bin_labels = []
    bin_err = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (deg >= b0) & (deg <= b1)
        if mask.sum() == 0:
            continue
        bin_labels.append(f"{b0}-{b1}")
        bin_err.append(abs_err_per_node[mask].mean())

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(bin_err)), bin_err, marker="o", linewidth=2, markersize=8)
    plt.xticks(np.arange(len(bin_err)), bin_labels, rotation=30, ha="right")
    plt.xlabel("degree bin")
    plt.ylabel("mean |p - y|")
    plt.title("GNN error vs node degree (test)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/error_vs_degree.png", dpi=200)
    plt.close()

    print("\nSaved figures:")
    print("  figures/diffusion_curve.png")
    print("  figures/loss_curve.png")
    print("  figures/error_vs_degree.png")


if __name__ == "__main__":
    main()

