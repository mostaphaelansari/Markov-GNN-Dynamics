"""
Degree-only baseline: p_i = sigmoid(a * deg(i) + b).
Uses same train/test split as GNN. Generates comparison figures.
Run after train_gnn.py (requires model_gnn.pt and trajectories.npy).
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from train_gnn import load_email_eu_core_graph, DiffusionGCN


def bce_numpy(p, y, eps=1e-8):
    """BCE loss: -y*log(p) - (1-y)*log(1-p). Clamp p to avoid log(0)."""
    p = np.clip(p, eps, 1 - eps)
    return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))


def main():
    # ---- Load trajectories ----
    traj_path = "trajectories.npy"
    if not os.path.exists(traj_path):
        raise FileNotFoundError("Missing trajectories.npy. Run diffusion_pipeline.py first.")

    X = np.load(traj_path)  # (T+1, N)
    T1, N = X.shape
    T = T1 - 1

    # ---- Rebuild graph (same as train_gnn) ----
    G, data = load_email_eu_core_graph()
    assert data.num_nodes == N
    deg = np.array([d for _, d in sorted(G.degree(), key=lambda x: x[0])], dtype=np.float64)

    # Same train/test split as GNN
    split = int(0.8 * T)
    train_targets_np = X[1 : split + 1]   # [split, N] targets for steps 0..split-1
    test_targets_np = X[split + 1 : T1]   # [T-split, N] targets for steps split..T-1
    n_test_steps = test_targets_np.shape[0]

    # ---- Load GNN and evaluate on test ----
    model_path = "model_gnn.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Missing model_gnn.pt. Run train_gnn.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    model = DiffusionGCN(hidden_dim=ckpt["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    inputs = torch.tensor(X[:-1], dtype=torch.float32).unsqueeze(-1)  # [T, N, 1]
    targets = torch.tensor(X[1:], dtype=torch.float32).unsqueeze(-1)  # [T, N, 1]
    test_inputs = inputs[split:].to(device)
    test_targets = targets[split:]
    edge_index = data.edge_index.to(device)

    gnn_test_bce = 0.0
    probs_list = []
    y_list = []
    with torch.no_grad():
        for t in range(test_inputs.size(0)):
            logits = model(test_inputs[t], edge_index)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            y = test_targets[t].squeeze(-1).cpu().numpy()
            probs_list.append(probs)
            y_list.append(y)
            gnn_test_bce += bce_numpy(probs, y)
    gnn_test_bce /= test_inputs.size(0)

    # GNN per-node mean absolute error (over test steps)
    abs_err_gnn = np.zeros(N)
    for i in range(N):
        pi = np.array([probs_list[t][i] for t in range(len(probs_list))])
        yi = np.array([y_list[t][i] for t in range(len(y_list))])
        abs_err_gnn[i] = np.mean(np.abs(pi - yi))

    # ---- Degree-only baseline: fit p_i = sigmoid(a*deg(i) + b) on train ----
    # Training data: (degree, target) for every (node, train step)
    X_train = np.tile(deg, split).reshape(-1, 1)  # [split*N, 1]
    y_train = train_targets_np.reshape(-1)         # [split*N]

    lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    lr.fit(X_train, y_train)
    a, b = lr.coef_[0, 0], lr.intercept_[0]
    # Probability for each node (same for all time steps)
    p_deg = 1.0 / (1.0 + np.exp(-(a * deg + b)))

    # Test BCE for degree-only (average over test steps)
    degree_test_bce = 0.0
    for t in range(n_test_steps):
        degree_test_bce += bce_numpy(p_deg, test_targets_np[t])
    degree_test_bce /= n_test_steps

    # Degree-only per-node mean absolute error
    abs_err_deg = np.zeros(N)
    for i in range(N):
        abs_err_deg[i] = np.mean(np.abs(p_deg[i] - test_targets_np[:, i]))

    # ---- Degree bins (same as train_gnn) ----
    bins = np.quantile(deg, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.unique(bins)
    if bins.size < 3:
        bins = np.array([deg.min(), np.median(deg), deg.max()])

    bin_labels = []
    bin_err_gnn = []
    bin_err_deg = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (deg >= b0) & (deg <= b1)
        if mask.sum() == 0:
            continue
        bin_labels.append(f"{b0:.0f}-{b1:.0f}")
        bin_err_gnn.append(np.mean(abs_err_gnn[mask]))
        bin_err_deg.append(np.mean(abs_err_deg[mask]))

    # ---- Figures ----
    os.makedirs("figures", exist_ok=True)

    # A) Error vs degree comparison
    x_bin = np.arange(len(bin_err_gnn))
    plt.figure(figsize=(6, 4))
    plt.plot(x_bin, bin_err_gnn, marker="o", linewidth=2, markersize=8, label="GNN")
    plt.plot(x_bin, bin_err_deg, marker="s", linewidth=2, markersize=8, label="Degree-only baseline")
    plt.xticks(x_bin, bin_labels, rotation=30, ha="right")
    plt.xlabel("degree bin")
    plt.ylabel("mean |p - y|")
    plt.title("Prediction error vs node degree (test)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/error_vs_degree_comparison.png", dpi=200)
    plt.close()

    # B) BCE comparison bar plot
    plt.figure(figsize=(5, 4))
    plt.bar([0], [gnn_test_bce], width=0.5, color="C0")
    plt.bar([1], [degree_test_bce], width=0.5, color="C1")
    plt.xticks([0, 1], ["GNN", "Degree-only"])
    plt.ylabel("BCE loss")
    plt.title("Test BCE: GNN vs degree-only baseline")
    plt.tight_layout()
    plt.savefig("figures/bce_comparison.png", dpi=200)
    plt.close()

    # ---- Print BCE ----
    print("Test BCE (same test time steps):")
    print(f"  GNN:             {gnn_test_bce:.4f}")
    print(f"  Degree-only:     {degree_test_bce:.4f}")
    print("\nSaved figures:")
    print("  figures/error_vs_degree_comparison.png")
    print("  figures/bce_comparison.png")


if __name__ == "__main__":
    main()
