"""
Minimal experimental pipeline for diffusion dynamics on graphs.
Implements graph loading and Markov diffusion simulation.
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import os
import urllib.request
import gzip


def download_email_eu_core():
    """Load Email-Eu-Core dataset from local file or download if not present."""
    # Check for user-provided dataset first (multiple possible locations/names)
    possible_datasets = [
        os.path.join("dataset", "email-Eu-core.txt"),
        os.path.join("dataset", "Email-EuAll.txt"),
        os.path.join("dataset", "email-EuAll.txt"),
    ]
    for user_dataset in possible_datasets:
        if os.path.exists(user_dataset):
            print(f"Using dataset from: {user_dataset}")
            return user_dataset
    
    # Otherwise check standard location
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    edge_file = os.path.join(data_dir, "email-Eu-core.txt")
    
    if not os.path.exists(edge_file):
        print("Downloading Email-Eu-Core dataset...")
        url = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
        gz_file = os.path.join(data_dir, "email-Eu-core.txt.gz")
        
        try:
            urllib.request.urlretrieve(url, gz_file)
            
            with gzip.open(gz_file, 'rt', encoding='utf-8') as f_in:
                with open(edge_file, 'w', encoding='utf-8') as f_out:
                    f_out.write(f_in.read())
            
            os.remove(gz_file)
            print(f"Dataset downloaded to {edge_file}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please download manually from: https://snap.stanford.edu/data/email-Eu-core.html")
            raise
    
    return edge_file


def load_graph():
    """
    Step 1: Graph loading
    - Load Email-Eu-Core edge list
    - Build undirected NetworkX graph
    - Extract largest connected component
    - Store adjacency in PyTorch Geometric format
    """
    print("Step 1: Loading graph...")
    
    # Download dataset if needed
    edge_file = download_email_eu_core()
    
    # Load edge list (skip comment lines and header)
    edges = []
    with open(edge_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle both space-separated and tab-separated formats
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        # Skip self-loops if present
                        if u != v:
                            edges.append((u, v))
                    except ValueError:
                        # Skip header lines that can't be converted to int
                        continue
    
    # Build undirected NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edges)
    print(f"  Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Extract largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc).copy()
    print(f"  Largest CC: {G_lcc.number_of_nodes()} nodes, {G_lcc.number_of_edges()} edges")
    
    # Create node mapping to consecutive indices [0, n-1]
    node_mapping = {node: idx for idx, node in enumerate(sorted(G_lcc.nodes()))}
    G_mapped = nx.relabel_nodes(G_lcc, node_mapping)
    
    # Convert to PyTorch Geometric format
    edge_index = torch.tensor(list(G_mapped.edges()), dtype=torch.long).t().contiguous()
    
    # Create bidirectional edges (undirected graph)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    data = Data(edge_index=edge_index, num_nodes=G_mapped.number_of_nodes())
    
    print(f"  PyTorch Geometric data: {data.num_nodes} nodes, {edge_index.shape[1]} edges")
    
    return G_mapped, data


def markov_diffusion_simulation(graph, data, beta=0.01, initial_fraction=0.01, T=25, seed=42):
    """
    Step 2: Markov diffusion simulation
    
    Binary diffusion process:
    - Each node has state x_i^t in {0, 1}
    - Initialize small random fraction of active nodes
    - At each time step:
      - Inactive node becomes active with probability: 1 - ∏_{j∈N(i)} (1 - β*x_j^t)
      - Active nodes remain active
    - Run for T time steps
    - Store state trajectories x_t
    """
    print("\nStep 2: Running Markov diffusion simulation...")
    print(f"  Parameters: β={beta}, initial_fraction={initial_fraction}, T={T}")
    
    np.random.seed(seed)
    n_nodes = graph.number_of_nodes()
    
    # Initialize: small random fraction of active nodes
    n_initial = max(1, int(initial_fraction * n_nodes))
    initial_active = np.random.choice(n_nodes, size=n_initial, replace=False)
    
    # State trajectory: x[t, i] = state of node i at time t
    x = np.zeros((T + 1, n_nodes), dtype=int)
    x[0, initial_active] = 1
    
    print(f"  Initial active nodes: {n_initial} ({100*n_initial/n_nodes:.2f}%)")
    
    # Build adjacency list for efficient neighbor access
    adj_list = {i: list(graph.neighbors(i)) for i in range(n_nodes)}
    
    # Run simulation for T time steps
    for t in range(T):
        x_next = x[t].copy()  # Active nodes remain active
        
        # Update inactive nodes
        for i in range(n_nodes):
            if x[t, i] == 0:  # Only process inactive nodes
                neighbors = adj_list[i]
                if len(neighbors) > 0:
                    # Probability of not being activated = ∏_{j∈N(i)} (1 - β*x_j^t)
                    prob_not_activated = 1.0
                    for j in neighbors:
                        if x[t, j] == 1:  # If neighbor j is active
                            prob_not_activated *= (1 - beta)
                    
                    # Activation probability = 1 - prob_not_activated
                    activation_prob = 1 - prob_not_activated
                    
                    # Sample activation
                    if np.random.random() < activation_prob:
                        x_next[i] = 1
        
        x[t + 1] = x_next
        
        # Log progress
        active_count = np.sum(x[t + 1])
        active_ratio = active_count / n_nodes
        
        # Safety guard: stop early if saturation reached
        if active_ratio > 0.9:
            print(f"  WARNING: Stopping early at t={t+1} - saturation reached ({100*active_ratio:.2f}%)")
            # Fill remaining time steps with current state
            for remaining_t in range(t + 2, T + 1):
                x[remaining_t] = x[t + 1]
            break
        
        if (t + 1) % 5 == 0 or t == 0:
            print(f"  t={t+1:3d}: {active_count:4d} active nodes ({100*active_ratio:5.2f}%)")
    
    final_active = np.sum(x[T])
    print(f"  Final: {final_active:4d} active nodes ({100*final_active/n_nodes:5.2f}%)")
    
    return x


if __name__ == "__main__":
    # Step 1: Load graph
    graph, data = load_graph()
    
    # Step 2: Run Markov diffusion simulation
    # Parameters tuned for gradual diffusion (20-60% final active, smooth S-curve)
    trajectories = markov_diffusion_simulation(
        graph, data, 
        beta=0.01,          # Reduced: infection probability per active neighbor
        initial_fraction=0.01,  # Reduced: 1% of nodes initially active
        T=25,               # Reduced: 25 time steps (sufficient for local transitions)
        seed=42
    )
    
    # Save trajectories for later use
    np.save("trajectories.npy", trajectories)
    print("\nTrajectories saved to trajectories.npy")
    print(f"Shape: {trajectories.shape} (time_steps, nodes)")

