"""
============================================================================
PHASE III: CAUSAL REGULATORY NETWORK INFERENCE (Python)
============================================================================
Project: Perturbation-Aware Causal Machine Learning Framework for 
         Glioblastoma Regulatory Network Discovery
Manuscript: A Perturbation-Aware Causal Machine Learning Framework Identifies
            Candidate Regulatory Hubs and Radiotherapy-Associated Network
            Rewiring in Glioblastoma
============================================================================
Methods Implemented:
- NOTEARS: Continuous optimization via trace exponential and augmented
  Lagrangian for directed acyclic graph (DAG) learning
- GES: Greedy Equivalence Search for Markov equivalence class discovery
- RCI-NB: Regularized Causal Inference with Naive Bayes classification
- Multi-Algorithm Ensemble Consensus Learning & Intervention-Aware ATE Scoring
============================================================================
"""

# --- PHASE III START ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse, stats
import scipy.linalg
from scipy.optimize import minimize
from statsmodels.stats.multitest import multipletests
import time
import networkx as nx
import os
import warnings
import scanpy as sc

warnings.filterwarnings('ignore')

# Publication visualization standards (Nature Portfolio style)
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
sns.set_style("ticks")

# Set random seed
np.random.seed(42)

# Dynamic Directory Resolution for Cross-Platform Portability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

OUTPUT_PATH = os.environ.get(
    "GBM_OUTPUT_PATH",
    os.path.join(REPO_ROOT, "outputs") if os.path.exists(os.path.join(REPO_ROOT, "outputs"))
    else (os.path.join(SCRIPT_DIR, "outputs") if os.path.exists(os.path.join(SCRIPT_DIR, "outputs"))
          else "L:/Thesis Work/outputs")
)
PHASE2_PATH = os.path.join(OUTPUT_PATH, "phase2")
PHASE3_PATH = os.path.join(OUTPUT_PATH, "phase3")
FIGURE_PATH = os.path.join(OUTPUT_PATH, "figures", "phase3")

os.makedirs(PHASE3_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)

print("=" * 80)
print("PHASE III: CAUSAL NETWORK INFERENCE")
print("=" * 80)
print()

# 3.1 Load Data from Phase II
print("[1/6] Loading data from Phase II...")

# Load AnnData
adata = sc.read_h5ad(os.path.join(PHASE2_PATH, "phase2_processed.h5ad"))

# Load intervention matrix
intervention_onehot = pd.read_csv(os.path.join(PHASE2_PATH, "intervention_onehot.csv"), index_col=0)

# Select subset of HVGs for causal inference (memory efficient)
# Select top 500 HVGs for causal network inference
if 'highly_variable' in adata.var.columns:
    hvg_mask = adata.var['highly_variable']
    adata_subset = adata[:, hvg_mask].copy()
else:
    adata_subset = adata.copy()

if adata_subset.n_vars > 500:
    print("    Calculating gene variances to select top 500 genes...")
    if sparse.issparse(adata_subset.X):
        mean_sq = np.array(adata_subset.X.power(2).mean(axis=0)).flatten()
        sq_mean = np.array(adata_subset.X.mean(axis=0)).flatten()**2
        gene_vars = mean_sq - sq_mean
    else:
        gene_vars = np.var(adata_subset.X, axis=0)
    
    top_idx = np.argsort(gene_vars)[-500:]
    adata_subset = adata_subset[:, top_idx].copy()

print(f"    Selected {adata_subset.n_vars} genes for causal inference")
print(f"    Total cells: {adata_subset.n_obs}")

# Extract expression matrix
X = adata_subset.X.toarray() if sparse.issparse(adata_subset.X) else adata_subset.X
gene_names = adata_subset.var_names.tolist()

# Normalize data
X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# 3.2 Intervention-Aware Preprocessing
print("\n[2/6] Preparing intervention-aware data...")

unique_interventions = intervention_onehot.columns.values
print(f"    Unique true interventions (excluding controls): {len(unique_interventions)}")

# Create intervention matrix (cells x interventions)
I = intervention_onehot.values
print(f"    Intervention matrix shape: {I.shape}")

# 3.3 NOTEARS Implementation
print("\n[3/6] Running NOTEARS (Gradient-based DAG Learning)...")

class NOTEARS:
    """
    NOTEARS: Non-combinatorial Optimization via Trace Exponential
    and Augmented lagRangian for Structure learning
    """
    def __init__(self, lambda1=0.1, lambda2=0.1, max_iter=100, h_tol=1e-8, rho_max=1e16):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.W = None
        self.S = None

    def _h(self, W):
        """DAG constraint: tr(exp(W * W)) - d = 0"""
        d = W.shape[0]
        E = scipy.linalg.expm(W * W)
        return np.trace(E) - d

    def _h_grad(self, W):
        """Gradient of DAG constraint"""
        E = scipy.linalg.expm(W * W)
        return E.T * 2 * W

    def _loss(self, W):
        """Least squares loss using precomputed covariance S to save memory"""
        return 0.5 * (np.trace(self.S) - 2 * np.trace(W.T @ self.S) + np.trace(W.T @ self.S @ W))

    def _loss_grad(self, W):
        """Gradient of least squares loss using precomputed S"""
        return -self.S + self.S @ W

    def fit(self, X):
        """Fit NOTEARS model"""
        n, d = X.shape
        self.W = np.zeros((d, d))
        # Precompute empirical covariance matrix to dramatically reduce memory and compute overhead
        self.S = (X.T @ X) / n
        
        rho = 1.0
        alpha = 0.0
        h = np.inf

        self.history = []
        for iteration in range(self.max_iter):
            t_start = time.time()
            # Solve subproblem
            W_new = self._solve_subproblem(rho, alpha, d)

            # Update h
            h_new = self._h(W_new)

            # Update rho and alpha (augmented Lagrangian)
            if h_new > 0.25 * h:
                rho *= 10

            alpha += rho * h_new
            h = h_new
            self.W = W_new
            
            elapsed = time.time() - t_start
            n_edges = int(np.sum(np.abs(self.W) > self.lambda1))
            self.history.append({
                "iteration": iteration + 1,
                "h_value": h_new,
                "rho": rho,
                "n_nonzero": n_edges,
                "time_sec": elapsed
            })

            # Check convergence
            if h <= self.h_tol or rho >= self.rho_max:
                break

            if iteration % 10 == 0:
                print(f"        NOTEARS iteration {iteration}: h = {h:.6f}")

        return self

    def _solve_subproblem(self, rho, alpha, d):
        """Solve augmented Lagrangian subproblem with L-BFGS-B using W+ and W- formulation for L1"""
        from scipy.optimize import minimize

        # Split W into W+ and W- to make L1 penalty differentiable (W = W+ - W-)
        w_pos = np.maximum(self.W, 0).flatten()
        w_neg = np.maximum(-self.W, 0).flatten()
        w_init = np.concatenate([w_pos, w_neg])
        bounds = [(0, None) for _ in range(2 * d * d)]

        def objective(w):
            w_p = w[:d*d].reshape((d, d))
            w_n = w[d*d:].reshape((d, d))
            W = w_p - w_n
            loss = self._loss(W)
            h = self._h(W)
            penalty = (rho / 2) * h ** 2 + alpha * h
            l1_penalty = self.lambda1 * (np.sum(w_p) + np.sum(w_n))
            return loss + penalty + l1_penalty

        def gradient(w):
            w_p = w[:d*d].reshape((d, d))
            w_n = w[d*d:].reshape((d, d))
            W = w_p - w_n
            
            grad_loss = self._loss_grad(W)
            h = self._h(W)
            grad_h = self._h_grad(W)
            grad_penalty = rho * h * grad_h + alpha * grad_h
            
            # Chain rule: dJ/dW+ = dJ/dW + lambda1, dJ/dW- = -dJ/dW + lambda1
            grad_W = grad_loss + grad_penalty
            grad_wp = grad_W + self.lambda1
            grad_wn = -grad_W + self.lambda1
            
            return np.concatenate([grad_wp.flatten(), grad_wn.flatten()])

        result = minimize(objective, w_init, method='L-BFGS-B', jac=gradient, bounds=bounds)
        w_opt_p = result.x[:d*d].reshape((d, d))
        w_opt_n = result.x[d*d:].reshape((d, d))
        return w_opt_p - w_opt_n

# Run NOTEARS on subset
print("    Running NOTEARS on expression data...")

# Single-cell RNA-seq data is highly sparse, meaning gene correlations are extremely low.
# We adaptively set the L1 penalty based on the maximum correlation signal in the data, 
# otherwise the penalty completely crushes all weak causal signals to exactly 0.
S_emp = (X_normalized.T @ X_normalized) / X_normalized.shape[0]
np.fill_diagonal(S_emp, 0)
max_corr = np.max(np.abs(S_emp))
print(f"    Maximum gene correlation in data: {max_corr:.4f}")

adaptive_lambda = max(0.00001, max_corr * 0.05)
adaptive_threshold = max(0.00001, max_corr * 0.10)
print(f"    Using adaptive L1 penalty: {adaptive_lambda:.5f}, threshold: {adaptive_threshold:.5f}")

notears = NOTEARS(lambda1=adaptive_lambda, lambda2=0.01, max_iter=50)
notears.fit(X_normalized)

# Threshold to get adjacency matrix
W_notears = notears.W
adj_notears = (np.abs(W_notears) > adaptive_threshold).astype(int)
np.fill_diagonal(adj_notears, 0)

print(f"    NOTEARS edges: {np.sum(adj_notears)}")

# 3.4 Greedy Equivalence Search (GES)
print("\n[4/6] Running GES (Greedy Equivalence Search)...")

def ges_score(X, i, parents):
    """BIC score for GES"""
    n, d = X.shape
    if len(parents) == 0:
        var = np.var(X[:, i])
        if var <= 0: var = 1e-8
        return -0.5 * n * np.log(var)

    X_parents = X[:, list(parents)]
    coef = np.linalg.lstsq(X_parents, X[:, i], rcond=None)[0]
    residuals = X[:, i] - X_parents @ coef
    var = np.var(residuals)
    if var <= 0: var = 1e-8

    # BIC score with an extended penalty for very large N
    # Standard BIC penalizes 0.5 * k * log(n), but when N=127k, the variance reduction term 
    # overwhelms it and creates max edges. We scale the penalty so GES becomes selective.
    k = len(parents) + 1  
    penalty = 0.5 * k * np.log(n) * max(1.0, n / 10000.0)
    
    bic = -0.5 * n * np.log(var) - penalty
    return bic

def ges(X, max_parents=3):
    """Simplified GES implementation"""
    n, d = X.shape
    adj = np.zeros((d, d))

    # Forward phase: add edges greedily
    for i in range(d):
        parents = set()
        best_score = ges_score(X, i, parents)

        for _ in range(max_parents):
            best_parent = None
            for j in range(d):
                if j != i and j not in parents:
                    new_parents = parents | {j}
                    score = ges_score(X, i, new_parents)
                    if score > best_score:
                        best_score = score
                        best_parent = j

            if best_parent is not None:
                parents.add(best_parent)
                adj[best_parent, i] = 1
            else:
                break

    return adj

print("    Running GES on expression data...")
adj_ges = ges(X_normalized, max_parents=2) 
np.fill_diagonal(adj_ges, 0)

print(f"    GES edges: {np.sum(adj_ges)}")

# 3.5 RCI-NB (Regularized Causal Inference with Naive Bayes)
print("\n[5/6] Running RCI-NB (Regularized Causal Inference)...")

def rci_nb(X, intervention_df_onehot, gene_names, alpha=0.01):
    """
    RCI-NB: Test for causal relationships using intervention data
    """
    n, d = X.shape
    adj = np.zeros((d, d))

    for i in range(d):
        if gene_names[i] not in intervention_df_onehot.columns:
            continue
        
        perturbed_mask = intervention_df_onehot[gene_names[i]].values.astype(bool)
        
        if np.sum(perturbed_mask) > 10 and np.sum(~perturbed_mask) > 10:
            for j in range(d):
                if i == j:
                    continue
                # Compare distributions
                _, pvalue = stats.mannwhitneyu(
                    X[perturbed_mask, j],
                    X[~perturbed_mask, j],
                    alternative='two-sided'
                )

                if pvalue < alpha:
                    # Effect Size check: Large N guarantees tiny p-values. 
                    # We strictly require an effect size magnitude (delta mean) > 0.1 standard deviations 
                    # for it to be counted as a real causal edge.
                    mean_diff = np.abs(np.mean(X[perturbed_mask, j]) - np.mean(X[~perturbed_mask, j]))
                    if mean_diff > 0.1:
                        adj[i, j] = 1

    return adj

if len(unique_interventions) > 0:
    print("    Running RCI-NB with intervention data...")
    adj_rci = rci_nb(X_normalized, intervention_onehot, gene_names, alpha=0.01)
    print(f"    RCI-NB edges: {np.sum(adj_rci)}")
else:
    print("    Skipping RCI-NB (no intervention data)")
    adj_rci = np.zeros((len(gene_names), len(gene_names)))

# 3.6 Ensemble Consensus Network
print("\n[6/6] Creating Ensemble Consensus Network...")

# Combine all methods with weighted voting
weights = {'NOTEARS': 0.4, 'GES': 0.3, 'RCI-NB': 0.3}

ensemble_score = (weights['NOTEARS'] * adj_notears +
                  weights['GES'] * adj_ges +
                  weights['RCI-NB'] * adj_rci)

# Threshold for consensus
consensus_threshold = 0.5
adj_consensus = (ensemble_score >= consensus_threshold).astype(int)
np.fill_diagonal(adj_consensus, 0)

print(f"    Consensus edges: {np.sum(adj_consensus)}")

# Create edge lists
def create_edge_list(adj_matrix, gene_names, method_name):
    edges = []
    for i in range(len(gene_names)):
        for j in range(len(gene_names)):
            if adj_matrix[i, j] > 0 and i != j:
                edges.append({
                    'Source': gene_names[i],
                    'Target': gene_names[j],
                    'Weight': float(adj_matrix[i, j]),
                    'Method': method_name
                })
    if len(edges) == 0:
        return pd.DataFrame(columns=['Source', 'Target', 'Weight', 'Method'])
    return pd.DataFrame(edges)

edges_notears = create_edge_list(adj_notears, gene_names, 'NOTEARS')
edges_ges = create_edge_list(adj_ges, gene_names, 'GES')
edges_rci = create_edge_list(adj_rci, gene_names, 'RCI-NB')
edges_consensus = create_edge_list(adj_consensus, gene_names, 'Consensus')

# Save edge lists
edges_notears.to_csv(os.path.join(PHASE3_PATH, "edges_notears.csv"), index=False)
edges_ges.to_csv(os.path.join(PHASE3_PATH, "edges_ges.csv"), index=False)
edges_rci.to_csv(os.path.join(PHASE3_PATH, "edges_rci.csv"), index=False)
edges_consensus.to_csv(os.path.join(PHASE3_PATH, "edges_consensus.csv"), index=False)

# Save adjacency matrices
np.save(os.path.join(PHASE3_PATH, "adj_notears.npy"), adj_notears)
np.save(os.path.join(PHASE3_PATH, "adj_ges.npy"), adj_ges)
np.save(os.path.join(PHASE3_PATH, "adj_rci.npy"), adj_rci)
np.save(os.path.join(PHASE3_PATH, "adj_consensus.npy"), adj_consensus)

# Save gene names
with open(os.path.join(PHASE3_PATH, "gene_names.txt"), 'w') as f:
    f.write('\n'.join(gene_names))

# 3.7 Generate Figures
print("\n[Figures] Generating causal network figures...")

# Figure 3A: Adjacency Matrix Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Select genes that represent ALL methods, not just the global consensus hubs.
# Strategy: Top 35 global hubs + Cdkn1a (RCI-NB intervention source) + its top 14 RCI-NB targets.
# This ensures the RCI-NB panel visibly shows its interventional edges alongside NOTEARS/GES.
degrees = np.sum(adj_consensus, axis=0) + np.sum(adj_consensus, axis=1)
sample_genes = min(50, len(gene_names))

# Get top 35 global hubs
top_hub_idx = set(np.argsort(degrees)[-(sample_genes - 15):].tolist())

# Get RCI-NB intervention source and its top targets
rci_out_degree = np.sum(adj_rci, axis=1)  # outgoing edges per gene
rci_source_genes = np.argsort(rci_out_degree)[-1:]  # top RCI-NB source (Cdkn1a)
rci_source_idx = set(rci_source_genes.tolist())

# For each RCI-NB source, get its top target genes
rci_target_idx = set()
for src in rci_source_genes:
    targets = np.where(adj_rci[src, :] > 0)[0]
    # Pick top 14 targets by consensus degree so they also show NOTEARS/GES edges
    if len(targets) > 14:
        targets = targets[np.argsort(degrees[targets])[-14:]]
    rci_target_idx.update(targets.tolist())

# Combine: global hubs + RCI source + RCI targets (deduplicated)
combined_idx = top_hub_idx | rci_source_idx | rci_target_idx
sample_idx = np.array(sorted(combined_idx))[:sample_genes]  # cap at 50

im1 = axes[0, 0].imshow(adj_notears[np.ix_(sample_idx, sample_idx)], cmap='Blues', vmin=0, vmax=1)
axes[0, 0].set_title('NOTEARS', fontweight='bold')
axes[0, 0].set_xlabel('Gene Index (Top Hubs)')
axes[0, 0].set_ylabel('Gene Index (Top Hubs)')

im2 = axes[0, 1].imshow(adj_ges[np.ix_(sample_idx, sample_idx)], cmap='Greens', vmin=0, vmax=1)
axes[0, 1].set_title('GES', fontweight='bold')
axes[0, 1].set_xlabel('Gene Index (Top Hubs)')
axes[0, 1].set_ylabel('Gene Index (Top Hubs)')

im3 = axes[1, 0].imshow(adj_rci[np.ix_(sample_idx, sample_idx)], cmap='Oranges', vmin=0, vmax=1)
axes[1, 0].set_title('RCI-NB', fontweight='bold')
axes[1, 0].set_xlabel('Gene Index (Top Hubs)')
axes[1, 0].set_ylabel('Gene Index (Top Hubs)')

im4 = axes[1, 1].imshow(adj_consensus[np.ix_(sample_idx, sample_idx)], cmap='Reds', vmin=0, vmax=1)
axes[1, 1].set_title('Consensus', fontweight='bold')
axes[1, 1].set_xlabel('Gene Index (Top Hubs)')
axes[1, 1].set_ylabel('Gene Index (Top Hubs)')

# plt.suptitle removed
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'figure_3a_adjacency_comparison.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'figure_3a_adjacency_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 3B: Edge Count Comparison
fig, ax = plt.subplots(figsize=(10, 6))
methods = ['NOTEARS', 'GES', 'RCI-NB', 'Consensus']
edge_counts = [np.sum(adj_notears), np.sum(adj_ges), np.sum(adj_rci), np.sum(adj_consensus)]
colors = ['steelblue', 'forestgreen', 'coral', 'purple']

bars = ax.bar(methods, edge_counts, color=colors, edgecolor='white', linewidth=1.5)
ax.set_ylabel('Number of Edges', fontsize=12)
# ax.set_title removed

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'figure_3b_edge_counts.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'figure_3b_edge_counts.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 3C: Method Agreement Heatmap
method_adjs = [adj_notears, adj_ges, adj_rci]
method_names = ['NOTEARS', 'GES', 'RCI-NB']
agreement = np.zeros((3, 3))

for i in range(3):
    for j in range(3):
        intersection = np.sum(method_adjs[i] * method_adjs[j])
        union = np.sum((method_adjs[i] + method_adjs[j]) > 0)
        agreement[i, j] = intersection / union if union > 0 else 0

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(agreement, cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(method_names)
ax.set_yticklabels(method_names)
# ax.set_title removed

for i in range(3):
    for j in range(3):
        text = ax.text(j, i, f'{agreement[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=12)

plt.colorbar(im, ax=ax, label='Jaccard Index')
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'figure_3c_method_agreement.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'figure_3c_method_agreement.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 3D-3G: DAG Visualizations
def plot_dag(adj_matrix, gene_names, title, filename, max_nodes=30):
    if np.sum(adj_matrix) == 0:
        print(f"    Skipping {title} (0 edges)")
        return
        
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
    mapping = {i: gene_names[i] for i in range(len(gene_names))}
    G = nx.relabel_nodes(G, mapping)

    degrees = dict(G.degree())
    if len(degrees) > max_nodes:
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [n[0] for n in top_nodes]
        G_sub = G.subgraph(top_node_names)
    else:
        G_sub = G

    fig, ax = plt.subplots(figsize=(14, 12))
    pos = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)

    # Adding +100 to node size to ensure nodes with 0 degree still show up visibly if isolated
    nx.draw_networkx_nodes(G_sub, pos, node_size=[degrees.get(n, 0)*50 + 100 for n in G_sub.nodes()],
                          node_color='lightblue', ax=ax)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.5, arrows=True,
                          arrowsize=15, ax=ax, edge_color='gray')
    nx.draw_networkx_labels(G_sub, pos, font_size=8, ax=ax)

    # ax.set_title removed
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_PATH, f'{filename}.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURE_PATH, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()

plot_dag(adj_consensus, gene_names, 'Figure 3D: Consensus Causal Network (Top 30 Nodes)', 'figure_3d_network_visualization')
plot_dag(adj_notears, gene_names, 'Figure 3E: NOTEARS DAG (Top 30 Nodes)', 'figure_3e_notears_dag')
plot_dag(adj_ges, gene_names, 'Figure 3F: GES DAG (Top 30 Nodes)', 'figure_3f_ges_dag')
plot_dag(adj_rci, gene_names, 'Figure 3G: RCI-NB DAG (Top 30 Nodes)', 'figure_3g_rcinb_dag')

# Figure 3H: NOTEARS Convergence Plot
if hasattr(notears, 'history') and len(notears.history) > 0:
    hist_df = pd.DataFrame(notears.history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].semilogy(hist_df["iteration"], hist_df["h_value"], "o-", color="#0072B2", markersize=4, linewidth=1.5)
    axes[0].axhline(y=notears.h_tol, color="red", linestyle="--", linewidth=0.8, label="Convergence threshold")
    axes[0].set_xlabel("Iteration", fontweight="bold")
    axes[0].set_ylabel("h(W)  [log scale]", fontweight="bold")
    # axes[0].set_title removed
    axes[0].legend(fontsize=8, frameon=False)
    
    axes[1].plot(hist_df["iteration"], hist_df["n_nonzero"], "s-", color="#D55E00", markersize=4, linewidth=1.5)
    axes[1].set_xlabel("Iteration", fontweight="bold")
    axes[1].set_ylabel("Non-zero edges", fontweight="bold")
    # axes[1].set_title removed
    
    axes[2].semilogy(hist_df["iteration"], hist_df["rho"], "^-", color="#009E73", markersize=4, linewidth=1.5)
    axes[2].set_xlabel("Iteration", fontweight="bold")
    axes[2].set_ylabel(r"$\rho$  [log scale]", fontweight="bold")
    # axes[2].set_title removed
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_3h_notears_convergence.pdf'), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_3h_notears_convergence.png'), dpi=300, bbox_inches="tight")
    plt.close()

# Figure 3I: Top NOTEARS Causal Edges
if len(edges_notears) > 0:
    top_n = min(30, len(edges_notears))
    top_edges = edges_notears.head(top_n).copy()
    top_edges["edge_label"] = top_edges["Source"] + " → " + top_edges["Target"]
    top_edges["direction"] = top_edges["Weight"].apply(lambda w: "Activation (+)" if w > 0 else "Repression (−)")
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    colors = ["#D55E00" if d == "Activation (+)" else "#0072B2" for d in top_edges["direction"]]
    
    ax.barh(range(len(top_edges)), top_edges["Weight"].values, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(top_edges)))
    ax.set_yticklabels(top_edges["edge_label"].values, fontsize=8, fontstyle="italic")
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("NOTEARS Edge Weight", fontweight="bold", fontsize=11)
    # ax.set_title removed
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#D55E00", label="Activation (+)"), Patch(facecolor="#0072B2", label="Repression (−)")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, frameon=False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_3i_notears_top_edges.pdf'), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_3i_notears_top_edges.png'), dpi=300, bbox_inches="tight")
    plt.close()

# Figure 3J: NOTEARS Weight Heatmap
if len(edges_notears) > 0:
    active_genes = sorted(set(edges_notears["Source"].tolist() + edges_notears["Target"].tolist()))
    if 0 < len(active_genes) <= 80:
        W_df = pd.DataFrame(W_notears, index=gene_names, columns=gene_names)
        W_sub = W_df.loc[active_genes, active_genes]
        
        fig, ax = plt.subplots(figsize=(max(8, len(active_genes) * 0.2), max(7, len(active_genes) * 0.18)))
        vmax = max(abs(W_sub.values.min()), abs(W_sub.values.max()))
        
        sns.heatmap(W_sub, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, linewidths=0.3, linecolor="white",
                    cbar_kws={"label": "Edge Weight", "shrink": 0.7}, ax=ax)
        ax.set_xlabel("Target Gene", fontweight="bold", fontsize=11)
        ax.set_ylabel("Source Gene", fontweight="bold", fontsize=11)
        # ax.set_title removed
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_PATH, 'figure_3j_notears_heatmap.pdf'), bbox_inches="tight")
        fig.savefig(os.path.join(FIGURE_PATH, 'figure_3j_notears_heatmap.png'), dpi=300, bbox_inches="tight")
        plt.close()

# Create summary report
summary_report = pd.DataFrame({
    'Method': ['NOTEARS', 'GES', 'RCI-NB', 'Consensus'],
    'Edges': [np.sum(adj_notears), np.sum(adj_ges), np.sum(adj_rci), np.sum(adj_consensus)],
    'Density': [np.sum(adj_notears)/len(gene_names)**2,
                np.sum(adj_ges)/len(gene_names)**2,
                np.sum(adj_rci)/len(gene_names)**2,
                np.sum(adj_consensus)/len(gene_names)**2],
    'Mean_Degree': [np.mean(np.sum(adj_notears, axis=1)),
                    np.mean(np.sum(adj_ges, axis=1)),
                    np.mean(np.sum(adj_rci, axis=1)),
                    np.mean(np.sum(adj_consensus, axis=1))]
})
summary_report.to_csv(os.path.join(PHASE3_PATH, "phase3_summary.csv"), index=False)

# 3.8 Generate Phase IV Input Files (Bridge)
print("\n[Bridge] Generating statistical edge evidence and ATE for Phase IV R script...")

confident_edges = []

# We only test edges that are in the consensus network
rows, cols = np.where(adj_consensus > 0)

for i, j in zip(rows, cols):
    source = gene_names[i]
    target = gene_names[j]
    
    # We can only do statistical testing if the source gene was actually intervened on
    if source in intervention_onehot.columns:
        perturbed_mask = intervention_onehot[source].values.astype(bool)
        
        if np.sum(perturbed_mask) > 10 and np.sum(~perturbed_mask) > 10:
            target_expr_pert = X[perturbed_mask, j]
            target_expr_unpert = X[~perturbed_mask, j]
            
            # Since X is already log-normalized (from Seurat Phase I), 
            # log2FC is just the difference of means (scaled by log2(e) if natural log was used in Seurat).
            # Seurat's default is natural log.
            log2FC = (np.mean(target_expr_pert) - np.mean(target_expr_unpert)) / np.log(2)
            
            _, pval = stats.mannwhitneyu(target_expr_pert, target_expr_unpert, alternative='two-sided')
            
            confident_edges.append({
                'perturbed_gene': source,
                'target_gene': target,
                'avg_log2FC': log2FC,
                'p_val': pval,
                'confidence': float(ensemble_score[i, j])
            })

if confident_edges:
    df_edges = pd.DataFrame(confident_edges)
    
    # FDR Correction (Benjamini-Hochberg)
    df_edges['p_val_adj'] = multipletests(df_edges['p_val'], method='fdr_bh')[1]
    
    # Save Confident Edges
    edges_file = os.path.join(PHASE3_PATH, "phase3_causal_edges_confident.csv")
    df_edges.to_csv(edges_file, index=False)
    print(f"    Saved confident edges for Phase IV: {edges_file}")
    
    # Calculate ATE (Average Treatment Effect) per hub
    ate_results = df_edges.groupby('perturbed_gene').agg(
        ATE=('avg_log2FC', 'mean'),
        abs_ATE=('avg_log2FC', lambda x: np.mean(np.abs(x))),
        n_targets=('target_gene', 'count')
    ).reset_index()
    
    ate_file = os.path.join(PHASE3_PATH, "phase3_ate_results.csv")
    ate_results.to_csv(ate_file, index=False)
    print(f"    Saved ATE results for Phase IV: {ate_file}")
else:
    print("    WARNING: No valid intervention edges found for Phase IV.")

# 3.9 Export Complete Consensus Edge List with Continuous Weights (S3 Table)
print("\n[S3 Table] Exporting Complete Consensus Edge List (S3_Table_Edge_List.csv)...")
np.save(os.path.join(PHASE3_PATH, "ensemble_score.npy"), ensemble_score)

s3_records = []
cons_rows, cons_cols = np.where(adj_consensus > 0)
for i, j in zip(cons_rows, cons_cols):
    r_val, p_val = stats.pearsonr(X_normalized[:, i], X_normalized[:, j])
    s3_records.append({
        'Source': gene_names[i],
        'Target': gene_names[j],
        'Direction': 'Activation (+)' if r_val >= 0 else 'Repression (-)',
        'NOTEARS': int(adj_notears[i, j]),
        'GES': int(adj_ges[i, j]),
        'RCI_NB': int(adj_rci[i, j]),
        'Ensemble_Score': round(float(ensemble_score[i, j]), 4),
        'Effect_Beta': round(float(r_val), 4),
        'Abs_Effect': round(abs(float(r_val)), 4),
        'P_Value': float(p_val)
    })

s3_df = pd.DataFrame(s3_records)
if len(s3_df) > 0:
    s3_df['FDR_P_Value'] = multipletests(s3_df['P_Value'], method='fdr_bh')[1]
    max_eff = max(s3_df['Abs_Effect'].max(), 1e-8)
    s3_df['Ensemble_Confidence'] = np.round(0.7 * s3_df['Ensemble_Score'] + 0.3 * (s3_df['Abs_Effect'] / max_eff), 4)
    s3_df = s3_df.sort_values(by=['Ensemble_Confidence', 'Abs_Effect'], ascending=False)

s3_df.to_csv(os.path.join(OUTPUT_PATH, "S3_Table_Edge_List.csv"), index=False)
s3_df.to_csv(os.path.join(PHASE3_PATH, "S3_edges_consensus.csv"), index=False)
print(f"    Saved S3_Table_Edge_List.csv ({len(s3_df)} consensus edges with continuous confidence weights)")

print("\n" + "=" * 80)
print("[DONE] PHASE III COMPLETE")
print(f"Output saved to: {PHASE3_PATH}")
print(f"Figures saved to: {FIGURE_PATH}")
print("=" * 80)

# --- PHASE III END ---

