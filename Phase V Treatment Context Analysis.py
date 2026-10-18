"""
============================================================================
PHASE V: TREATMENT CONTEXT ANALYSIS AND NETWORK REWIRING (Python)
============================================================================
Project: Perturbation-Aware Causal Machine Learning Framework for 
         Glioblastoma Regulatory Network Discovery
Manuscript: A Perturbation-Aware Causal Machine Learning Framework Identifies
            Candidate Regulatory Hubs and Radiotherapy-Associated Network
            Rewiring in Glioblastoma
============================================================================
Description:
Comparative causal network analysis between irradiated (RT) and untreated
(noRT) glioblastoma microenvironments. Identifies radiotherapy-induced network
rewiring, edge gains/losses, hub shifts, and differential co-expression dynamics.
============================================================================
"""

# --- PHASE V START ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import networkx as nx
from scipy import stats
import os
import warnings
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

# Set random seed and plotting style
np.random.seed(42)
sns.set_style("ticks")

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
PHASE5_PATH = os.path.join(OUTPUT_PATH, "phase5")
FIGURE_PATH = os.path.join(OUTPUT_PATH, "figures", "phase5")

os.makedirs(PHASE5_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)

print("=" * 80)
print("PHASE V: TREATMENT CONTEXT ANALYSIS (RT vs noRT)")
print("=" * 80)
print()

# 5.1 Load Data
print("[1/6] Loading data...")

try:
    adata = sc.read_h5ad(os.path.join(PHASE2_PATH, "phase2_processed.h5ad"))
except FileNotFoundError:
    raise FileNotFoundError("Phase 2 output not found! Please run Phase 2 first.")

try:
    with open(os.path.join(PHASE3_PATH, "gene_names.txt"), 'r') as f:
        causal_genes = [line.strip() for line in f]
except FileNotFoundError:
    raise FileNotFoundError("Phase 3 gene list not found! Please run Phase 3 first.")

# Filter to causal genes (only keep those present in the AnnData)
valid_causal_genes = [g for g in causal_genes if g in adata.var_names]
adata_subset = adata[:, valid_causal_genes].copy()

print(f"    Loaded data: {adata_subset.n_obs} cells x {adata_subset.n_vars} causal genes")

# 5.2 Separate RT and noRT Cells
print("\n[2/6] Separating RT and noRT cells...")

if 'cond' not in adata_subset.obs:
    raise ValueError("The column 'cond' was not found in the dataset metadata. Cannot separate RT vs noRT.")

rt_mask = adata_subset.obs['cond'] == 'RT'
nort_mask = adata_subset.obs['cond'] == 'noRT'

adata_rt = adata_subset[rt_mask].copy()
adata_nort = adata_subset[nort_mask].copy()

print(f"    RT cells: {adata_rt.n_obs}")
print(f"    noRT cells: {adata_nort.n_obs}")

# 5.3 Build Condition-Specific Networks
print("\n[3/6] Building condition-specific networks...")

def build_correlation_network(adata_split, threshold=0.25):
    """Build network from Pearson correlation matrix with NaN handling."""
    X = adata_split.X.toarray() if hasattr(adata_split.X, 'toarray') else adata_split.X
    
    # Calculate correlation (handling zero-variance genes causing NaNs)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Threshold
    adj = (np.abs(corr_matrix) > threshold).astype(int)
    np.fill_diagonal(adj, 0)

    # Create graph
    G = nx.Graph()
    genes = adata_split.var_names.tolist()
    G.add_nodes_from(genes)

    # Add edges
    rows, cols = np.where(adj > 0)
    for i, j in zip(rows, cols):
        if i < j:  # upper triangle only (undirected)
            G.add_edge(genes[i], genes[j], weight=abs(corr_matrix[i, j]))

    return G, corr_matrix, adj

print("    Building RT network...")
G_rt, corr_rt, adj_rt = build_correlation_network(adata_rt, threshold=0.25)

print("    Building noRT network...")
G_nort, corr_nort, adj_nort = build_correlation_network(adata_nort, threshold=0.25)

print(f"    RT network: {G_rt.number_of_nodes()} nodes, {G_rt.number_of_edges()} edges")
print(f"    noRT network: {G_nort.number_of_nodes()} nodes, {G_nort.number_of_edges()} edges")

# 5.4 Calculate Network Differences
print("\n[4/6] Calculating network differences (Rewiring)...")

diff_corr = np.abs(corr_rt - corr_nort)
genes = adata_subset.var_names.tolist()
rewired_edges = []

# Identify rewired edges using fast array operations
diff_adj = adj_rt - adj_nort
rt_gain_rows, rt_gain_cols = np.where(diff_adj == 1)
rt_loss_rows, rt_loss_cols = np.where(diff_adj == -1)

for i, j in zip(rt_gain_rows, rt_gain_cols):
    if i < j:
        rewired_edges.append({
            'Gene1': genes[i], 'Gene2': genes[j], 'Type': 'RT_gain',
            'RT_correlation': corr_rt[i, j], 'noRT_correlation': corr_nort[i, j],
            'Difference': diff_corr[i, j]
        })

for i, j in zip(rt_loss_rows, rt_loss_cols):
    if i < j:
        rewired_edges.append({
            'Gene1': genes[i], 'Gene2': genes[j], 'Type': 'RT_loss',
            'RT_correlation': corr_rt[i, j], 'noRT_correlation': corr_nort[i, j],
            'Difference': diff_corr[i, j]
        })

rewired_df = pd.DataFrame(rewired_edges)
if len(rewired_df) > 0:
    rewired_df = rewired_df.sort_values('Difference', ascending=False)
    rewired_df.to_csv(os.path.join(PHASE5_PATH, "rewired_edges.csv"), index=False)
    print(f"    Total rewired edges: {len(rewired_df)}")
    print(f"    RT gains: {sum(rewired_df['Type'] == 'RT_gain')}")
    print(f"    RT losses: {sum(rewired_df['Type'] == 'RT_loss')}")
else:
    print("    No rewired edges found at this threshold.")

# Calculate node-level differences
node_stats = []
for gene in genes:
    rt_deg = G_rt.degree(gene) if gene in G_rt else 0
    nort_deg = G_nort.degree(gene) if gene in G_nort else 0
    rt_str = sum([d['weight'] for _, d in G_rt[gene].items()]) if gene in G_rt else 0
    nort_str = sum([d['weight'] for _, d in G_nort[gene].items()]) if gene in G_nort else 0

    node_stats.append({
        'Gene': gene,
        'RT_Degree': rt_deg,
        'noRT_Degree': nort_deg,
        'Degree_Difference': rt_deg - nort_deg,
        'RT_Strength': rt_str,
        'noRT_Strength': nort_str,
        'Strength_Difference': rt_str - nort_str
    })

node_stats_df = pd.DataFrame(node_stats).sort_values('Degree_Difference', ascending=False)
node_stats_df.to_csv(os.path.join(PHASE5_PATH, "node_condition_stats.csv"), index=False)
print(f"    Node stats saved for {len(node_stats_df)} genes.")

# 5.5 Pseudotime Trajectory Network Analysis
print("\n[5/6] Building pseudotime stage networks (Early, Middle, Late)...")
try:
    pt_df = pd.read_csv(os.path.join(PHASE2_PATH, "phase2_pseudotime.csv")).set_index('cell')
    adata_subset.obs['pt_bin'] = pt_df['pt_bin']
    
    print("    Building Early Network...")
    G_early, _, _ = build_correlation_network(adata_subset[adata_subset.obs['pt_bin'] == 'Early'].copy(), threshold=0.25)
    
    print("    Building Middle Network...")
    G_middle, _, _ = build_correlation_network(adata_subset[adata_subset.obs['pt_bin'] == 'Middle'].copy(), threshold=0.25)
    
    print("    Building Late Network...")
    G_late, _, _ = build_correlation_network(adata_subset[adata_subset.obs['pt_bin'] == 'Late'].copy(), threshold=0.25)
    
    print(f"    Early: {G_early.number_of_nodes()} nodes | {G_early.number_of_edges()} edges")
    print(f"    Middle: {G_middle.number_of_nodes()} nodes | {G_middle.number_of_edges()} edges")
    print(f"    Late: {G_late.number_of_nodes()} nodes | {G_late.number_of_edges()} edges")
    
    pt_summary = pd.DataFrame({
        'Stage': ['Early', 'Middle', 'Late'],
        'Nodes': [sum(d > 0 for n, d in G_early.degree()), sum(d > 0 for n, d in G_middle.degree()), sum(d > 0 for n, d in G_late.degree())],
        'Edges': [G_early.number_of_edges(), G_middle.number_of_edges(), G_late.number_of_edges()]
    })
    pt_summary.to_csv(os.path.join(PHASE5_PATH, "phase5_pseudotime_GRN_summary.csv"), index=False)
    
except Exception as e:
    print(f"    Warning: Could not process pseudotime networks. Error: {e}")
    G_early, G_middle, G_late = nx.Graph(), nx.Graph(), nx.Graph()

# 5.6 Generate Treatment Context Figures
print("\n[6/6] Generating treatment context figures...")

# Figure 5A: Network Edge Comparison
fig, ax = plt.subplots(figsize=(8, 6))
networks = ['RT Network', 'noRT Network']
edge_counts = [G_rt.number_of_edges(), G_nort.number_of_edges()]
colors = ['#D55E00', '#0072B2']

bars = ax.bar(networks, edge_counts, color=colors, edgecolor='white', linewidth=2, width=0.6)
ax.set_ylabel('Number of Edges', fontsize=12, fontweight='bold')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + max(edge_counts)*0.02,
            f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'Fig5A_Network_Comparison.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'Fig5A_Network_Comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 5A-2: Visual Network Topology (Node-and-Edge)
def draw_beautiful_network(G, title, ax):
    # Remove boring Ribosomal and Mitochondrial housekeeping genes from visualization
    interesting_nodes = [n for n in G.nodes() if not (n.capitalize().startswith('Rp') or n.capitalize().startswith('Mt-'))]
    G_filtered = G.subgraph(interesting_nodes)
    if G_filtered.number_of_edges() > 100:
        edges = sorted(G_filtered.edges(data=True), key=lambda x: abs(x[2].get('weight', 0)), reverse=True)[:100]
        G_sub = nx.DiGraph(edges)
    else:
        G_sub = G_filtered
    pos = nx.spring_layout(G_sub, k=2.5, iterations=150, seed=42)
    degrees = dict(G_sub.degree())
    node_sizes = [v * 150 + 100 for k, v in degrees.items()]
    hub_threshold = sorted(degrees.values(), reverse=True)[min(5, len(degrees)-1)] if len(degrees) > 0 else 0
    node_colors = ['#D55E00' if v >= hub_threshold else '#56B4E9' for k, v in degrees.items()]
    nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes, node_color=node_colors, edgecolors='white', ax=ax)
    edge_weights = [abs(d.get('weight', 0.5)) * 2 for u, v, d in G_sub.edges(data=True)]
    if len(G_sub.edges()) > 0:
        nx.draw_networkx_edges(G_sub, pos, width=edge_weights, arrowsize=15, edge_color='#BDBDBD', alpha=0.7, ax=ax, connectionstyle="arc3,rad=0.1")
    labels = {n: n for n in G_sub.nodes() if degrees[n] >= hub_threshold or G_sub.number_of_nodes() < 30}
    nx.draw_networkx_labels(G_sub, pos, labels=labels, font_size=11, font_weight='bold', ax=ax)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
draw_beautiful_network(G_rt, "Radiotherapy (RT) Network Structure", ax1)
draw_beautiful_network(G_nort, "Untreated (NoRT) Network Structure", ax2)
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'Fig5A_Visual_Network_Topology.png'), dpi=300, bbox_inches='tight')
plt.close()


# Figure 9A: Degree Distribution Histogram
fig1, ax1 = plt.subplots(figsize=(7, 5))

rt_degrees = [d for n, d in G_rt.degree()]
nort_degrees = [d for n, d in G_nort.degree()]

ax1.hist(rt_degrees, bins=20, alpha=0.7, label='RT', color='#D55E00', edgecolor='white')
ax1.hist(nort_degrees, bins=20, alpha=0.7, label='noRT', color='#0072B2', edgecolor='white')
ax1.set_xlabel('Node Degree', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.legend(frameon=False)

fig1.tight_layout()
fig1.savefig(os.path.join(FIGURE_PATH, 'Fig9A_Degree_Distribution.pdf'), dpi=300, bbox_inches='tight')
fig1.savefig(os.path.join(FIGURE_PATH, 'Fig9A_Degree_Distribution.png'), dpi=300, bbox_inches='tight')
plt.close(fig1)

# Figure 9B: RT vs noRT Degree Scatter
fig2, ax2 = plt.subplots(figsize=(7, 5))

rt_degree_dict = dict(G_rt.degree())
nort_degree_dict = dict(G_nort.degree())
rt_vals = [rt_degree_dict.get(g, 0) for g in genes]
nort_vals = [nort_degree_dict.get(g, 0) for g in genes]
max_val = max(max(rt_vals), max(nort_vals), 1)

ax2.scatter(nort_vals, rt_vals, alpha=0.6, color='#CC79A7')
ax2.plot([0, max_val], [0, max_val], 'k--', lw=1.5, alpha=0.7)
ax2.set_xlabel('noRT Degree', fontsize=12, fontweight='bold')
ax2.set_ylabel('RT Degree', fontsize=12, fontweight='bold')

# Annotate top genes gaining and losing connectivity
for gene in ["Hsph1", "Nr4a1", "Junb"]:
    if gene in genes:
        x = nort_degree_dict.get(gene, 0)
        y = rt_degree_dict.get(gene, 0)
        ax2.annotate(gene, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold', color='black')

for gene in ["Akr1b3", "Sp110"]:
    if gene in genes:
        x = nort_degree_dict.get(gene, 0)
        y = rt_degree_dict.get(gene, 0)
        ax2.annotate(gene, (x, y), xytext=(5, -12), textcoords='offset points', fontsize=9, fontweight='bold', color='black')

fig2.tight_layout()
fig2.savefig(os.path.join(FIGURE_PATH, 'Fig9B_RT_vs_noRT_Scatter.pdf'), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(FIGURE_PATH, 'Fig9B_RT_vs_noRT_Scatter.png'), dpi=300, bbox_inches='tight')
plt.close(fig2)

# Figure 5C: Top Differential Genes
fig, ax = plt.subplots(figsize=(10, 8))
top_diff = pd.concat([node_stats_df.head(10), node_stats_df.tail(10)]).sort_values('Degree_Difference')
colors = ['#D55E00' if x > 0 else '#0072B2' for x in top_diff['Degree_Difference']]

ax.barh(range(len(top_diff)), top_diff['Degree_Difference'], color=colors, edgecolor='white')
ax.set_yticks(range(len(top_diff)))
ax.set_yticklabels(top_diff['Gene'], fontsize=11, fontstyle='italic')
ax.set_xlabel('Degree Difference (RT - noRT)', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#D55E00", label="Gained in RT"), Patch(facecolor="#0072B2", label="Lost in RT")]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10, frameon=False)

plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'Fig5C_Differential_Genes.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'Fig5C_Differential_Genes.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 5D: Pseudotime GRN Complexity
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
stages = ['Early', 'Middle', 'Late']

# Calculate metrics
node_counts = [sum(d > 0 for n, d in G_early.degree()), 
               sum(d > 0 for n, d in G_middle.degree()), 
               sum(d > 0 for n, d in G_late.degree())]
edge_counts = [G_early.number_of_edges(), G_middle.number_of_edges(), G_late.number_of_edges()]
densities = [nx.density(G_early) if G_early.number_of_nodes() > 1 else 0,
             nx.density(G_middle) if G_middle.number_of_nodes() > 1 else 0,
             nx.density(G_late) if G_late.number_of_nodes() > 1 else 0]

colors = ['#56B4E9', '#F0E442', '#D55E00']

# Panel 1: Nodes
bars0 = axes[0].bar(stages, node_counts, color=colors, edgecolor='none', width=0.6)
axes[0].set_ylabel('Network Nodes', fontsize=12, fontweight='bold')
for bar in bars0:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + max(node_counts)*0.01,
            f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel 2: Edges
bars1 = axes[1].bar(stages, edge_counts, color=colors, edgecolor='none', width=0.6)
axes[1].set_ylabel('Network Edges', fontsize=12, fontweight='bold')
for bar in bars1:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + max(edge_counts)*0.01,
            f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel 3: Density
bars2 = axes[2].bar(stages, densities, color=colors, edgecolor='none', width=0.6)
axes[2].set_ylabel('Network Density', fontsize=12, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height + max(densities)*0.01,
            f'{height:.5f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'Fig5D_Pseudotime_GRN_Complexity.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'Fig5D_Pseudotime_GRN_Complexity.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 5E: Rewired Edges Distribution
if len(rewired_df) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    type_counts = rewired_df['Type'].value_counts()
    
    ordered_types = []
    if 'RT_gain' in type_counts: ordered_types.append('RT_gain')
    if 'RT_loss' in type_counts: ordered_types.append('RT_loss')
    
    colors = ['#009E73' if t == 'RT_gain' else '#D55E00' for t in ordered_types]
    vals = [type_counts[t] for t in ordered_types]
    labels = ['Gained in RT' if t == 'RT_gain' else 'Lost in RT' for t in ordered_types]
    
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=2, width=0.6)
    ax.set_ylabel('Number of Rewired Edges', fontsize=12, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(vals)*0.02,
                f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_PATH, 'Fig5E_Rewired_Edges.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURE_PATH, 'Fig5E_Rewired_Edges.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Figure 5F: Network Visualization Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Get top 30 genes by ABSOLUTE degree difference to show the most rewired genes
top_genes_viz = node_stats_df.assign(abs_diff=node_stats_df['Degree_Difference'].abs()).nlargest(30, 'abs_diff')['Gene'].tolist()

G_rt_viz = G_rt.subgraph(top_genes_viz).copy()
G_nort_viz = G_nort.subgraph(top_genes_viz).copy()

# Add empty nodes to noRT if they are in top_genes_viz so they show up!
for g in top_genes_viz:
    if g not in G_nort_viz:
        G_nort_viz.add_node(g)

pos = nx.spring_layout(nx.compose(G_rt_viz, G_nort_viz), k=1.5, seed=42)

# Set node sizes dynamically (minimum size 100)
rt_sizes = [G_rt_viz.degree(n)*50 + 100 for n in G_rt_viz.nodes()]
nort_sizes = [G_nort_viz.degree(n)*50 + 100 for n in G_nort_viz.nodes()]

# RT network
nx.draw_networkx_nodes(G_rt_viz, pos, node_color='#D55E00', edgecolors='white', alpha=0.9, ax=axes[0], node_size=rt_sizes)
nx.draw_networkx_edges(G_rt_viz, pos, alpha=0.4, ax=axes[0], edge_color='gray')
nx.draw_networkx_labels(G_rt_viz, pos, font_size=9, font_weight='bold', ax=axes[0])
axes[0].set_title("Radiotherapy (RT)", fontsize=14, fontweight='bold')
axes[0].axis('off')

# noRT network (use same positions to show rewiring clearly)
nx.draw_networkx_nodes(G_nort_viz, pos, node_color='#0072B2', edgecolors='white', alpha=0.9, ax=axes[1], node_size=nort_sizes)
nx.draw_networkx_edges(G_nort_viz, pos, alpha=0.4, ax=axes[1], edge_color='gray')
nx.draw_networkx_labels(G_nort_viz, pos, font_size=9, font_weight='bold', ax=axes[1])
axes[1].set_title("Untreated (noRT)", fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'Fig5F_Network_Comparison.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'Fig5F_Network_Comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 5G: Differential Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 10))
indices = [genes.index(g) for g in top_genes_viz if g in genes]
sub_genes = [g for g in top_genes_viz if g in genes]
sub_diff_corr = diff_corr[np.ix_(indices, indices)]
sns.heatmap(sub_diff_corr, xticklabels=sub_genes, yticklabels=sub_genes, cmap='Reds', cbar_kws={'label': 'Absolute Delta Correlation (|Δr|)'}, ax=ax)
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'Fig5G_Differential_Correlation_Heatmap.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'Fig5G_Differential_Correlation_Heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save networks
nx.write_gml(G_rt, os.path.join(PHASE5_PATH, "network_rt.gml"))
nx.write_gml(G_nort, os.path.join(PHASE5_PATH, "network_nort.gml"))

# 5.7 Summary Report
summary_report = pd.DataFrame({
    'Metric': ['RT Cells', 'noRT Cells', 'RT Edges', 'noRT Edges',
               'Rewired Edges', 'RT Gains', 'RT Losses',
               'Top RT Gained Hub', 'Top RT Lost Hub',
               'Pseudotime Early Nodes', 'Pseudotime Early Edges',
               'Pseudotime Middle Nodes', 'Pseudotime Middle Edges',
               'Pseudotime Late Nodes', 'Pseudotime Late Edges'],
    'Value': [adata_rt.n_obs, adata_nort.n_obs,
              G_rt.number_of_edges(), G_nort.number_of_edges(),
              len(rewired_df) if len(rewired_df) > 0 else 0,
              sum(rewired_df['Type'] == 'RT_gain') if len(rewired_df) > 0 else 0,
              sum(rewired_df['Type'] == 'RT_loss') if len(rewired_df) > 0 else 0,
              node_stats_df.iloc[0]['Gene'] if len(node_stats_df)>0 else "None",
              node_stats_df.iloc[-1]['Gene'] if len(node_stats_df)>0 else "None",
              sum(d > 0 for n, d in G_early.degree()), G_early.number_of_edges(),
              sum(d > 0 for n, d in G_middle.degree()), G_middle.number_of_edges(),
              sum(d > 0 for n, d in G_late.degree()), G_late.number_of_edges()]
})
summary_report.to_csv(os.path.join(PHASE5_PATH, "phase5_summary.csv"), index=False)

print("\n" + "=" * 80)
print("[DONE] PHASE V COMPLETE")
print(f"Output saved to: {PHASE5_PATH}")
print(f"Figures saved to: {FIGURE_PATH}")
print("=" * 80)

# --- PHASE V END ---
